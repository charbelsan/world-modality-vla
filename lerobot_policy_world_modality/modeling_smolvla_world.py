from __future__ import annotations

from collections import deque
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.smolvla.modeling_smolvla import (  # type: ignore
    SmolVLAPolicy,
    VLAFlowMatching,
    make_att_2d_masks,
)
from lerobot.utils.constants import ACTION

from world_modality.model import GatedCrossAttention, Prophet
from world_modality.vision import VisionEncoder

from .configuration_smolvla_world import SmolVLAWorldConfig


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f = x.float()
    m = mask.to(dtype=x_f.dtype)
    return (x_f * m).sum() / (m.sum().clamp_min(eps))


def compute_world_loss_continuous_masked(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    pred_f = pred.float()
    tgt_f = target.float()
    pred_n = pred_f / pred_f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tgt_n = tgt_f / tgt_f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos = (pred_n * tgt_n).sum(dim=-1)  # [B, K]
    loss_per = 1.0 - cos
    return _masked_mean(loss_per, valid)


@torch.no_grad()
def compute_world_cosine_masked(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    pred_f = pred.float()
    tgt_f = target.float()
    pred_n = pred_f / pred_f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tgt_n = tgt_f / tgt_f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos = (pred_n * tgt_n).sum(dim=-1)  # [B, K]
    return float(_masked_mean(cos, valid).cpu().item())


class WorldInjectedVLAFlowMatching(VLAFlowMatching):
    """VLAFlowMatching with Model-F style world injection into action expert only.

    Important: this class is constructed by *wrapping* an existing VLAFlowMatching instance
    to avoid allocating a second VLM/expert copy.
    """

    def __init__(
        self,
        base: VLAFlowMatching,
        *,
        future_dim: int,
        inject_num_heads: int,
        gate_init: float,
        enable_world_injection: bool,
        track_stats: bool,
    ):
        # Do not call VLAFlowMatching.__init__ (it would allocate a second VLM).
        nn.Module.__init__(self)

        # Transfer all modules/attributes from base.
        self.config = base.config
        self.vlm_with_expert = base.vlm_with_expert
        self.state_proj = base.state_proj
        self.action_in_proj = base.action_in_proj
        self.action_out_proj = base.action_out_proj
        self.action_time_mlp_in = base.action_time_mlp_in
        self.action_time_mlp_out = base.action_time_mlp_out

        self.fake_image_token = base.fake_image_token
        self.global_image_token = base.global_image_token
        self.global_image_start_token = base.global_image_start_token
        self.add_image_special_tokens = base.add_image_special_tokens
        self.image_end_token = base.image_end_token
        self.prefix_length = base.prefix_length
        self.rtc_processor = getattr(base, "rtc_processor", None)

        self._world_memory: Optional[torch.Tensor] = None
        self.world_inject = None
        if bool(enable_world_injection):
            self.world_inject = GatedCrossAttention(
                d_model=self.vlm_with_expert.expert_hidden_size,
                n_heads=int(inject_num_heads),
                future_dim=int(future_dim),
                track_stats=bool(track_stats),
            )
            with torch.no_grad():
                self.world_inject.gate.copy_(torch.tensor(float(gate_init)))

    def set_world_memory(self, world_memory: Optional[torch.Tensor]) -> None:
        self._world_memory = world_memory

    def world_gate_value(self) -> float:
        if self.world_inject is None:
            return 0.0
        return float(torch.tanh(self.world_inject.gate).detach().cpu().item())

    def _inject(self, suffix_out: torch.Tensor) -> torch.Tensor:
        if self.world_inject is None or self._world_memory is None:
            return suffix_out
        # GatedCrossAttention is float32; cast world_memory if needed (encoder may output float16).
        mem = self._world_memory.float() if self._world_memory.dtype != suffix_out.dtype else self._world_memory
        return self.world_inject(suffix_out, mem)

    def world_last_stats(self) -> dict[str, float]:
        if self.world_inject is None:
            return {}
        return self.world_inject.last_stats()

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        # Copy of VLAFlowMatching.forward with an extra injection before action_out_proj.
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        suffix_out = self._inject(suffix_out)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        # Copy of VLAFlowMatching.denoise_step with injection before action_out_proj.
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        suffix_out = self._inject(suffix_out)
        v_t = self.action_out_proj(suffix_out)
        return v_t


class SmolVLAWorldPolicy(SmolVLAPolicy):
    """SmolVLA policy augmented with world modality memory injection."""

    config_class = SmolVLAWorldConfig
    name = "smolvla_world"

    def __init__(self, config: SmolVLAWorldConfig):
        # If we're asked to init from a pretrained SmolVLA policy, first align this config's
        # *architecture* fields to the init checkpoint's config so weight loading is shape-consistent
        # (e.g., smolvla_base vs smolvla_libero differ in depth/width).
        init_policy: Optional[SmolVLAPolicy] = None
        if getattr(config, "init_from_policy_path", ""):
            init_policy = SmolVLAPolicy.from_pretrained(str(config.init_from_policy_path))
            base_cfg = init_policy.config
            # Avoid overriding experiment wiring knobs.
            skip = {
                "device",
                "push_to_hub",
            }
            for k, v in vars(base_cfg).items():
                if k in skip:
                    continue
                if hasattr(config, k):
                    try:
                        setattr(config, k, v)
                    except Exception:
                        # Best-effort; ignore read-only / computed fields.
                        pass

        # NOTE: SmolVLAPolicy.__init__ calls self.reset(); initialize our state first so the
        # override doesn't access missing attributes during base init.
        self.config: SmolVLAWorldConfig = config
        self._latent_dim: int = int(config.world_latent_dim)
        self.prophet: Optional[Prophet] = None
        self.world_encoder: Optional[VisionEncoder] = None
        self._world_hist: deque[torch.Tensor] = deque(maxlen=int(config.context_frames))

        super().__init__(config)

        if self.config.enable_world_predictor:
            self.prophet = Prophet(
                emb_dim=self._latent_dim,
                hidden_dim=self._latent_dim,
                future_horizon=int(self.config.future_offset),
                n_layers=int(self.config.prophet_layers),
                n_heads=int(self.config.prophet_heads),
                dropout=float(self.config.prophet_dropout),
            )

        base_model = self.model
        self.model = WorldInjectedVLAFlowMatching(
            base_model,
            future_dim=self._latent_dim,
            inject_num_heads=int(self.config.world_inject_num_heads),
            gate_init=float(self.config.world_gate_init),
            enable_world_injection=bool(self.config.enable_world_injection),
            track_stats=bool(self.config.log_attn_stats),
        )

        self._grad_accum: dict[str, torch.Tensor] = {}
        self._last_grad_norms: dict[str, float] = {}
        if self.config.log_grad_stats:
            self._register_grad_hooks()

        if init_policy is not None:
            incompatible = self.load_state_dict(init_policy.state_dict(), strict=False)
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            if bool(self.config.init_from_strict) and unexpected:
                raise RuntimeError(f"Unexpected keys when loading init policy: {unexpected[:50]}")
            if bool(self.config.init_from_strict):
                if any(k.startswith("model.") for k in missing):
                    raise RuntimeError(
                        "Init policy load missed core SmolVLA keys. Check that smolvla_world config "
                        "(num layers/width) matches the init policy."
                    )

    def reset(self):
        super().reset()
        self._world_hist.clear()

    def _ensure_world_modules(self, detected_latent_dim: int) -> None:
        """Validate that world modules match the detected latent dimension from data.

        This is called on first forward pass with world latents to ensure the configured
        world_latent_dim matches what's actually in the cached latents.
        """
        if detected_latent_dim != self._latent_dim:
            raise ValueError(
                f"World latent dimension mismatch: config has world_latent_dim={self._latent_dim}, "
                f"but cached latents have dim={detected_latent_dim}. "
                f"Update --policy.world_latent_dim to match your cached latents."
            )

    def _register_grad_hooks(self) -> None:
        # Compute grad norms for world modules without modifying the LeRobot training loop.
        # Hooks accumulate grad^2 during backward; we report them on the *next* forward call.
        def make_hook(name: str):
            def _hook(grad: torch.Tensor):
                if grad is None:
                    return
                acc = self._grad_accum.get(name)
                val = grad.detach().float().pow(2).sum()
                self._grad_accum[name] = val if acc is None else (acc + val)
            return _hook

        if isinstance(self.model, WorldInjectedVLAFlowMatching) and self.model.world_inject is not None:
            for p in self.model.world_inject.parameters():
                if p.requires_grad:
                    p.register_hook(make_hook("grad2_world_inject"))
        if self.prophet is not None:
            for p in self.prophet.parameters():
                if p.requires_grad:
                    p.register_hook(make_hook("grad2_prophet"))

    def _build_world_memory_train(
        self, z_hist: torch.Tensor, z_future: torch.Tensor, valid: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, float]:
        mode = str(self.config.world_memory_mode_train)
        if int(z_hist.shape[-1]) != int(self._latent_dim):
            raise ValueError(
                f"world_z_hist dim={int(z_hist.shape[-1])} does not match policy.world_latent_dim={self._latent_dim}. "
                "Fix by setting --policy.world_latent_dim and recomputing cache consistently."
            )
        z_current = z_hist[:, -1:, :]

        if self.prophet is None:
            z_pred = torch.zeros_like(z_future)
        else:
            # Prophet is float32; cast if cached latents are float16.
            z_pred = self.prophet(z_hist.float())

        if self.config.delta_prediction:
            z_pred_abs = z_current + z_pred
            target_delta = z_future - z_current
            world_loss = compute_world_loss_continuous_masked(z_pred, target_delta, valid)
            world_cos = float(compute_world_cosine_masked(z_pred, target_delta, valid))
        else:
            z_pred_abs = z_pred
            world_loss = compute_world_loss_continuous_masked(z_pred, z_future, valid)
            world_cos = float(compute_world_cosine_masked(z_pred, z_future, valid))

        if mode == "oracle":
            mem = z_future
        elif mode == "zero":
            mem = torch.zeros_like(z_future)
        elif mode == "shuffle":
            perm = torch.randperm(z_future.shape[0], device=z_future.device)
            mem = z_future[perm]
        elif mode == "random":
            mem = torch.randn_like(z_future)
        else:  # "pred"
            mem = z_pred_abs

        return mem, world_loss, world_cos

    @torch.no_grad()
    def _build_world_memory_rollout(self, images: list[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.world_encoder is None:
            self.world_encoder = VisionEncoder(
                model_name=str(self.config.world_vision_model_name),
                device=str(self.config.device),
                dtype="float16",
            )
        if self.prophet is None:
            return None
        if len(images) == 0:
            return None

        # Images are in [-1, 1] (SigLIP expects that); convert back to [0, 1] for world encoder.
        img = images[0]
        img01 = ((img + 1.0) * 0.5).clamp(0.0, 1.0)

        z_t = self.world_encoder.encode(img01)  # [B, D]
        if int(z_t.shape[-1]) != int(self._latent_dim):
            raise ValueError(
                f"Online world encoder dim={int(z_t.shape[-1])} does not match policy.world_latent_dim={self._latent_dim}. "
                "Fix by setting --policy.world_latent_dim and/or --policy.world_vision_model_name."
            )
        self._world_hist.append(z_t)
        while len(self._world_hist) < int(self.config.context_frames):
            self._world_hist.appendleft(z_t)
        z_hist = torch.stack(list(self._world_hist), dim=1)  # [B, T, D]

        mode = str(self.config.world_memory_mode_rollout)
        if mode == "zero":
            return torch.zeros(
                (z_hist.shape[0], int(self.config.future_offset), self._latent_dim),
                device=z_hist.device,
                dtype=z_hist.dtype,
            )
        if mode == "random":
            return torch.randn(
                (z_hist.shape[0], int(self.config.future_offset), self._latent_dim),
                device=z_hist.device,
                dtype=z_hist.dtype,
            )

        # Prophet is in float32; cast z_hist if needed (world encoder may output float16).
        z_hist_f = z_hist.float()
        z_pred = self.prophet(z_hist_f)
        if self.config.delta_prediction:
            z_current = z_hist[:, -1:, :]
            return z_current + z_pred
        return z_pred

    def forward(self, batch: dict[str, Tensor], noise=None, time=None):
        # Report previous-step grad norms (hooks fill them during backward).
        if self.config.log_grad_stats and self._grad_accum:
            out: dict[str, float] = {}
            for k, v in list(self._grad_accum.items()):
                out[k.replace("grad2_", "grad_")] = float(v.sqrt().detach().cpu().item())
            self._last_grad_norms = out
            self._grad_accum = {}

        # Cached latents are provided by the processor during offline training.
        z_hist = batch.get("world_z_hist", None)
        z_future = batch.get("world_z_future", None)
        valid = batch.get("world_future_valid", None)

        world_valid_frac = 0.0
        world_mem_norm = 0.0
        world_z_hist_norm = 0.0
        world_z_future_norm = 0.0

        world_loss = torch.tensor(0.0, device=batch[ACTION].device)
        world_cos = 0.0

        if z_hist is not None and z_future is not None:
            self._ensure_world_modules(int(z_hist.shape[-1]))
            assert valid is not None, "world_future_valid missing (expected from processor)"
            mem, world_loss, world_cos = self._build_world_memory_train(z_hist, z_future, valid)
            world_valid_frac = float(valid.float().mean().detach().cpu().item())
            world_z_hist_norm = float(z_hist[:, -1, :].float().norm(dim=-1).mean().detach().cpu().item())
            world_z_future_norm = float(z_future.float().norm(dim=-1).mean().detach().cpu().item())
            if mem is not None:
                world_mem_norm = float(mem.float().norm(dim=-1).mean().detach().cpu().item())
            if isinstance(self.model, WorldInjectedVLAFlowMatching):
                self.model.set_world_memory(mem)
        else:
            if isinstance(self.model, WorldInjectedVLAFlowMatching):
                self.model.set_world_memory(None)

        action_loss, metrics = super().forward(batch, noise=noise, time=time)

        total = action_loss
        if float(self.config.lambda_world) > 0 and z_hist is not None and z_future is not None:
            total = total + float(self.config.lambda_world) * world_loss

        metrics["world_loss"] = float(world_loss.detach().cpu().item())
        metrics["world_cos"] = float(world_cos)
        metrics["world_valid_frac"] = float(world_valid_frac)
        metrics["world_mem_norm"] = float(world_mem_norm)
        metrics["world_z_hist_norm"] = float(world_z_hist_norm)
        metrics["world_z_future_norm"] = float(world_z_future_norm)
        if isinstance(self.model, WorldInjectedVLAFlowMatching):
            metrics["world_gate"] = self.model.world_gate_value()
            if self.config.log_attn_stats:
                stats = self.model.world_last_stats()
                for k, v in stats.items():
                    metrics[f"world_{k}"] = float(v)
        if self._last_grad_norms:
            metrics.update(self._last_grad_norms)
        metrics["loss_total"] = float(total.detach().cpu().item())
        return total, metrics

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs):
        if isinstance(self.model, WorldInjectedVLAFlowMatching):
            # Do-no-harm: if the gate is effectively closed, don't even compute the world memory.
            # This avoids expensive vision/proph runs and prevents NaN/Inf in that path from
            # contaminating actions.
            gate = float(self.model.world_gate_value())
            if (not self.config.enable_world_injection) or abs(gate) < 1e-6:
                self.model.set_world_memory(None)
            else:
                images, _ = self.prepare_images(batch)
                mem = self._build_world_memory_rollout(images)
                if mem is not None and (not torch.isfinite(mem).all()):
                    mem = None
                self.model.set_world_memory(mem)
        return super().select_action(batch, noise=noise, **kwargs)

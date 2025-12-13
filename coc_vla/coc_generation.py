from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


try:
    # Newer LeRobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
except Exception:  # pragma: no cover
    # Older LeRobot
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate episode-level Chain-of-Causality (CoC) labels.")
    parser.add_argument("--dataset_name", type=str, required=True, help="LeRobot dataset name (e.g. HuggingFaceVLA/libero).")
    parser.add_argument("--image_key", type=str, default="rgb")
    parser.add_argument("--instruction_key", type=str, default="instruction", help="Key for instruction text, if available.")
    parser.add_argument("--episode_id_key", type=str, default="episode_id", help="Key used to group timesteps into episodes.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"])
    parser.add_argument("--train_val_split", type=float, default=0.9, help="Episode split ratio for train/val when split!=all.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--resume", action="store_true", help="Append and skip already-labeled episode_ids.")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Open VLM model on HF.")
    parser.add_argument(
        "--backend",
        type=str,
        default="qwen3-vl",
        choices=["qwen3-vl", "qwen2.5-vl", "auto"],
        help="Which transformers backend to use.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_episodes", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--prompt_file", type=str, default="", help="Optional prompt template file.")

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    return parser.parse_args()


def read_prompt(prompt_file: str) -> str:
    if prompt_file:
        with open(prompt_file, "r") as f:
            return f.read().strip()
    default_path = os.path.join(os.path.dirname(__file__), "prompts", "coc_episode_prompt.txt")
    if os.path.exists(default_path):
        with open(default_path, "r") as f:
            return f.read().strip()
    return (
        "Write a Chain-of-Causality explanation in 4–7 numbered steps. "
        "Each step must say what the robot does and why it helps achieve the goal."
    )


def to_pil_image(x: Any) -> Optional[Image.Image]:
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, torch.Tensor):
        img = x
        if img.dim() == 3 and img.shape[0] in (1, 3):
            if img.dtype != torch.uint8:
                img = (img * 255).clamp(0, 255).to(torch.uint8)
            img_np = img.permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(img_np)
    if isinstance(x, np.ndarray):
        if x.ndim == 3:
            return Image.fromarray(x.astype(np.uint8))
    return None


def build_episode_index(ds, episode_id_key: str) -> Dict[int, List[int]]:
    episode_to_indices: Dict[int, List[int]] = {}
    for idx in range(len(ds)):
        step = ds[idx]
        ep_id = int(step.get(episode_id_key, 0))
        episode_to_indices.setdefault(ep_id, []).append(idx)
    return episode_to_indices


def filter_episode_ids(episode_ids: List[int], split: str, train_val_split: float) -> List[int]:
    if split == "all":
        return episode_ids
    split_idx = int(len(episode_ids) * train_val_split)
    if split == "train":
        return episode_ids[:split_idx]
    return episode_ids[split_idx:]


def load_done_episode_ids(output_jsonl: str) -> set[int]:
    done: set[int] = set()
    if not os.path.exists(output_jsonl):
        return done
    with open(output_jsonl, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(int(obj["episode_id"]))
            except Exception:
                continue
    return done


def extract_images_from_messages(messages: List[Dict[str, Any]]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                img = item.get("image")
                if isinstance(img, Image.Image):
                    images.append(img)
    return images


@dataclass
class VlmClient:
    processor: Any
    model: Any
    device: torch.device

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = extract_images_from_messages(messages)

        inputs = self.processor(
            text=[text],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        out_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()


def load_vlm(model_name: str, backend: str, device: str) -> VlmClient:
    from transformers import AutoProcessor

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_name)

    model = None
    if backend == "qwen3-vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Qwen3VLForConditionalGeneration not available. "
                "Upgrade transformers (>=4.56) or use --backend qwen2.5-vl/auto."
            ) from e
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    elif backend == "qwen2.5-vl":
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Qwen2_5_VLForConditionalGeneration not available. "
                "Upgrade transformers or use --backend auto."
            ) from e
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    else:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    model = model.to(torch_device)
    model.eval()
    return VlmClient(processor=processor, model=model, device=torch_device)


def build_messages(instruction: str, frames: List[Image.Image], prompt: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for img in frames:
        content.append({"type": "image", "image": img})
    content.append(
        {
            "type": "text",
            "text": f"Task instruction: {instruction}\n\n{prompt}\n",
        }
    )
    return [{"role": "user", "content": content}]


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    prompt = read_prompt(args.prompt_file)
    client = load_vlm(args.model_name, args.backend, args.device)

    ds = LeRobotDataset(args.dataset_name)
    episode_to_indices = build_episode_index(ds, args.episode_id_key)
    episode_ids = sorted(episode_to_indices.keys())
    episode_ids = filter_episode_ids(episode_ids, args.split, args.train_val_split)

    done = load_done_episode_ids(args.output_jsonl) if args.resume else set()

    mode = "a" if (args.resume and os.path.exists(args.output_jsonl)) else "w"
    with open(args.output_jsonl, mode) as out_f:
        n = 0
        for ep_id in tqdm(episode_ids, desc=f"CoC ({args.split})"):
            if ep_id in done:
                continue
            if args.max_episodes and n >= args.max_episodes:
                break

            indices = episode_to_indices[ep_id]
            if not indices:
                continue

            step0 = ds[indices[0]]
            instruction = str(step0.get(args.instruction_key, ""))

            i0 = indices[0]
            im = indices[len(indices) // 2]
            ie = indices[-1]

            frames: List[Image.Image] = []
            for idx in [i0, im, ie]:
                img = ds[idx].get(args.image_key)
                pil = to_pil_image(img)
                if pil is not None:
                    frames.append(pil)

            messages = build_messages(instruction, frames, prompt)
            coc_text = ""
            error: Optional[str] = None
            try:
                coc_text = client.generate(
                    messages=messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            except Exception as e:
                error = repr(e)

            obj = {
                "episode_id": ep_id,
                "instruction": instruction,
                "keyframe_indices": [i0, im, ie],
                "model_name": args.model_name,
                "backend": args.backend,
                "coc_text": coc_text,
                "error": error,
            }
            out_f.write(json.dumps(obj) + "\n")
            out_f.flush()
            n += 1


if __name__ == "__main__":
    main()


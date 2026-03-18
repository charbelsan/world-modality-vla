#!/usr/bin/env bash
set -euo pipefail

# Queue follow-up experiments behind the currently running P5 jobs.
#
# Modes:
#   after_gpu6_current -> wait for gpu6_h100_queue.pid, then ablate E2_wrist
#   after_gpu7_current -> wait for gpu7_h100_queue.pid, then eval F2 and run F3b

MODE=${MODE:-""}
GPU_ID=${GPU_ID:-0}
ROOT_DIR=${ROOT_DIR:-$(pwd)}
VENV_PATH=${VENV_PATH:-/opt/dlami/nvme/.venvs/world-modality-vla-ss/bin/activate}
HF_ROOT=${HF_ROOT:-${ROOT_DIR}/.hf_cache}

export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

export HF_HOME=${HF_HOME:-${HF_ROOT}}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_ROOT}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_ROOT}/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_ROOT}/transformers}

export EVAL_TASK=${EVAL_TASK:-libero_spatial}
export EVAL_EPISODES=${EVAL_EPISODES:-500}
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-5}
export EVAL_N_ACTION_STEPS=${EVAL_N_ACTION_STEPS:-10}
export BATCH_SIZE=${BATCH_SIZE:-192}
export NUM_WORKERS=${NUM_WORKERS:-48}
export STEPS=${STEPS:-25000}
export SEED=${SEED:-0}

resolve_checkpoint_dir() {
  local base_dir="$1"
  local steps="$2"
  local padded
  padded="$(printf "%06d" "${steps}")"
  if [[ -d "${base_dir}/checkpoints/${padded}/pretrained_model" ]]; then
    echo "${base_dir}/checkpoints/${padded}/pretrained_model"
  else
    echo "${base_dir}/checkpoints/${steps}/pretrained_model"
  fi
}

F2_BASE=${F2_BASE:-outputs/train/p5_h100/F2_suffix_in_seed0_bs192_amp_h100}
WRIST_BASE=${WRIST_BASE:-outputs/train/p5_h100/E2_wrist_seed0_bs192_amp_h100}
F2_CKPT=${F2_CKPT:-$(resolve_checkpoint_dir "${F2_BASE}" "${STEPS}")}
WRIST_CKPT=${WRIST_CKPT:-$(resolve_checkpoint_dir "${WRIST_BASE}" "${STEPS}")}

cd "${ROOT_DIR}"
source "${VENV_PATH}"

wait_and_exec() {
  local pid_file="$1"
  shift 1
  ./scripts/wait_for_pid_then_run.sh "${pid_file}" "$@"
}

case "${MODE}" in
  after_gpu6_current)
    export GPU_ID=6
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    wait_and_exec \
      logs/gpu6_h100_queue.pid \
      bash -lc "cd '${ROOT_DIR}' && source '${VENV_PATH}' && \
        export HF_HOME='${HF_HOME}' && \
        export HUGGINGFACE_HUB_CACHE='${HUGGINGFACE_HUB_CACHE}' && \
        export HF_DATASETS_CACHE='${HF_DATASETS_CACHE}' && \
        export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}' && \
        export MUJOCO_GL='${MUJOCO_GL}' && \
        export OMP_NUM_THREADS='${OMP_NUM_THREADS}' && \
        export MKL_NUM_THREADS='${MKL_NUM_THREADS}' && \
        export OPENBLAS_NUM_THREADS='${OPENBLAS_NUM_THREADS}' && \
        export NUMEXPR_NUM_THREADS='${NUMEXPR_NUM_THREADS}' && \
        GPU_ID=6 QUEUE=ablation CKPT='${WRIST_CKPT}' EVAL_TASK='${EVAL_TASK}' EVAL_EPISODES='${EVAL_EPISODES}' EVAL_BATCH_SIZE='${EVAL_BATCH_SIZE}' EVAL_N_ACTION_STEPS='${EVAL_N_ACTION_STEPS}' ./scripts/launch_h100_followup_queue.sh"
    ;;
  after_gpu7_current)
    export GPU_ID=7
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    wait_and_exec \
      logs/gpu7_h100_queue.pid \
      bash -lc "cd '${ROOT_DIR}' && source '${VENV_PATH}' && \
        export HF_HOME='${HF_HOME}' && \
        export HUGGINGFACE_HUB_CACHE='${HUGGINGFACE_HUB_CACHE}' && \
        export HF_DATASETS_CACHE='${HF_DATASETS_CACHE}' && \
        export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}' && \
        export MUJOCO_GL='${MUJOCO_GL}' && \
        export OMP_NUM_THREADS='${OMP_NUM_THREADS}' && \
        export MKL_NUM_THREADS='${MKL_NUM_THREADS}' && \
        export OPENBLAS_NUM_THREADS='${OPENBLAS_NUM_THREADS}' && \
        export NUMEXPR_NUM_THREADS='${NUMEXPR_NUM_THREADS}' && \
        export CUDA_VISIBLE_DEVICES=7 && \
        lerobot-wm-eval --policy.pretrained_path='${F2_CKPT}' --policy.device=cuda --policy.n_action_steps='${EVAL_N_ACTION_STEPS}' --env.type=libero --env.task='${EVAL_TASK}' --eval.n_episodes='${EVAL_EPISODES}' --eval.batch_size='${EVAL_BATCH_SIZE}' 2>&1 | tee logs/F2_suffix_in_seed${SEED}_bs${BATCH_SIZE}_amp_h100_eval.log && \
        lerobot-wm-train --dataset.repo_id=HuggingFaceVLA/libero --policy.type=smolvla_world --policy.device=cuda --policy.push_to_hub=false --batch_size='${BATCH_SIZE}' --num_workers='${NUM_WORKERS}' --steps='${STEPS}' --output_dir=outputs/train/p5_h100/F3b_prefix_cross_seed${SEED}_bs${BATCH_SIZE}_amp_h100 --seed='${SEED}' --wandb.enable=false --policy.use_amp=true --policy.init_from_policy_path=HuggingFaceVLA/smolvla_libero --policy.dataset_repo_id=HuggingFaceVLA/libero --policy.cache_dir=cache --policy.world_latents_source=vjepa --policy.latent_suffix=m4 --policy.world_latent_dim=1408 --policy.context_frames=4 --policy.future_offset=8 --policy.lambda_world=0.2 --policy.world_memory_mode_train=pred --policy.world_camera=front --policy.enable_world_injection=true --policy.world_prefix_cross_attn=true 2>&1 | tee logs/F3b_prefix_cross_seed${SEED}_bs${BATCH_SIZE}_amp_h100.log && \
        F3B_CKPT=\\$(if [ -d outputs/train/p5_h100/F3b_prefix_cross_seed${SEED}_bs${BATCH_SIZE}_amp_h100/checkpoints/$(printf '%06d' ${STEPS})/pretrained_model ]; then echo outputs/train/p5_h100/F3b_prefix_cross_seed${SEED}_bs${BATCH_SIZE}_amp_h100/checkpoints/$(printf '%06d' ${STEPS})/pretrained_model; else echo outputs/train/p5_h100/F3b_prefix_cross_seed${SEED}_bs${BATCH_SIZE}_amp_h100/checkpoints/${STEPS}/pretrained_model; fi) && \
        lerobot-wm-eval --policy.pretrained_path=\\${F3B_CKPT} --policy.device=cuda --policy.n_action_steps='${EVAL_N_ACTION_STEPS}' --env.type=libero --env.task='${EVAL_TASK}' --eval.n_episodes='${EVAL_EPISODES}' --eval.batch_size='${EVAL_BATCH_SIZE}' 2>&1 | tee logs/F3b_prefix_cross_seed${SEED}_bs${BATCH_SIZE}_amp_h100_eval.log"
    ;;
  *)
    echo "Usage: MODE=after_gpu6_current|after_gpu7_current $0"
    exit 2
    ;;
esac

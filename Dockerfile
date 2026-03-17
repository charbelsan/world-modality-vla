# World Modality VLA Docker Image
#
# Intended use:
# - isolate SmolVLA/LeRobot/LIBERO experiments from other jobs on shared GPU boxes
# - run on NVIDIA machines such as P5/H100 with `--gpus '"device=6"'`
#
# Notes:
# - This image targets the current SmolVLA + `smolvla_world` workflow, not the older
#   standalone Qwen training path.
# - LIBERO/MuJoCo headless rollouts default to EGL inside the container.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

LABEL maintainer="charbel"
LABEL description="World modality as external memory for VLA policies"

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    git-lfs \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libglew2.2 \
    libglfw3 \
    libgomp1 \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# Copy minimal metadata first so dependency installation benefits from Docker layer caching.
COPY pyproject.toml requirements.txt README.md ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies used by the current SmolVLA/LeRobot path.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        "lerobot @ git+https://github.com/huggingface/lerobot.git" \
        mujoco \
        libero

# Copy the repo after dependencies to keep rebuilds fast when only code changes.
COPY . .

RUN pip install --no-cache-dir -e .

RUN mkdir -p \
    /workspace/cache \
    /workspace/eval_libero_results \
    /workspace/logs \
    /workspace/logs_llm \
    /workspace/outputs \
    /workspace/.hf_cache

ENV HF_HOME=/workspace/.hf_cache
ENV HF_HUB_CACHE=/workspace/.hf_cache
ENV TRANSFORMERS_CACHE=/workspace/.hf_cache
ENV MUJOCO_GL=egl
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

CMD ["bash"]

# World Modality VLA Docker Image
# Supports NVIDIA GPUs with CUDA 12.1

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="charbel"
LABEL description="World tokens as first-class modality for VLA policies"

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install lerobot separately (large dependency)
RUN pip install --no-cache-dir "lerobot @ git+https://github.com/huggingface/lerobot.git"

# Copy project code
COPY . .

# Install project in editable mode
RUN pip install --no-cache-dir -e .

# Create directories for data and logs
RUN mkdir -p /workspace/cache /workspace/logs

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command: show help
CMD ["python", "-m", "world_modality.train", "--help"]

# Example usage:
# docker build -t world-vla .
# docker run --gpus all -v $(pwd)/cache:/workspace/cache -v $(pwd)/logs:/workspace/logs world-vla \
#   python -m world_modality.train --model_type B_cont --batch_size 256 --max_epochs 1

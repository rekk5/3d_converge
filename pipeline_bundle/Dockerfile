# Minimal Dockerfile to run the pipeline via YAML without modifying code
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create workdir
WORKDIR /app

# Copy only the pipeline (main + needed modules)
# Adjust this list if you add/remove modules
COPY main.py ./
COPY naming_convention.py ./
COPY vggt_recon.py ./
COPY colmap_recon.py ./
COPY tsdf_fusion.py ./
COPY segmentation.py ./
COPY diffraction.py ./
COPY diffraction_noseg.py ./
COPY comparison.py ./
COPY vggt ./vggt

# YAML runner
COPY pipeline_bundle/run_from_yaml.py ./
COPY pipeline_bundle/config.yaml ./config.yaml

# Python deps
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
      open3d==0.19.0 \
      numpy<2 \
      trimesh \
      scikit-learn \
      pyyaml \
      opencv-python \
      einops safetensors huggingface_hub \
      # torch + vision CUDA 12.1 wheels
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      torch==2.3.1 torchvision==0.18.1

# Default command uses the YAML
CMD ["python3", "run_from_yaml.py", "config.yaml"]

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    ninja-build \
    git \
    git-lfs \
    wget \
    ffmpeg \
    python3-opencv \
    libglib2.0-0 \
    libgl1 \
    libgl1-mesa-glx \
    libglvnd-dev \
    libegl1 \
    libglfw3 \
    libosmesa6 \
    libosmesa6-dev \
    libglew2.2 \
    libglew-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxrender-dev \
    patchelf \
    tmux \
    unzip \
    zip \
    bzip2 \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip
RUN git lfs install --system

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace

# Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.1)
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# Clone repository
RUN git clone https://github.com/ZhaoyangLi-1/condBFNPol.git

WORKDIR /workspace/condBFNPol

# Install other dependencies
RUN sed -E \
    -e '/^torch$/d' \
    -e '/^torchvision==/d' \
    -e '/^torchaudio==/d' \
    requirements.txt > requirements.filtered.txt \
    && pip install -r requirements.filtered.txt

ENV PYTHONPATH="/workspace/condBFNPol/src/diffusion-policy:/workspace/condBFNPol"
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

CMD ["/bin/bash"]
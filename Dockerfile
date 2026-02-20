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
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    libegl1 \
    libglfw3 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libosmesa6 \
    libglew2.2 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip
RUN git lfs install --system

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace

# clone repo
RUN git clone https://github.com/ZhaoyangLi-1/condBFNPol.git

# install requirements
WORKDIR /workspace/condBFNPol
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# set python path
ENV PYTHONPATH="/workspace/condBFNPol/src/diffusion-policy:/workspace/condBFNPol:${PYTHONPATH}"
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

WORKDIR /workspace
CMD ["/bin/bash"]
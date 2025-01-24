#FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG USERNAME=vscode

# Match the following two ids with the values from your local dev machine
ARG USER_UID=610213736
ARG USER_GID=610200513

# Keeps Python from generating .pyc files in the container
#ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Update and install debian stuff
RUN apt-get update && apt-get -y install \  
    sudo \
    wget \
    unzip \
    git \
    curl \
    vim \
    htop \
    mc \
    lsb-release \
    manpages-dev \
    build-essential \
    libgl1-mesa-glx \
    mesa-utils\
    libboost-dev \
    libxerces-c-dev \
    libeigen3-dev\
    python-is-python3 \
    python3-pip \
    python3-tk \
    python3-dev \
    libopenblas-dev \
    libsparsehash-dev \
    cmake && \
    # Clear APT cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add user that will automatically used by vscode
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME

# Enable sudo for user
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set env variable for MinkowskiEngine
ENV OMP_NUM_THREADS=16

COPY requirements.txt .
COPY third_party/ third_party/

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"

RUN pip install --no-cache-dir "cython<3.0.0" && \
    pip install --no-cache-dir --no-build-isolation pyyaml==5.4.1 && \
    pip install --no-cache-dir -r requirements.txt && \
    # Workaround for https://github.com/kumuji/volumentations/issues/252:
    pip install --no-cache-dir importlib-metadata==8.0.0

# torch-scatter requires torch to be installed first. Therefore, it cannot be included in requirements.txt.
RUN pip install --no-cache-dir torch-scatter==2.1.2

RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git@70f454304e1a38378200459dd2dbca0f0f4a5ab4'

RUN cd third_party/MinkowskiEngine && \
    python setup.py install --force_cuda --blas=openblas

# Only required for ScanNet (segment the scenes into smaller parts to save memory)
#RUN cd third_party/ScanNet/Segmentator && \
#    make

RUN cd third_party/pointnet2 && \
    python setup.py install

#ENV HYDRA_FULL_ERROR=1

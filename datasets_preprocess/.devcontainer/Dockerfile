#FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04
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
RUN apt-get update && apt-get -y -qq install \
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
    msttcorefonts \
    fontconfig

    # If you have issues with the Arial font missing, reinstall the msttcorefonts package. You need to accept the EULA in the prompt that appears during installation.

# Add user that will automatically used by vscode
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME

# Enable sudo for user
RUN apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Clean APT cache
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set env variable for MinkowskiEngine
ENV OMP_NUM_THREADS=16

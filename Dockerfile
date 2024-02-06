# Set the base image
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

# Disable interactive configuration
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Set working directory
WORKDIR /ahmed

# Install basics
RUN apt-get update -y && \
    apt-get install -y build-essential apt-utils git curl ca-certificates bzip2 tree htop wget libglib2.0-0 libsm6 libxext6 libxrender-dev bmon iotop g++ python3.7 python3.7-dev python3.7-distutils libgl1-mesa-glx

# Install cmake v3.13.2
RUN apt-get purge -y cmake && \
    mkdir /root/temp && \
    cd /root/temp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2.tar.gz && \
    tar -xzvf cmake-3.13.2.tar.gz && \
    cd cmake-3.13.2 && \
    bash ./bootstrap && \
    make && \
    make install && \
    cmake --version && \
    rm -rf /root/temp

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya

# Update PATH
ENV PATH /opt/conda/bin:$PATH

# Create a conda environment and install packages
RUN conda create -n active3d python=3.7

SHELL ["conda", "run", "-n", "active3d", "/bin/bash", "-c"]

COPY . .

RUN pip install -r requirements.txt


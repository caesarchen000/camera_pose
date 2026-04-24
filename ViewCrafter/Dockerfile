FROM docker.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    wget \
    pkg-config \
    build-essential \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    python3 \ 
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ensure python points to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip to avoid issues
RUN pip install --upgrade pip

# Install Cython<3 and numpy to build av
RUN pip install "Cython<3" numpy

# Install av with no build isolation to use the installed Cython<3
RUN pip install av==10.0.0 --no-build-isolation

# Install requirements
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Install pytorch3d
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"

# Pre-download OpenCLIP model to make image immutable
RUN python3 -c "import open_clip; open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')"

# Download checkpoints
RUN mkdir -p /app/checkpoints/
RUN wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P /app/checkpoints/
RUN wget https://huggingface.co/Drexubery/ViewCrafter_25/resolve/main/model.ckpt -P /app/checkpoints/

# Copy source code
COPY . /app
WORKDIR /app
RUN sed -i s/127.0.0.1/0.0.0.0/ gradio_app.py

# Start gradio app
EXPOSE 80
CMD ["python3", "gradio_app.py"]
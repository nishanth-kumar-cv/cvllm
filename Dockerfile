FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install system dependencies and clean up
#RUN apt update && apt install -y \
#    git curl build-essential python3 python3-pip python-is-python3 \
#    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    echo 'source $HOME/.cargo/env' >> /root/.bashrc
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Copy only requirement files first for caching
COPY requirements.txt ./python_finetune/requirements.txt

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r python_finetune/requirements.txt

# Install system and CUDA dev dependencies required to build bitsandbytes
# System packages required for building bitsandbytes
#RUN apt update && \
#    apt install -y --allow-change-held-packages \
#    git cmake build-essential libnccl2 libnccl-dev python3-dev wget unzip


# Clone and build bitsandbytes
#RUN git clone https://github.com/TimDettmers/bitsandbytes.git /opt/bnb && \
#    cd /opt/bnb && \
#    python setup.py install

# Install core dependencies
#RUN pip install --default-timeout=100 --retries=10 --no-cache-dir \
#      torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
#    pip install --default-timeout=100 --retries=10 --no-cache-dir \ 
#      bitsandbytes==0.42.0 accelerate transformers peft datasets protobuf huggingface_hub

# Install CUDA 11 runtime compatibility libs for bitsandbytes
#RUN apt update && \
#    apt install -y --allow-change-held-packages \
#    cuda-cudart-11-0

# Symlink libcudart for compatibility with bitsandbytes


RUN pip install bitsandbytes-cuda113

# Copy the entire app
COPY . .

# Build Rust components
# WORKDIR /app/rust_preprocessing
# RUN cargo build --release

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

RUN python python_finetune/finetune.py

# Set working directory and default command
WORKDIR /app
CMD ["uvicorn", "python_finetune.startup:app", "--host", "0.0.0.0", "--port", "8000"]
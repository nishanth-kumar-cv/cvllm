FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
#FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Base dependencies
RUN apt update && apt install -y \
    git curl build-essential python3 python3-pip python-is-python3 \
    python3-dev cmake wget unzip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

ENV PATH="/root/.cargo/bin:$PATH"
ENV BNB_CUDA_VERSION=124
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV BITSANDBYTES_CUDA_SETUP_DIR=/tmp/bnb_cache

# Install PyTorch and other packages
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio

# Set working directory
WORKDIR /app

# Copy your source code
COPY . .

RUN mkdir -p ~/.huggingface && \
    echo "${HF_TOKEN}" > ~/.huggingface/token


#RUN pip uninstall -y triton
#RUN pip install git+https://github.com/openai/triton@main

# Install bitsandbytes without C extensions (we'll supply .so files)
#RUN pip install bitsandbytes==0.39.1 --no-binary bitsandbytes
# Download the secured bitsandbytes wheel
#RUN pip install --upgrade pip && \
#    pip install "huggingface_hub" && \
#    pip download --quiet --no-deps --pre --extra-index-url https://download.pytorch.org/whl/cu124 \
#       --trusted-host huggingface.co \
#        --no-cache-dir \
#        --dest /tmp \
#        bitsandbytes==0.39.1 && \
#    pip install /tmp/bitsandbytes-0.39.1*.whl
RUN pip install bitsandbytes triton
# Copy precompiled binaries (libbitsandbytes_cuda124.so + libbitsandbytes_cpu.so)
COPY bitsandbytes/libbitsandbytes_cuda124.so /usr/local/lib/python3.10/dist-packages/bitsandbytes/libbitsandbytes_cuda124.so
#COPY bitsandbytes/bitsandbytes/libbitsandbytes_cpu.so /usr/local/lib/python3.10/dist-packages/bitsandbytes/libbitsandbytes_cpu.so
RUN mkdir -p /tmp/offload

COPY requirements.txt python_finetune/requirements.txt
COPY mistral-finetuned python_finetune/.
# Install app dependencies
RUN pip install -r python_finetune/requirements.txt

CMD ["uvicorn", "python_finetune.startup:app", "--host", "0.0.0.0", "--port", "8000"]
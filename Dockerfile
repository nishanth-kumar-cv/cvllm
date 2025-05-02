FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt update && apt install -y git curl build-essential python3 python3-pip python-is-python3

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    echo 'source $HOME/.cargo/env' >> /root/.bashrc
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app
COPY . .

RUN pip install --upgrade pip && \
    pip install -r python_finetune/requirements.txt && \
    pip install fastapi uvicorn accelerate transformers peft datasets bitsandbytes

WORKDIR /app/rust_preprocessing
RUN cargo build --release

WORKDIR /app/python_finetune
CMD ["uvicorn", "infer_api:app", "--host", "0.0.0.0", "--port", "8000"]

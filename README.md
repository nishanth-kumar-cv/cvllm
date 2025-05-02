# Mistral-7B Finetuning with Rust + Python

This project fine-tunes the Mistral-7B language model using a hybrid setup:

- **Rust**: For fast tokenization and preprocessing
- **Python**: For LoRA-based model fine-tuning using Hugging Face Transformers
- **FastAPI**: For inference serving
- **Terraform**: For provisioning a GPU VM on GCP
- **GitHub Actions**: For CI/CD pipeline automation

## Quick Start

### Prerequisites
- Docker
- Terraform
- GCP account with billing enabled

### 1. Preprocess Data
```bash
./scripts/run_preprocessing.sh
```

### 2. Fine-Tune Mistral
```bash
./scripts/train.sh
```

### 3. Run Inference API
```bash
./scripts/serve.sh
```

### 4. Deploy to GCP with Terraform
```bash
cd terraform
terraform init
terraform apply
```

### 5. CI/CD Deployment
- Push to `main` triggers GitHub Actions to build & push Docker image
- Optional: Deploys to GPU VM if SSH secrets are configured

## Folder Structure
```
.
├── rust_preprocessing/   # Rust-based tokenizer
├── python_finetune/      # Python training & API
├── scripts/              # Helper scripts
├── terraform/            # GCP provisioning
├── .github/workflows/    # GitHub Actions
├── Dockerfile            # Unified container
├── README.md             # Project overview
└── .gitignore
```

## License
MIT

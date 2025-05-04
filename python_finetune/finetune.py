import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import torch
from huggingface_hub import HfFolder
from transformers import BitsAndBytesConfig

print("BNB cache dir:", os.environ.get("BITSANDBYTES_CUDA_SETUP_DIR"))

hf_token = os.getenv("HF_TOKEN")

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"

if hf_token:
    HfFolder.save_token(hf_token)
else:
    raise RuntimeError("HF_TOKEN not set")

# Step 1: Load tokenizer and model
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token



quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=hf_token
)

model.config.pad_token_id = tokenizer.pad_token_id

# PEFT config for QLoRA
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

def preprocess_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"  # ensures uniform tensor shapes
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return {k: v.tolist() for k, v in tokenized.items()}


# Step 2: Prepare toy training data (replace with real data later)
raw_data = [
    {"text": "What is the capital of France? Paris."},
    {"text": "Translate 'hello' to Spanish. Hola."},
]

# Step 3: Tokenize the dataset
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

dataset = Dataset.from_list(raw_data)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

model = get_peft_model(model, peft_config)

# Step 5: Training setup
args = TrainingArguments(
    output_dir="./mistral-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    logging_steps=1,
    save_steps=10,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=args
)

trainer.train()

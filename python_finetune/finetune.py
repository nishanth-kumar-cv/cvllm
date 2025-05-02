from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import torch
from huggingface_hub import HfFolder
import os
from transformers import BitsAndBytesConfig

hf_token = os.getenv("HF_TOKEN")
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

# PEFT config for QLoRA
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Step 2: Prepare toy training data (replace with real data later)
raw_data = [
    {"text": "What is the capital of France? Paris."},
    {"text": "Translate 'hello' to Spanish. Hola."},
]

# Step 3: Tokenize the dataset
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

dataset = Dataset.from_list(raw_data)
tokenized_dataset = dataset.map(tokenize)

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

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset, load_dataset, concatenate_datasets
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


# Step 2: Load and combine real datasets from rag_pipeline.py
print("[INFO] Loading datasets from HuggingFace...")
ds_mini_reasoning = load_dataset("KingNish/mini_reasoning_1k")['train']
ds_finance_alpaca = load_dataset("gbharti/finance-alpaca")['train']
ds_openai_mrcr = load_dataset("openai/mrcr")['train']
ds_anthropic_economic_index = load_dataset("Anthropic/EconomicIndex")['train']
ds_general_reasoning = load_dataset("GeneralReasoning/GeneralThought-430K")['train']
ds_hf_ultrafeedback = load_dataset("HuggingFaceH4/ultrafeedback_binarized")['train_prefs']
ds_zennykenny_finance = load_dataset("ZennyKenny/synthetic_vc_financial_decisions_reasoning_dataset")['test']

all_datasets = [
    ds_mini_reasoning,
    ds_finance_alpaca,
    ds_openai_mrcr,
    ds_anthropic_economic_index,
    ds_general_reasoning,
    ds_hf_ultrafeedback,
    ds_zennykenny_finance
]

combined = concatenate_datasets(all_datasets)
print(f"[INFO] Combined dataset size: {len(combined)}")

def get_text(example):
    return example.get("text") or example.get("prompt") or ""

filtered = combined.filter(lambda x: (x.get("text") or x.get("prompt")) and (x.get("text") or x.get("prompt")).strip())
filtered = filtered.map(lambda x: {"text": get_text(x)})

# Step 3: Tokenize the dataset
print("[INFO] Tokenizing dataset...")
tokenized_dataset = filtered.map(preprocess_function, batched=True)

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

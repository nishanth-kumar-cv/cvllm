from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
dataset = load_dataset("json", data_files="../data/tokenized.jsonl")

args = TrainingArguments(
    output_dir="./mistral-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    args=args
)

trainer.train()

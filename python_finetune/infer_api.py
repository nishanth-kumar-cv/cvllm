from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from python_finetune.load_faiss import load_faiss_index
import torch
import os

app = FastAPI()

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Step 1: Load the base model
base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", offload_folder="/tmp/offload")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Step 2: Load the adapter
adapter_path = "./mistral-finetuned/checkpoint-1"
model = PeftModel.from_pretrained(model, adapter_path,offload_folder="/tmp/offload", device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#retriever = FAISS.load_local("faiss_store", embedding_model, allow_dangerous_deserialization=True)
retriever = load_faiss_index("faiss_store_safe", embedding_model)
docs = retriever.similarity_search("how to improve my sales for new shoe", k=3)

class Prompt(BaseModel):
    text: str

@app.post("/generate")
async def generate_text(prompt: Prompt):
    docs = retriever.similarity_search(prompt.text, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    full_prompt = f"### Context:\n{context}\n\n### Question:\n{prompt.text}\n\n### Answer:"
    output = generator(full_prompt, max_new_tokens=150)[0]['generated_text']
    return {"response": output}

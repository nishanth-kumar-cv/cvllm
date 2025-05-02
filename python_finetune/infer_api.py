from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

app = FastAPI()

model = AutoModelForCausalLM.from_pretrained("./mistral-finetuned", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = FAISS.load_local("faiss_store", embedding_model)

class Prompt(BaseModel):
    text: str

@app.post("/generate")
async def generate_text(prompt: Prompt):
    docs = retriever.similarity_search(prompt.text, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    full_prompt = f"### Context:\n{context}\n\n### Question:\n{prompt.text}\n\n### Answer:"
    output = generator(full_prompt, max_new_tokens=150)[0]['generated_text']
    return {"response": output}

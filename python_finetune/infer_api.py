from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from python_finetune.load_faiss import load_faiss_index
from peft import PeftModel
import torch
import os
import time
import threading
from typing import Optional, Dict, Any
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

router = APIRouter()

# Global state for tracking inference progress
inference_progress: Dict[str, Dict[str, Any]] = {}

# Thread pool for handling inference tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

# Load base model
base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # You can switch to a smaller or quantized model here, e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", offload_folder="/tmp/offload")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load adapter
adapter_path = "./mistral-finetuned/checkpoint-1"
model = PeftModel.from_pretrained(model, adapter_path, offload_folder="/tmp/offload", device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = load_faiss_index("faiss_store_safe", embedding_model)

class Prompt(BaseModel):
    text: str
    file_id: Optional[str] = None

class FileUpload(BaseModel):
    file: UploadFile
    text: Optional[str] = None

def update_progress(task_id: str, status: str, progress: float = 0.0, result: Optional[str] = None):
    """Update the progress of an inference task"""
    inference_progress[task_id] = {
        "status": status,
        "progress": progress,
        "result": result,
        "timestamp": datetime.now().isoformat()
    }

def run_inference(task_id: str, prompt: str, file_content: Optional[bytes] = None):
    """Run inference in a separate thread with detailed timing logs"""
    try:
        timings = {}
        start_total = time.time()
        update_progress(task_id, "processing", 0.1)
        print(f"[{datetime.now()}] [TASK {task_id}] Inference started.")

        # Process file if provided
        if file_content:
            temp_path = f"/tmp/{task_id}_{int(time.time())}"
            with open(temp_path, "wb") as f:
                f.write(file_content)
            update_progress(task_id, "processing", 0.3)
            print(f"[{datetime.now()}] [TASK {task_id}] File processed and saved to temp.")
            os.remove(temp_path)

        # RAG retrieval
        start_rag = time.time()
        update_progress(task_id, "processing", 0.4)
        print(f"[{datetime.now()}] [TASK {task_id}] RAG retrieval started.")
        docs = retriever.similarity_search(prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        end_rag = time.time()
        timings['rag_retrieval'] = end_rag - start_rag
        print(f"[{datetime.now()}] [TASK {task_id}] RAG retrieval finished. Time: {timings['rag_retrieval']:.2f}s")

        # Prompt construction
        start_prompt = time.time()
        full_prompt = f"### Context:\n{context}\n\n### Question:\n{prompt}\n\n### Answer:"
        end_prompt = time.time()
        timings['prompt_construction'] = end_prompt - start_prompt
        print(f"[{datetime.now()}] [TASK {task_id}] Prompt constructed. Time: {timings['prompt_construction']:.2f}s")

        # Mistral inference
        start_mistral = time.time()
        update_progress(task_id, "processing", 0.6)
        print(f"[{datetime.now()}] [TASK {task_id}] Mistral inference started.")
        # Reduce max_new_tokens for faster inference (default: 64)
        max_new_tokens = 64
        with torch.inference_mode():
            output = generator(full_prompt, max_new_tokens=max_new_tokens)[0]['generated_text']
        end_mistral = time.time()
        timings['mistral_inference'] = end_mistral - start_mistral
        print(f"[{datetime.now()}] [TASK {task_id}] Mistral inference finished. Time: {timings['mistral_inference']:.2f}s")

        # Total time
        end_total = time.time()
        timings['total'] = end_total - start_total
        print(f"[{datetime.now()}] [TASK {task_id}] Inference completed. Total time: {timings['total']:.2f}s")

        # Update progress with result and timings
        update_progress(task_id, "completed", 1.0, {
            'output': output,
            'timings': timings
        })

    except Exception as e:
        print(f"[{datetime.now()}] [TASK {task_id}] ERROR: {str(e)}")
        update_progress(task_id, "error", 0.0, str(e))

@router.post("/text")
async def generate_text(prompt: Prompt, background_tasks: BackgroundTasks):
    """Generate text with progress tracking"""
    task_id = f"task_{int(time.time())}"
    update_progress(task_id, "started", 0.0)
    
    # Start inference in background
    background_tasks.add_task(run_inference, task_id, prompt.text)
    
    return {"task_id": task_id}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), text: Optional[str] = None):
    """Handle file upload with progress tracking"""
    task_id = f"file_{int(time.time())}"
    update_progress(task_id, "started", 0.0)
    
    # Read file content
    contents = await file.read()
    
    # Start processing in background
    background_tasks = BackgroundTasks()
    background_tasks.add_task(run_inference, task_id, text or "", contents)
    
    return {"task_id": task_id}

@router.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """Get the progress of an inference task"""
    if task_id not in inference_progress:
        return {"status": "not_found"}
    prog = inference_progress[task_id]
    # If completed, unwrap output for backward compatibility
    if prog.get('status') == 'completed' and isinstance(prog.get('result'), dict):
        return {
            **prog,
            'response': prog['result'].get('output', ''),
            'timings': prog['result'].get('timings', {})
        }
    return prog

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "retriever_loaded": retriever is not None,
        "active_tasks": len(inference_progress)
    }

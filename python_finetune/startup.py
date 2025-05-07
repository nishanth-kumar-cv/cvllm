from fastapi import FastAPI
from sqlmodel import SQLModel, create_engine, Session
from python_finetune.api_routes import auth_routes
import os
from python_finetune.rag_pipeline import build_faiss_index
if not os.path.exists("faiss_store_safe/index.faiss"):
    print("Index not found. Building FAISS index...")
    build_faiss_index()
#from python_finetune.infer_api import app as infer_app
from python_finetune.upload_endpoint import app as upload_app
from python_finetune.infer_api import router as infer_router
from typing import Optional


DATABASE_URL = "postgresql://cv-videos-rdb:cV13579$@35.234.145.222:5432/cvllm"
engine = create_engine(DATABASE_URL)

app = FastAPI()



app.include_router(auth_routes.router)
app.include_router(infer_router, prefix="/generate")
#app.mount("/", infer_app)
app.mount("/upload", upload_app)

@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

@app.get("/")
def root():
    return {"status": "Mistral LLM API running"}

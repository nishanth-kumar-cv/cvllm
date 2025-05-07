from fastapi import FastAPI, File, UploadFile
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from python_finetune.utils.loader import parse_and_embed
import os

app = FastAPI()
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    parse_and_embed(file.filename, contents, embedding_model)
    return {"status": "Indexed", "file": file.filename}

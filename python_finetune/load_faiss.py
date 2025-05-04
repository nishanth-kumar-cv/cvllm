from langchain.schema import Document
import faiss
import os
import json

def load_faiss_index(path, embedding_model):
    # Load FAISS index
    index = faiss.read_index(os.path.join(path, "index.faiss"))

    # Load metadata and text
    with open(os.path.join(path, "index.json")) as f:
        metadata = json.load(f)
    
    documents = [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in metadata]

    # Create a new FAISS vectorstore
    vectorstore = FAISS(embedding_model.embed_query, index, documents)
    return vectorstore

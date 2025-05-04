import faiss
import json
import os

def save_faiss_index(index, documents, path):
    os.makedirs(path, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(path, "index.faiss"))

    # Save metadata and text
    metadata = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]
    with open(os.path.join(path, "index.json"), "w") as f:
        json.dump(metadata, f)
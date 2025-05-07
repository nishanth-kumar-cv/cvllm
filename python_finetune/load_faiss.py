from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
import faiss
import os
import json

def load_faiss_index(path, embedding_model):
    # Load FAISS index
    index = faiss.read_index(os.path.join(path, "index.faiss"))

    # Load metadata and text
    with open(os.path.join(path, "index.json")) as f:
        metadata = json.load(f)
    
     # Reconstruct Document list
    documents = [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in metadata]

    # Create required components
    docstore_dict = {str(i): doc for i, doc in enumerate(documents)}
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    # Create the FAISS vectorstore
    vectorstore = FAISS(
        embedding_function=embedding_model.embed_query,
        index=index,
        docstore=InMemoryDocstore(docstore_dict),
        index_to_docstore_id=index_to_docstore_id
    )
    return vectorstore

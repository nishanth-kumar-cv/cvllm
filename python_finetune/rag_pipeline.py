from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from python_finetune.load_faiss import load_faiss_index
from python_finetune.save_faiss import save_faiss_index
import os

# Load documents and create vector DB
loader = TextLoader("../data/business_docs/earnings_q4.txt")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, embedding_model)
#vectorstore.save_local("faiss_store")
save_faiss_index(vectorstore.index, splits, "faiss_store_safe")

# Retrieval + Generation
query = "What were Acme Corp's Q4 earnings?"
retriever = load_faiss_index("faiss_store_safe", embedding_model)
docs = retriever.similarity_search(query, k=3)

context = "\n\n".join([doc.page_content for doc in docs])
prompt = f"### Context:\n{context}\n\n### Question:\n{query}\n\n### Answer:"

# Load model for generation
tokenizer = AutoTokenizer.from_pretrained("./mistral-finetuned/checkpoint-1")
model = AutoModelForCausalLM.from_pretrained("./mistral-finetuned/checkpoint-1", device_map="auto")
gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

response = gen(prompt, max_new_tokens=150)[0]['generated_text']
print(response)

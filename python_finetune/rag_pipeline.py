from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from python_finetune.load_faiss import load_faiss_index
from python_finetune.save_faiss import save_faiss_index
import os
from datasets import load_dataset
from datasets import concatenate_datasets
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
import numpy as np
import faiss
from langchain.docstore.in_memory import InMemoryDocstore


def build_faiss_index():
    ds_mini_reasoning = load_dataset("KingNish/mini_reasoning_1k")
    ds_finance_alpaca = load_dataset("gbharti/finance-alpaca")
    ds_openai_mrcr = load_dataset("openai/mrcr")
    ds_anthropic_economic_index = load_dataset("Anthropic/EconomicIndex")
    ds_general_reasoning = load_dataset("GeneralReasoning/GeneralThought-430K")
    ds_hf_ultrafeedback = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    ds_zennykenny_finance = load_dataset("ZennyKenny/synthetic_vc_financial_decisions_reasoning_dataset")

    all_datasets = {
        "ds1":ds_anthropic_economic_index["train"],
        "ds2":ds_finance_alpaca["train"],
        "ds3":ds_general_reasoning["train"],
        "ds4":ds_hf_ultrafeedback["train_prefs"],
        "ds5":ds_mini_reasoning["train"],
        "ds6":ds_openai_mrcr["train"],
        "ds7":ds_zennykenny_finance["test"]
    }

    alldocs = concatenate_datasets([
    ds_anthropic_economic_index["train"],
    ds_finance_alpaca["train"],
    ds_general_reasoning["train"],
    ds_hf_ultrafeedback["train_prefs"],
    ds_mini_reasoning["train"],
    ds_openai_mrcr["train"],
    ds_zennykenny_finance["test"]])
    seen = set()
    docs = []
    raw_docs = []
    for name, dataset in all_datasets.items():
        for idx, example in enumerate(dataset):
            content = example.get("text") or example.get("prompt")
            if content and content.strip():
                raw_docs.append(Document(
                    page_content=content.strip(),
                    metadata={"source": name, "record_id": idx}
                ))

    
    #splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""])

    #embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"})  # ✅ Use GPU
    print(embedding_model.client.device)  # Should be "cuda"

    print(f"[INFO] Total raw docs: {len(raw_docs)}")
    splits = splitter.split_documents(raw_docs)

    print(f"[INFO] Total splits: {len(splits)}")
    texts = [doc.page_content for doc in splits if doc.page_content.strip()]
    print(f"[INFO] Total valid texts for embedding: {len(texts)}")

    if not texts:
        raise ValueError("No valid documents to embed. Check input data.")

    batch_size = 512
    index = faiss.IndexFlatL2(embedding_model.client.get_sentence_embedding_dimension())
    stored_docs = []
    try:
        for i in range(0, len(splits), batch_size):
            batch_docs = splits[i:i+batch_size]
            batch_texts = [doc.page_content for doc in batch_docs]
            try:
                embeddings = embedding_model.embed_documents(batch_texts)
                index.add(np.array(embeddings).astype("float32"))
                stored_docs.extend(batch_docs)
                if i % (batch_size * 500) == 0:
                    print(f"[INFO] Embedded {i} texts...",flush=True)
                    docstore_dict = {str(i): doc for i, doc in enumerate(stored_docs)}
                    index_to_docstore_id = {i: str(i) for i in range(len(stored_docs))}

                    # Create FAISS vectorstore instance correctly
                    vectorstore = FAISS(
                        embedding_function=embedding_model,
                        index=index,
                        docstore=InMemoryDocstore(docstore_dict),
                        index_to_docstore_id=index_to_docstore_id,)
                    print("Saving embeddings at checkpoint", flush=True)
                    save_faiss_index(vectorstore.index, stored_docs, "faiss_store_safe")
                    print(f"[CHECKPOINT] Saved at {i} texts", flush=True)
            except Exception as e:
                print(f"[ERROR] at batch {i}: {str(e)}",flush=True)
                continue
    except Exception as outer:
        print(f"[FATAL ERROR] Embedding crashed: {outer}", flush=True)
    # Compute embeddings
    #embeddings = embedding_model.embed_documents(texts)
    #vectorstore = FAISS(embedding_model=embedding_model, index=index, docstore=stored_docs)
    print("Saving embeddings",flush=True)
    #vectorstore.save_local("faiss_store_safe")
    #save_faiss_index(vectorstore.index, splits, "faiss_store_safe")



# Retrieval + Generation
#query = "What were Acme Corp's Q4 earnings?"
#retriever = load_faiss_index("faiss_store_safe", embedding_model)
#docs = retriever.similarity_search(query, k=3)

#context = "\n\n".join([doc.page_content for doc in docs])
#prompt = f"### Context:\n{context}\n\n### Question:\n{query}\n\n### Answer:"

# Load model for generation
#tokenizer = AutoTokenizer.from_pretrained("./mistral-finetuned/checkpoint-1")
#model = AutoModelForCausalLM.from_pretrained("./mistral-finetuned/checkpoint-1", device_map="auto")
#gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

#response = gen(prompt, max_new_tokens=150)[0]['generated_text']
#print(response)

from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from tempfile import NamedTemporaryFile

def parse_and_embed(filename, content, embedding_model):
    ext = filename.split('.')[-1].lower()
    with NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(content)
        tmp.flush()
        if ext == 'pdf':
            loader = PyPDFLoader(tmp.name)
        elif ext == 'csv':
            loader = CSVLoader(tmp.name)
        else:
            raise ValueError("Unsupported file type")
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        store = FAISS.from_documents(splits, embedding_model)
        store.save_local("faiss_store")

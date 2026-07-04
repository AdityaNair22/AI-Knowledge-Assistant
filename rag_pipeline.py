
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_DIR = Path(__file__).parent / "data"
VECTORSTORE_PATH = Path(__file__).parent / "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def load_documents():
    docs = []
    for file in DATA_DIR.iterdir():
        try:
            if file.suffix == ".pdf":
                loader = PyPDFLoader(str(file))
                docs.extend(loader.load())
            elif file.suffix == ".txt":
                loader = TextLoader(str(file), encoding="utf-8")
                docs.extend(loader.load())
        except Exception as e:
            print("Could not load " + file.name)
    return docs

def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(VECTORSTORE_PATH))
    return vectorstore

def load_vectorstore():
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(str(VECTORSTORE_PATH), embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def get_retriever():
    if VECTORSTORE_PATH.exists():
        vectorstore = load_vectorstore()
    else:
        docs = load_documents()
        if not docs:
            return None
        vectorstore = build_vectorstore(docs)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def rebuild_vectorstore():
    import shutil
    if VECTORSTORE_PATH.exists():
        shutil.rmtree(VECTORSTORE_PATH)
    docs = load_documents()
    if not docs:
        return None
    vectorstore = build_vectorstore(docs)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def get_document_count():
    if not DATA_DIR.exists():
        return 0
    return len([f for f in DATA_DIR.iterdir() if f.suffix in [".pdf", ".txt"]])

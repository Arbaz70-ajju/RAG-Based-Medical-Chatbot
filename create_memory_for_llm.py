from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os

# =====================================
# STEP 1: Load PDFs
# =====================================

DATA_PATH = "Data"


def load_pdfs(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


documents = load_pdfs(DATA_PATH)

print("Documents Loaded:", len(documents))

# =====================================
# STEP 2: Create Chunks
# =====================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=60
)

text_chunks = text_splitter.split_documents(documents)

print("Chunks Created:", len(text_chunks))

# =====================================
# STEP 3: Load Embedding Model
# =====================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded")

# =====================================
# STEP 4: Create FAISS Vector Store
# =====================================

DB_FAISS_PATH = "vectorstore/db_faiss"

os.makedirs("vectorstore", exist_ok=True)

db = FAISS.from_documents(
    documents=text_chunks,
    embedding=embedding_model
)

db.save_local(DB_FAISS_PATH)

print("FAISS Vector Store Created Successfully")
print("Saved at:", DB_FAISS_PATH)
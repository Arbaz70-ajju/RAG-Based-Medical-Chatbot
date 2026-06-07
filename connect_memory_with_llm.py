import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# Fixed imports to use standard langchain paths
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# =====================================
# STEP 1: Load Hugging Face Token
# =====================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

print("GROQ API Key Loaded Successfully")

# =====================================
# STEP 2: Load LLM
# =====================================
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.4

)

print("LLM Loaded Successfully")

# =====================================
# STEP 3: Load Embedding Model
# =====================================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded")

# =====================================
# STEP 4: Load FAISS Database
# =====================================
DB_FAISS_PATH = "vectorstore/db_faiss"

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

print("FAISS Database Loaded")

# =====================================
# STEP 5: Create Retriever
# =====================================
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# =====================================
# STEP 6: Prompt Template
# =====================================
prompt = ChatPromptTemplate.from_template(
"""
Use the provided context to answer the user's question.

If the answer is not present in the context,
say that you do not know.

Do not make up information.

Context:
{context}

Question:
{input}

Answer:
"""
)

# =====================================
# STEP 7: Create Document Chain
# =====================================
document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

# =====================================
# STEP 8: Create Retrieval Chain
# =====================================
retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain
)

print("Medical Chatbot Ready!")

# =====================================
# STEP 9: Chat Loop (Indentation Fixed)
# =====================================
while True:
    query = input("\nEnter Question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    try:
        response = retrieval_chain.invoke(
            {"input": query}
        )

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        
        print(response["answer"])
        
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
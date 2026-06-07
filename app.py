
import os
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import (
    create_retrieval_chain
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

# =====================================
# LOAD ENV VARIABLES
# =====================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺",
    layout="wide"
)
st.title("🩺 Medical Chatbot")
st.caption("Ask questions from your medical knowledge base.")

# =====================================
# SESSION STATES
# =====================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================================
# LOAD LLM
# =====================================
@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.5
    )

# =====================================
# LOAD EMBEDDINGS
# =====================================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# =====================================
# LOAD FAISS DATABASE
# =====================================
@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()

    db = FAISS.load_local(
        "vectorstore/db_faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db

# =====================================
# CREATE RAG CHAIN
# =====================================
@st.cache_resource
def create_chain():
    llm = load_llm()
    db = load_vectorstore()

    retriever = db.as_retriever(
        search_kwargs={"k": 3}
    )

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
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain

# =====================================
# SIDEBAR
# =====================================
with st.sidebar:
    st.title("🩺 Chat History")

    if len(st.session_state.chat_history) == 0:
        st.caption("No chats yet")

    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"• {chat}")

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# =====================================
# DISPLAY OLD CHAT
# =====================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================
# USER INPUT
# =====================================
user_question = st.chat_input("Ask your medical question...")

if user_question:
    timestamp = datetime.now().strftime("%H:%M")
    
    st.session_state.chat_history.append(
        f"[{timestamp}] {user_question}"
    )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Searching medical knowledge base..."):
        chain = create_chain()
        
        response = chain.invoke(
            {
                "input": user_question
            }
        )
        answer = response["answer"]

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


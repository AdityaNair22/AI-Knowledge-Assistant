"""
app.py
-------
Main Streamlit UI for the AI Knowledge Assistant.

Features:
- Upload PDFs and text notes directly from the sidebar
- Chat interface with memory (remembers last 10 exchanges)
- Shows which route the agent took (RAG / General / Clarify)
- Shows source document chunks used in RAG answers
- Optional emotion detection for empathetic responses
- Rebuild vector store button after uploading new files
"""

import os
import shutil
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from rag_pipeline import get_retriever, rebuild_vectorstore, get_document_count
from agent_flow import run_agent

# Optional — comment out if you don't want emotion detection
from emotion_model import detect_emotion, get_empathy_prefix


# ── Setup ──────────────────────────────────────────────────────────────────────

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)


# ── Load retriever (cached so it only loads once) ─────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_retriever():
    return get_retriever()


# ── Session state init ────────────────────────────────────────────────────────

if "memory" not in st.session_state:
    st.session_state.memory = []

if "retriever" not in st.session_state:
    st.session_state.retriever = load_retriever()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 AI Knowledge Assistant")
    st.markdown("---")

    # API Key input (if not in .env)
    if not os.getenv("GROQ_API_KEY"):
        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        st.caption("Get a free key at console.groq.com")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
    else:
        st.success("API Key loaded from .env")

    st.markdown("---")

    # File uploader
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or .txt notes",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = DATA_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} file(s)!")

    # Rebuild vector store
    if st.button("Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Rebuilding..."):
            st.cache_resource.clear()
            new_retriever = rebuild_vectorstore()
            st.session_state.retriever = new_retriever
        st.success("Knowledge base updated!")

    # Document count
    doc_count = get_document_count()
    st.markdown(f"**Documents loaded:** {doc_count}")

    # List files in data/
    if doc_count > 0:
        st.markdown("**Files in knowledge base:**")
        for f in DATA_DIR.iterdir():
            if f.suffix in [".pdf", ".txt"]:
                st.markdown(f"- {f.name}")

    st.markdown("---")

    # Clear chat
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.memory = []
        st.rerun()

    st.markdown("---")
    st.caption("Built with LangGraph + RAG + Streamlit")


# ── Main chat area ────────────────────────────────────────────────────────────

st.title("🧠 AI Knowledge Assistant")
st.markdown("Ask anything. I'll answer from your documents or my general knowledge.")

# Route badge colors
ROUTE_COLORS = {
    "rag":     ("📄", "Document Answer",  "#1a3a5c"),
    "general": ("🌐", "General Knowledge", "#1a3a2a"),
    "clarify": ("❓", "Needs Clarification", "#3a1a1a"),
    "":        ("💬", "Thinking...",        "#2a2a2a")
}

# Display chat history
for msg in st.session_state.memory:
    with st.chat_message("user"):
        st.write(msg["question"])

    with st.chat_message("assistant"):
        # Route badge
        route = msg.get("route", "")
        icon, label, color = ROUTE_COLORS.get(route, ROUTE_COLORS[""])
        st.markdown(
            f'<span style="background:{color};padding:3px 10px;border-radius:12px;'
            f'font-size:12px;color:#ccc">{icon} {label}</span>',
            unsafe_allow_html=True
        )
        st.write(msg["answer"])

        # Show sources if RAG
        if msg.get("route") == "rag" and msg.get("context"):
            with st.expander("View Source Chunks"):
                for i, doc in enumerate(msg["context"]):
                    source = doc.metadata.get("source", "Unknown")
                    page   = doc.metadata.get("page", "")
                    st.markdown(f"**Source {i+1}:** `{Path(source).name}` {f'— Page {page}' if page else ''}")
                    st.markdown(f"> {doc.page_content[:300]}...")
                    st.markdown("---")


# ── Chat input ────────────────────────────────────────────────────────────────

question = st.chat_input("Ask a question about your documents or anything else...")

if question:
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("Please enter your Groq API key in the sidebar first.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.write(question)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            prefix = ""

            # Run the LangGraph agent
            result = run_agent(
                question=question,
                memory=st.session_state.memory,
                retriever=st.session_state.retriever
            )

            answer = prefix + result["answer"]
            route  = result["route"]
            context = result["context"]

        # Show route badge
        icon, label, color = ROUTE_COLORS.get(route, ROUTE_COLORS[""])
        st.markdown(
            f'<span style="background:{color};padding:3px 10px;border-radius:12px;'
            f'font-size:12px;color:#ccc">{icon} {label}</span>',
            unsafe_allow_html=True
        )

        # Show answer
        st.write(answer)

        # Show sources if RAG
        if route == "rag" and context:
            with st.expander("View Source Chunks"):
                for i, doc in enumerate(context):
                    source = doc.metadata.get("source", "Unknown")
                    page   = doc.metadata.get("page", "")
                    st.markdown(f"**Source {i+1}:** `{Path(source).name}` {f'— Page {page}' if page else ''}")
                    st.markdown(f"> {doc.page_content[:300]}...")
                    st.markdown("---")

    # Save to memory (keep last 10)
    st.session_state.memory.append({
        "question": question,
        "answer":   answer,
        "route":    route,
        "context":  context
    })
    if len(st.session_state.memory) > 10:
        st.session_state.memory = st.session_state.memory[-10:]
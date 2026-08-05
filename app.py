"""
Streamlit front-end for the RAG Personal Study Assistant.

Conversation memory is persistent across restarts and days: each
"study session" you name is a `thread_id`, and LangGraph's SQLite
checkpointer saves/loads that session's full chat history automatically.

Run with:
    streamlit run app.py
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src import config
from src.ingest import ingest_pdf, load_existing_vectorstore
from src.graph import build_graph, ask, get_thread_messages

st.set_page_config(page_title="RAG Study Assistant", page_icon="📚", layout="wide")

st.title("📚 RAG Personal Study Assistant")
st.caption("Upload notes/textbook PDFs. Conversations are saved per study session, "
           "so you can continue tomorrow where you left off.")

# --- Auto-load an existing vector index on startup (survives restarts) ---
if "vectorstore" not in st.session_state:
    try:
        existing = load_existing_vectorstore()
        # A fresh/empty Chroma collection has 0 items — only use it if populated
        st.session_state.vectorstore = existing if existing._collection.count() > 0 else None
    except Exception:
        st.session_state.vectorstore = None

if "graph" not in st.session_state:
    st.session_state.graph = build_graph(st.session_state.vectorstore) if st.session_state.vectorstore else None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "default-session"

if "loaded_thread_id" not in st.session_state:
    st.session_state.loaded_thread_id = None

# --- Sidebar: ingestion + session management ---
with st.sidebar:
    st.header("1. Upload study material")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Index this PDF", type="primary"):
            save_path = config.UPLOAD_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Reading and indexing {uploaded_file.name}..."):
                try:
                    st.session_state.vectorstore = ingest_pdf(save_path)
                    st.session_state.graph = build_graph(st.session_state.vectorstore)
                    st.success(f"Indexed {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Failed to index file: {e}")

    st.divider()
    st.header("2. Study session")
    st.caption("Give each topic its own session name. Re-enter the same "
               "name later (even tomorrow) to continue that conversation.")

    thread_input = st.text_input("Session name", value=st.session_state.thread_id)
    if thread_input != st.session_state.thread_id:
        st.session_state.thread_id = thread_input
        st.session_state.loaded_thread_id = None  # force reload of history below
        st.rerun()

    if st.button("Start a brand-new session"):
        st.session_state.thread_id = "session-" + str(hash(thread_input))[-6:]
        st.session_state.loaded_thread_id = None
        st.rerun()

# --- Load this thread's saved history whenever the active thread changes ---
if st.session_state.graph and st.session_state.loaded_thread_id != st.session_state.thread_id:
    saved_messages = get_thread_messages(st.session_state.graph, st.session_state.thread_id)
    st.session_state.messages = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in saved_messages
    ]
    st.session_state.loaded_thread_id = st.session_state.thread_id

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Main: chat ---
st.header("3. Ask questions")
st.caption(f"Current session: **{st.session_state.thread_id}**")

if st.session_state.vectorstore is None:
    st.info("Upload and index a PDF from the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask something about your notes...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = ask(
                        st.session_state.graph,
                        question,
                        thread_id=st.session_state.thread_id,
                    )
                    st.markdown(result["answer"])

                    with st.expander("📖 Sources used"):
                        for src in result["sources"]:
                            st.markdown(
                                f"**{src['file']}** (page {src['page']}): "
                                f"_{src['snippet']}..._"
                            )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["answer"]}
                    )
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
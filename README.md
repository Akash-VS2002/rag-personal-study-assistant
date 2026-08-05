# 📚 RAG Personal Study Assistant

A Retrieval-Augmented Generation app that lets you upload PDF notes or
textbooks and ask questions grounded in that material, with cited sources.

## Features

* Upload and index PDF documents
* Automatic document chunking using RecursiveCharacterTextSplitter
* Local embeddings using Sentence Transformers (`all-MiniLM-L6-v2`)
* Persistent Chroma vector database
* Conversational memory with LangGraph + SQLite
* Context-aware question answering
* Source-grounded responses
* Streamlit-based interactive user interface

## Tech Stack

* Python
* Streamlit
* LangChain
* LangGraph
* ChromaDB
* Hugging Face Sentence Transformers
* Groq LLM
* SQLite
* PyPDFLoader

## Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* Large Language Model (LLM) Integration
* Semantic Search
* Vector Databases
* Embeddings
* Prompt Engineering
* Conversational AI
* LangGraph Workflows
* Document Processing
* Python Development

## Architecture

```
PDF upload
   │
   ▼
PyPDFLoader (load pages)          src/ingest.py
   │
   ▼
RecursiveCharacterTextSplitter (chunk)
   │
   ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
   │
   ▼
Chroma vector store (persisted locally)
   │
   ▼
LangGraph StateGraph                       src/graph.py
   │
   ├── Node: retrieve
   │     - rewrites follow-up questions using chat history
   │     - retrieves top-k chunks from Chroma
   │
   └── Node: generate
         - answers using retrieved context + full chat history
   │
   ▼
SqliteSaver checkpointer  ──►  checkpoints.db
   (loads/saves conversation state, keyed by thread_id
    = "study session name" you type in the sidebar)
```

* **Embeddings run locally** (no API cost, no rate limits) via
  `sentence-transformers`.
* **Generation runs on Groq** for fast, cheap inference.
* **Chroma** persists document vectors to disk.
* **LangGraph + SqliteSaver** persists *conversation memory* to disk,
  separately from the documents — so you can close the app, come back
  days later, reuse the same session name, and continue where you left off.

## Setup

Choose whichever matches your environment (shown in your editor: `pyproject.toml`

* `.python-version` means you're set up with **uv** — use option A).

**Option A — using `uv`:**

```bash
cd rag_study_assistant
uv sync
```

**Option B — using plain `venv` + pip:**

```bash
cd rag_study_assistant
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Then, either way, set up your API key:**

```bash
cp .env.example .env
# open .env and add your GROQ_API_KEY (get one free at https://console.groq.com/keys)
```

> ⚠️ `.env` contains your secret key and is already excluded via `.gitignore` —
> never commit it. Only `.env.example` (with a placeholder) should go to GitHub.

## Run

```bash
streamlit run app.py
# or, with uv:
uv run streamlit run app.py
```

Then open the local URL Streamlit prints:

1. Upload a PDF from the sidebar, click **Index this PDF**.
2. Give this topic a **session name** (e.g. `ml-basics`) in the sidebar.
3. Ask questions. Close the app any time.
4. Come back later, re-enter the same session name — your conversation
   history reloads automatically.

## Project structure

```
rag_study_assistant/
├── app.py                  # Streamlit UI — session management + chat
├── pyproject.toml            # project metadata + dependencies (uv)
├── requirements.txt           # dependencies (pip alternative)
├── .env.example
├── .env                         # your local secrets — gitignored, never pushed
├── .gitignore
├── data/uploads/                  # uploaded PDFs land here — gitignored except .gitkeep
├── vectorstore/                     # persisted Chroma DB — gitignored (regenerated locally)
├── checkpoints.db                    # SQLite conversation history — gitignored (regenerated locally)
└── src/
    ├── config.py              # paths, model names, chunk sizes, checkpoint DB path
    ├── ingest.py                # load -> split -> embed -> store
    ├── graph.py                  # LangGraph StateGraph + SqliteSaver (used by app.py)
    └── rag_chain.py                # v1 reference: plain LangChain chain, no persistence
```



## Possible next steps (v2 ideas)

* Swap basic RAG for **agentic RAG** with LangGraph: route between
  "answer from notes" vs "summarize chapter" vs "generate quiz"
  as separate tool-calling nodes.
* Add **conversation memory** so follow-up questions use chat history.
* Add **multi-file** filtering (ask questions scoped to one PDF).
* Add **evaluation** (RAGAS) to measure answer faithfulness/relevance.
* Deploy the backend as a **FastAPI** service and keep Streamlit (or
  swap in a React frontend) as a thin client.

## Interview talking points

* Why local embeddings + hosted LLM: cost/latency tradeoff.
* Why `RecursiveCharacterTextSplitter` with overlap: preserves context
  across chunk boundaries so answers aren't cut off mid-thought.
* Why Chroma over FAISS here: simpler persistence API, metadata
  filtering support, good for a single-machine app like this.
* How retrieval-grounding + a "say you don't know" instruction in the
  prompt reduces hallucination risk.

"""
Central configuration for the RAG Study Assistant.
All tunable parameters and environment-driven settings live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
CHECKPOINT_DB_PATH = BASE_DIR / "checkpoints.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# --- API keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Embedding model ---
# Small, fast, good quality sentence embedding model (runs locally, no API cost)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM ---
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = 0.2

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Retrieval ---
RETRIEVER_TOP_K = 4

# --- Conversation memory ---
# Cap how many past (role, content) messages are sent to the LLM as chat
# history, to bound token usage/cost as a session grows.
MAX_HISTORY_MESSAGES = 10

# --- Chroma collection ---
COLLECTION_NAME = "study_assistant_docs"
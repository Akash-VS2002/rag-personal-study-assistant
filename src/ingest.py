"""
Ingestion pipeline: PDF -> chunks -> embeddings -> Chroma vector store.
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pdf(file_path: Path) -> List[Document]:
    """Load a single PDF into LangChain Document objects (one per page)."""
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()
    for doc in documents:
        doc.metadata["source_file"] = file_path.name
    logger.info("Loaded %d pages from %s", len(documents), file_path.name)
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks suited for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks", len(chunks))
    return chunks


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the shared embedding model instance."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Embed chunks and persist them into a Chroma collection."""
    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.VECTORSTORE_DIR),
    )
    logger.info("Persisted %d chunks to Chroma at %s", len(chunks), config.VECTORSTORE_DIR)
    return vectorstore


def load_existing_vectorstore() -> Chroma:
    """Load a previously persisted Chroma collection without re-ingesting."""
    embeddings = get_embedding_model()
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.VECTORSTORE_DIR),
    )


def ingest_pdf(file_path: Path) -> Chroma:
    """Full pipeline: load -> split -> embed -> store. Returns the vectorstore."""
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    return build_vectorstore(chunks)

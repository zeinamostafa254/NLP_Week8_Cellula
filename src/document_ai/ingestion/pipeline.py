from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_chroma import Chroma

import logging
from document_ai.config import COLLECTION_NAME, VECTOR_DB_DIR
from document_ai.ingestion.loader import load_file, load_directory
from document_ai.ingestion.ocr_loader import OCRNotAvailableError
from document_ai.ingestion.parser import parse_documents
from document_ai.ingestion.cleaner import clean_documents
from document_ai.ingestion.chunker import chunk_documents
from document_ai.ingestion.embedder import get_embedding_model

logger = logging.getLogger(__name__)

def build_vector_store(chunks: list[Document]) -> Chroma:
    """Persist document chunks and their embeddings into Chroma."""
    logger.info(f"[Ingestion] Building vector store for {len(chunks)} chunks...")
    embeddings = get_embedding_model()

    if not chunks:
        raise ValueError("No chunks were produced. Check the input documents.")

    # Deterministic IDs prevent accidental duplicate records for the same chunk.
    ids = []
    for i, chunk in enumerate(chunks):
        doc_name = chunk.metadata.get("doc", "unknown")
        page = chunk.metadata.get("page", "na")
        ids.append(f"{doc_name}:{page}:{i}")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
    )

    vector_store.add_documents(documents=chunks, ids=ids)
    logger.info(f"[Ingestion] Successfully persisted to Chroma at {VECTOR_DB_DIR}")
    return vector_store


def ingest_files(paths: Iterable[str | Path]) -> dict:
    """Run load -> parse -> clean -> chunk -> embed/store.

    Each file is loaded independently: if one file fails (e.g. an image
    that can't be OCR'd because Tesseract isn't installed, or a corrupt
    PDF), it's skipped with a warning instead of aborting the whole batch.
    """
    paths = list(paths)
    raw_documents: list[Document] = []
    failed: list[dict] = []

    for path in paths:
        logger.info(f"[Ingestion] Loading file: {path}")
        try:
            raw_documents.extend(load_file(path))
        except OCRNotAvailableError as e:
            logger.warning(f"[Ingestion] Skipping '{path}' — OCR unavailable: {e}")
            failed.append({"file": str(path), "reason": str(e)})
        except Exception as e:
            logger.warning(f"[Ingestion] Skipping '{path}' — failed to load: {e}")
            failed.append({"file": str(path), "reason": str(e)})

    if not raw_documents:
        raise ValueError(
            "No documents could be loaded. "
            f"{len(failed)} file(s) failed: {[f['file'] for f in failed]}"
        )

    logger.info(f"[Ingestion] Parsing {len(raw_documents)} raw documents...")
    parsed = parse_documents(raw_documents)
    logger.info("[Ingestion] Cleaning documents...")
    cleaned = clean_documents(parsed)
    logger.info("[Ingestion] Chunking documents...")
    chunks = chunk_documents(cleaned)
    logger.info(f"[Ingestion] Generated {len(chunks)} chunks.")
    vector_store = build_vector_store(chunks)

    return {
        "files": len(paths),
        "documents": len(raw_documents),
        "chunks": len(chunks),
        "vector_db": str(VECTOR_DB_DIR),
        "collection": COLLECTION_NAME,
        "vector_store": vector_store,
        "failed": failed,
    }


def ingest_directory(directory: str | Path) -> dict:
    """Ingest all supported documents in a directory."""
    raw_documents = load_directory(directory)
    parsed = parse_documents(raw_documents)
    cleaned = clean_documents(parsed)
    chunks = chunk_documents(cleaned)
    vector_store = build_vector_store(chunks)

    return {
        "documents": len(raw_documents),
        "chunks": len(chunks),
        "vector_db": str(VECTOR_DB_DIR),
        "collection": COLLECTION_NAME,
        "vector_store": vector_store,
    }

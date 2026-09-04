"""
api/main.py
------------
FastAPI backend for the Document AI Assistant.

Endpoints:
  GET  /health                — liveness check
  POST /ingest                — upload & ingest one or more documents
  GET  /documents             — list ingested document names
  POST /query                 — full pipeline Q&A → FinalAnswer

Member 3 deliverable.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from document_ai.config import RAW_DOCS_DIR, VECTOR_DB_DIR
from document_ai.monitoring import setup_langsmith, get_langsmith_status
from document_ai.ingestion.pipeline import ingest_files
from document_ai.retriever.agent import RetrieverAgent
from document_ai.analyst.agent import AnalystAgent
from document_ai.analyst.retrieve_more import set_retriever_callback
from document_ai.answer.agent import AnswerAgent
from document_ai.orchestrator.orchestrator import Orchestrator
from document_ai.schemas.answer import FinalAnswer
from document_ai.logger import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

# Must run before any LangChain/LangGraph client is constructed so the
# very first LLM call is traced too. No-ops safely if not configured.
setup_langsmith()

app = FastAPI(
    title="Document AI Assistant",
    description=(
        "Multi-agent RAG system: Retriever → Analyst → Answer Agent, "
        "orchestrated by LangGraph."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy-initialised agent singletons ─────────────────────────────────
_retriever: RetrieverAgent | None = None
_analyst:   AnalystAgent   | None = None
_answer:    AnswerAgent     | None = None
_orch:      Orchestrator    | None = None


def _get_orchestrator() -> Orchestrator:
    global _retriever, _analyst, _answer, _orch
    if _orch is None:
        _retriever = RetrieverAgent()
        # Wire the retrieve_more @tool to call the real retriever
        set_retriever_callback(_retriever.retrieve)
        _analyst   = AnalystAgent(retriever_callback=_retriever.retrieve)
        _answer    = AnswerAgent()
        _orch      = Orchestrator(_retriever, _analyst, _answer)
    return _orch


# ── Request / Response models ──────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, Any]] = None
    max_loops: int = 3


class IngestResponse(BaseModel):
    status: str
    files_ingested: int
    total_chunks: int
    collection: str
    failed: Optional[List[Dict[str, Any]]] = None


class DocumentListResponse(BaseModel):
    documents: List[str]


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
def root():
    """Root status endpoint."""
    return {
        "status": "online",
        "message": "Document AI Assistant API is running.",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", tags=["System"])
def health_check():
    """Liveness check."""
    return {
        "status": "ok",
        "vector_db": str(VECTOR_DB_DIR),
        "langsmith": get_langsmith_status(),
    }


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents (PDF, DOCX, TXT) and run the full
    ingestion pipeline (load → parse → clean → chunk → embed → store).
    """
    if not files:
        logger.warning("Ingest called with no files.")
        raise HTTPException(status_code=400, detail="No files provided.")

    logger.info(f"Starting ingestion for {len(files)} files: {[f.filename for f in files]}")
    saved_paths: List[Path] = []
    try:
        for upload in files:
            suffix = Path(upload.filename or "doc").suffix or ".pdf"
            tmp = Path(tempfile.mktemp(suffix=suffix, dir=RAW_DOCS_DIR))
            with tmp.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved_paths.append(tmp)

        result = ingest_files(saved_paths)
        logger.info(f"Ingestion successful: {result['chunks']} chunks created.")
        failed = result.get("failed") or None
        files_ingested = len(saved_paths) - len(failed or [])
        return IngestResponse(
            status="success" if not failed else "partial_success",
            files_ingested=files_ingested,
            total_chunks=result["chunks"],
            collection=result["collection"],
            failed=failed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents():
    """List the filenames of all raw documents in the document store."""
    docs = sorted(p.name for p in RAW_DOCS_DIR.iterdir() if p.is_file())
    return DocumentListResponse(documents=docs)


@app.post("/query", response_model=FinalAnswer, tags=["Query"])
def query(request: QueryRequest):
    """
    Run the full pipeline:
      question → Retriever → Analyst (with optional feedback loop) → Answer Agent
    Returns a FinalAnswer with the response text, inline citations, and bibliography.
    """
    if not request.question.strip():
        logger.warning("Query called with empty question.")
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    logger.info(f"Received query: '{request.question}' (max_loops={request.max_loops})")
    try:
        orch = _get_orchestrator()
        answer: FinalAnswer = orch.run(
            question=request.question,
            filters=request.filters,
            max_loops=request.max_loops,
        )
        logger.info(f"Query completed successfully. Confidence: {answer.confidence:.2f}")
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

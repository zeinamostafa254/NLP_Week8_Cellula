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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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
from document_ai.report.agent import ReportAgent
from document_ai.voice.transcriber import VoiceTranscriber
from fastapi.responses import FileResponse
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

# Serve the frontend
frontend_dir = Path(__file__).parent.parent.parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

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

_report_agent: ReportAgent | None = None

def _get_report_agent() -> ReportAgent:
    global _report_agent
    if _report_agent is None:
        _report_agent = ReportAgent()
    return _report_agent

_transcriber: VoiceTranscriber | None = None

def _get_transcriber() -> VoiceTranscriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = VoiceTranscriber()
    return _transcriber

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

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    duration: float

# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["System"])
def root():
    """Root status endpoint - returns the frontend."""
    index_file = frontend_dir / "index.html"
    return index_file.read_text(encoding="utf-8")


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

@app.post("/transcribe", response_model=TranscriptionResponse, tags=["Voice"])
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Query(
        default=None,
        description="Force a language code (e.g. 'en', 'ar'). Omit to auto-detect.",
    ),
):
    """
    Transcribe a recorded voice question (wav/mp3/m4a/webm/ogg) to text
    using faster-whisper.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio data received.")
    logger.info(f"Transcribing voice query: {file.filename} ({len(audio_bytes)} bytes)")
    try:
        transcriber = _get_transcriber()
        result = transcriber.transcribe(
            audio_bytes, filename=file.filename or "audio.wav", language=language
        )
        if not result.text:
            raise HTTPException(
                status_code=422,
                detail="Could not detect any speech in the recording.",
            )
        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            language_probability=result.language_probability,
            duration=result.duration,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Transcription failed.")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

@app.post("/query/voice", response_model=FinalAnswer, tags=["Voice"])
async def query_voice(
    file: UploadFile = File(...),
    max_loops: int = Query(default=3),
    language: Optional[str] = Query(default=None),
):
    """
    Convenience endpoint that chains transcription + the full Q&A pipeline.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio data received.")
    transcriber = _get_transcriber()
    transcription = transcriber.transcribe(
        audio_bytes, filename=file.filename or "audio.wav", language=language
    )
    if not transcription.text:
        raise HTTPException(
            status_code=422, detail="Could not detect any speech in the recording."
        )
    logger.info(f"Voice query transcribed to: '{transcription.text}'")
    orch = _get_orchestrator()
    answer: FinalAnswer = orch.run(
        question=transcription.text, filters=None, max_loops=max_loops
    )
    return answer

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

@app.post("/generate_report", tags=["Report"])
def generate_report(answer: FinalAnswer):
    """
    Generates a PDF report from a FinalAnswer, extracting design 
    constraints from the user's question via LLM.
    """
    try:
        report_agent = _get_report_agent()
        pdf_path = report_agent.generate_pdf(answer, answer.question)
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename="AI_Report.pdf",
            headers={"Content-Disposition": "attachment; filename=AI_Report.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

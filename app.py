"""
app.py
-------
Streamlit UI for the Document AI Assistant.

Tabs:
  📄 Upload Documents  — ingest PDFs/DOCX/TXT via POST /ingest
  💬 Ask Questions     — chat-style Q&A via POST /query, shows
                         answer + citations + bibliography + metadata

Run:
    uv run streamlit run app.py

Requires the FastAPI backend to be running:
    uv run uvicorn document_ai.api.main:app --reload

Member 3 deliverable.
"""

import json
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"
SUPPORTED_TYPES = ["pdf", "docx", "txt", "png", "jpg", "jpeg", "bmp", "tiff", "webp"]

st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="🔍",
    layout="wide",
)

# ── Session state ──────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, Any]] = []
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs: List[str] = []


# ── Helpers ─────────────────────────────────────────────────────────────
def _api_health() -> bool:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _ingest(files) -> Dict[str, Any]:
    file_tuples = [("files", (f.name, f.read(), f.type or "application/octet-stream")) for f in files]
    r = httpx.post(f"{API_BASE}/ingest", files=file_tuples, timeout=120)
    r.raise_for_status()
    return r.json()


def _query(question: str, max_loops: int = 3) -> Dict[str, Any]:
    r = httpx.post(
        f"{API_BASE}/query",
        json={"question": question, "max_loops": max_loops},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def _list_docs() -> List[str]:
    try:
        r = httpx.get(f"{API_BASE}/documents", timeout=5)
        return r.json().get("documents", [])
    except Exception:
        return []


def _langsmith_status() -> Optional[Dict[str, Any]]:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=3)
        return r.json().get("langsmith")
    except Exception:
        return None


# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Document AI")
    st.caption("Multi-agent RAG: Retriever → Analyst → Answer")

    # API status
    if _api_health():
        st.success("API connected ✓")
    else:
        st.error("API offline — start the FastAPI server first.")
        st.code("uv run uvicorn document_ai.api.main:app --reload", language="bash")

    st.divider()

    # Retrieval settings
    st.subheader("⚙️ Settings")
    max_loops = st.slider("Max retrieval loops", min_value=1, max_value=5, value=3,
                          help="How many times the analyst can request more evidence")

    st.divider()

    # LangSmith monitoring status
    st.subheader("📡 Monitoring")
    ls_status = _langsmith_status()
    if ls_status and ls_status.get("enabled"):
        st.success(f"LangSmith tracing ON — project `{ls_status['project']}`")
    else:
        st.caption("LangSmith tracing off. Set LANGCHAIN_TRACING_V2=true and "
                    "LANGCHAIN_API_KEY in .env to enable it.")

    st.divider()

    # Ingested documents list
    st.subheader("📚 Ingested documents")
    docs = _list_docs()
    if docs:
        for d in docs:
            st.markdown(f"- `{d}`")
    else:
        st.caption("No documents ingested yet.")


# ── Main area — tabs ─────────────────────────────────────────────────────
tab_upload, tab_chat = st.tabs(["📄 Upload Documents", "💬 Ask Questions"])


# ────────────────────────────────────────────────────────────────────────
# TAB 1 — UPLOAD
# ────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.header("Upload & Ingest Documents")
    st.markdown(
        "Upload PDF, DOCX, TXT, or image files (PNG/JPG/JPEG/BMP/TIFF/WEBP). "
        "Images are run through OCR to extract their text automatically. "
        "Everything is then parsed, chunked, embedded, and stored in the vector "
        "database for retrieval."
    )

    uploaded = st.file_uploader(
        "Choose files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )

    if uploaded:
        st.info(f"{len(uploaded)} file(s) selected.")

    if st.button("⚡ Ingest Documents", disabled=not uploaded, type="primary"):
        with st.spinner("Ingesting — this may take a moment for large files…"):
            try:
                result = _ingest(uploaded)
                st.success(
                    f"✅ **{result['files_ingested']}** file(s) ingested — "
                    f"**{result['total_chunks']}** chunks stored in collection "
                    f"`{result['collection']}`."
                )
                if result.get("failed"):
                    with st.expander(f"⚠️ {len(result['failed'])} file(s) failed", expanded=True):
                        for f in result["failed"]:
                            st.markdown(f"- `{f['file']}` — {f['reason']}")
            except httpx.HTTPError as e:
                st.error(f"Ingestion failed: {e}")
            except Exception as e:
                st.error(f"Error: {e}")


# ────────────────────────────────────────────────────────────────────────
# TAB 2 — CHAT
# ────────────────────────────────────────────────────────────────────────
with tab_chat:
    st.header("Ask Questions")

    # Render chat history
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            if turn.get("citations"):
                with st.expander(f"📎 {len(turn['citations'])} citation(s)", expanded=False):
                    for c in turn["citations"]:
                        page_str = f", page {c['page']}" if c.get("page") else ""
                        st.markdown(
                            f"**[{c['ref_id']}]** `{c['doc']}`{page_str} "
                            f"*(score: {c['score']:.3f})*"
                        )
                        st.caption(c.get("snippet", ""))
                        st.divider()
            if turn.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    st.text(turn["sources"])
            # Metadata chips
            meta = turn.get("metadata", {})
            cols = st.columns(3)
            cols[0].metric("Confidence", f"{turn.get('confidence', 0):.0%}")
            cols[1].metric("Evidence chunks", meta.get("evidence_count", "—"))
            cols[2].metric("Retrieval loops", meta.get("iterations", "—"))

    # Input
    question = st.chat_input("Ask a question about your documents…")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving → Analyzing → Answering…"):
                try:
                    resp = _query(question, max_loops=max_loops)

                    answer_text = resp.get("answer", "")
                    citations   = resp.get("citations", [])
                    sources     = resp.get("sources", "")
                    confidence  = resp.get("confidence", 0.0)
                    metadata    = resp.get("metadata", {})

                    st.markdown(answer_text)

                    if citations:
                        with st.expander(f"📎 {len(citations)} citation(s)", expanded=True):
                            for c in citations:
                                page_str = f", page {c['page']}" if c.get("page") else ""
                                st.markdown(
                                    f"**[{c['ref_id']}]** `{c['doc']}`{page_str} "
                                    f"*(score: {c['score']:.3f})*"
                                )
                                st.caption(c.get("snippet", ""))
                                st.divider()

                    if sources:
                        with st.expander("📚 Sources", expanded=False):
                            st.text(sources)

                    cols = st.columns(3)
                    cols[0].metric("Confidence", f"{confidence:.0%}")
                    cols[1].metric("Evidence chunks", metadata.get("evidence_count", "—"))
                    cols[2].metric("Retrieval loops", metadata.get("iterations", "—"))

                    st.session_state.chat_history.append({
                        "question":   question,
                        "answer":     answer_text,
                        "citations":  citations,
                        "sources":    sources,
                        "confidence": confidence,
                        "metadata":   metadata,
                    })

                except httpx.HTTPStatusError as e:
                    detail = e.response.json().get("detail", str(e))
                    st.error(f"Error from API: {detail}")
                except httpx.ConnectError:
                    st.error("Cannot reach the API — is the FastAPI server running?")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

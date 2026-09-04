"""
monitoring.py
--------------
LangSmith tracing for the orchestrator (Retriever -> Analyst -> Answer,
run as a LangGraph StateGraph).

LangChain/LangGraph pick up tracing purely from environment variables
(LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT,
LANGCHAIN_ENDPOINT) — there's no code change needed in the agents
themselves. This module's only job is to:

  1. Set those env vars from config.py (so they live in one place: .env)
  2. Do it safely — if no API key is configured, tracing is simply left
     off. Nothing crashes, nothing else in the app changes.
  3. Give the API/CLI a single call (setup_langsmith()) to run once at
     startup, and a status dict (get_langsmith_status()) for the
     /health endpoint or CLI banner.

Call setup_langsmith() as early as possible — before any LangChain
client (ChatOpenAI, LangGraph, etc.) is constructed — so the very first
LLM call is traced too.
"""

import os
import logging

from document_ai.config import (
    LANGCHAIN_TRACING_V2,
    LANGCHAIN_API_KEY,
    LANGCHAIN_PROJECT,
    LANGCHAIN_ENDPOINT,
)

logger = logging.getLogger(__name__)

_configured = False
_enabled = False


def setup_langsmith() -> bool:
    """
    Enable LangSmith tracing if the user opted in via .env.
    Safe to call multiple times (idempotent) and safe to call even if
    langsmith isn't installed or no API key is set — it just no-ops.

    Returns True if tracing is active, False otherwise.
    """
    global _configured, _enabled

    if _configured:
        return _enabled

    _configured = True

    if not LANGCHAIN_TRACING_V2:
        logger.info("[Monitoring] LangSmith tracing disabled (LANGCHAIN_TRACING_V2=false).")
        _enabled = False
        return False

    if not LANGCHAIN_API_KEY:
        logger.warning(
            "[Monitoring] LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is missing — "
            "tracing stays OFF. Add your key to .env to enable it."
        )
        _enabled = False
        return False

    try:
        import langsmith  # noqa: F401  (import check only — confirms the package exists)
    except ImportError:
        logger.warning(
            "[Monitoring] LANGCHAIN_TRACING_V2=true but the `langsmith` package isn't "
            "installed — tracing stays OFF. Run `uv add langsmith`."
        )
        _enabled = False
        return False

    # These are the exact env vars LangChain/LangGraph read internally.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT

    logger.info(f"[Monitoring] LangSmith tracing ENABLED — project '{LANGCHAIN_PROJECT}'.")
    _enabled = True
    return True


def get_langsmith_status() -> dict:
    """Small status dict, handy for a /health endpoint or CLI banner."""
    return {
        "enabled": _enabled,
        "project": LANGCHAIN_PROJECT if _enabled else None,
        "endpoint": LANGCHAIN_ENDPOINT if _enabled else None,
    }

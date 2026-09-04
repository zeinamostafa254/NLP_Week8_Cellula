"""
analyst/retrieve_more.py
-------------------------
The "Search / Retrieve More Evidence" tool — implements the feedback-loop
arrow from Analyst Agent → Retriever Agent in the architecture diagram.

Original design by Alaa (Member 2) — fixed import (uses Zeina's Evidence
from schemas) and wrapped with @tool.

The callback interface lets Member 3's Orchestrator wire in the real
RetrieverAgent without creating a circular import.
"""

import json
from typing import Any, Callable, Dict, List, Protocol

from langchain_core.tools import tool

from document_ai.schemas.evidence import Evidence, EvidenceBundle


class RetrieverCallback(Protocol):
    """Matches RetrieverAgent.retrieve() signature: query → EvidenceBundle."""

    def __call__(self, query: str) -> EvidenceBundle:
        ...


# Module-level reference set by the Orchestrator at startup.
# AnswerAgent / AnalystAgent tool calls go through this.
_retriever_callback: RetrieverCallback | None = None


def set_retriever_callback(callback: RetrieverCallback) -> None:
    """Called by the Orchestrator to wire in the real RetrieverAgent."""
    global _retriever_callback
    _retriever_callback = callback


@tool
def retrieve_more_evidence(follow_up_query: str) -> str:
    """
    Send a refined follow-up query back to the Retriever Agent and return
    additional evidence chunks as a JSON array.
    Use this when the current evidence is insufficient to answer the question.
    """
    if _retriever_callback is None:
        return json.dumps([])
    bundle: EvidenceBundle = _retriever_callback(follow_up_query)
    return bundle.model_dump_json()


# ── Raw class kept for direct use inside AnalystAgent ──
class RetrieveMoreTool:
    """Direct-call wrapper used by AnalystAgent's internal feedback loop."""

    def __init__(self, retriever_callback: RetrieverCallback):
        self.retriever_callback = retriever_callback
        self.call_log: List[str] = []

    def request_more_evidence(self, follow_up_query: str) -> List[Evidence]:
        self.call_log.append(follow_up_query)
        bundle = self.retriever_callback(follow_up_query)
        return bundle.evidence


class MockRetrieverCallback:
    """Stand-in for tests / isolated development."""

    def __init__(self, canned_evidence: List[Evidence] | None = None):
        self.canned_evidence = canned_evidence or []

    def __call__(self, query: str) -> EvidenceBundle:
        return EvidenceBundle(
            query=query,
            rewritten_query=query,
            evidence=self.canned_evidence,
            source_count=len({e.doc for e in self.canned_evidence}),
        )

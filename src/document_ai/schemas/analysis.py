"""
schemas/analysis.py
--------------------
AnalystResult — the output contract between the Analyst Agent (Member 2)
and the Orchestrator / Answer Agent (Member 3).

status == "enough_evidence"   → ready for AnswerAgent
status == "need_more_evidence" → Orchestrator should loop back to Retriever
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from document_ai.schemas.evidence import Evidence


class AnalystResult(BaseModel):
    status: Literal["enough_evidence", "need_more_evidence"]

    # Populated when status == "enough_evidence"
    analysis: Optional[str] = None
    calculations: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    comparisons: Optional[Dict[str, Any]] = None

    # Always populated
    evidence_used: List[Evidence] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    follow_up_query: Optional[str] = None
    iterations: int = 0

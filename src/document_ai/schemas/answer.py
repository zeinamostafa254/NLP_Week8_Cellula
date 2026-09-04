"""
schemas/answer.py
------------------
FinalAnswer — the output contract between the Answer Agent (Member 3)
and the API / Streamlit UI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A single numbered citation pointing back to a retrieved chunk."""

    ref_id: int = Field(..., description="Inline reference number, e.g. 1 → [1]")
    doc: str = Field(..., description="Source document filename")
    page: Optional[int] = Field(None, description="1-based page number")
    score: float = Field(..., description="Relevance score from retrieval")
    snippet: str = Field(..., description="Short excerpt used as evidence")


class FinalAnswer(BaseModel):
    """The complete response returned to the user / API."""

    question: str
    answer: str = Field(
        ...,
        description="Full natural-language response with inline [1][2] citation markers",
    )
    citations: List[Citation] = Field(default_factory=list)
    sources: str = Field(
        default="",
        description="Formatted bibliography string (numbered list)",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average relevance score of evidence used",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

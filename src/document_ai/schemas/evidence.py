from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """One retrieved chunk plus the metadata needed for later citation."""

    doc: str = Field(..., description="Original document filename")
    page: Optional[int] = Field(None, description="1-based page number when known")
    score: float = Field(..., description="Normalized relevance score")
    content: str = Field(..., description="Retrieved text")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieval_method: str = "semantic"


class EvidenceBundle(BaseModel):
    """Stable interface between Member 1 (Retriever) and Member 2 (Analyst)."""

    query: str
    rewritten_query: str
    evidence: List[Evidence] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    source_count: int = 0

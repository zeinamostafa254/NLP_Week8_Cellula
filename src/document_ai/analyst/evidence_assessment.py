"""
analyst/evidence_assessment.py
--------------------------------
Evidence Assessment gate — decides whether the current evidence set is
sufficient to answer a question, or whether the Analyst should request
more from the Retriever.

This file was referenced in Alaa's agent.py and __init__.py but was never
created. Written from scratch (Member 3) based on the interface contract
defined in agent.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from document_ai.schemas.evidence import Evidence


@dataclass
class AssessmentConfig:
    """Configurable thresholds for the evidence gate."""

    min_evidence_count: int = 2
    """Minimum number of evidence chunks required."""

    min_avg_score: float = 0.5
    """Minimum average relevance score across all chunks."""

    min_source_count: int = 1
    """Minimum number of distinct source documents required."""

    require_numeric_content: bool = False
    """If True, at least one chunk must contain a digit (for numeric Q&A)."""


@dataclass
class Assessment:
    """Result produced by EvidenceAssessor.assess()."""

    enough_evidence: bool
    missing_information: List[str] = field(default_factory=list)
    suggested_follow_up_query: Optional[str] = None


class EvidenceAssessor:
    """
    Determines whether the current evidence set is sufficient to answer
    the user's question.

    Rules (all configurable via AssessmentConfig):
      1. Enough chunks present (count ≥ min_evidence_count)
      2. Evidence is relevant enough (avg score ≥ min_avg_score)
      3. Covers enough distinct sources (source count ≥ min_source_count)
      4. (Optional) Contains at least one numeric mention
    """

    def __init__(self, config: Optional[AssessmentConfig] = None):
        self.config = config or AssessmentConfig()

    def assess(self, question: str, evidence: List[Evidence]) -> Assessment:
        missing: List[str] = []

        if not evidence:
            return Assessment(
                enough_evidence=False,
                missing_information=["No evidence retrieved at all."],
                suggested_follow_up_query=question,
            )

        # --- Rule 1: count ---
        if len(evidence) < self.config.min_evidence_count:
            missing.append(
                f"Only {len(evidence)} evidence chunk(s) found "
                f"(need ≥ {self.config.min_evidence_count})."
            )

        # --- Rule 2: relevance score ---
        avg_score = sum(e.score for e in evidence) / len(evidence)
        if avg_score < self.config.min_avg_score:
            missing.append(
                f"Average relevance score {avg_score:.2f} is below threshold "
                f"{self.config.min_avg_score}."
            )

        # --- Rule 3: source diversity ---
        source_count = len({e.doc for e in evidence})
        if source_count < self.config.min_source_count:
            missing.append(
                f"Evidence comes from only {source_count} document(s) "
                f"(need ≥ {self.config.min_source_count})."
            )

        # --- Rule 4 (optional): numeric content ---
        if self.config.require_numeric_content:
            has_numeric = any(
                any(ch.isdigit() for ch in e.content) for e in evidence
            )
            if not has_numeric:
                missing.append("No numeric data found in evidence (required for this question).")

        enough = len(missing) == 0

        follow_up: Optional[str] = None
        if not enough:
            # Build a simple follow-up query hint for the Retriever
            follow_up = f"Find more detailed information about: {question}"

        return Assessment(
            enough_evidence=enough,
            missing_information=missing,
            suggested_follow_up_query=follow_up,
        )

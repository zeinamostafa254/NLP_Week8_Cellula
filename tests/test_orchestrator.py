"""
tests/test_orchestrator.py
---------------------------
End-to-end integration test for the LangGraph Orchestrator.
All three agents are mocked so no real LLM or vector DB is needed.
"""

import pytest

from document_ai.schemas.evidence import Evidence, EvidenceBundle
from document_ai.schemas.analysis import AnalystResult
from document_ai.schemas.answer import FinalAnswer, Citation
from document_ai.orchestrator.orchestrator import OrchestratorGraph


# ── Mocks ─────────────────────────────────────────────────────────────────

class MockRetriever:
    def retrieve(self, query, filters=None, **kwargs):
        return EvidenceBundle(
            query=query,
            rewritten_query=query,
            evidence=[
                Evidence(doc="TestDoc.pdf", page=1, score=0.9, content="CNN acc=92%"),
                Evidence(doc="TestDoc2.pdf", page=2, score=0.85, content="RNN acc=89%"),
            ],
            source_count=2,
        )


class MockAnalyst:
    def __init__(self, status="enough_evidence"):
        self._status = status
        self.call_count = 0

    def analyze(self, question, evidence_bundle):
        self.call_count += 1
        return AnalystResult(
            status=self._status,
            analysis="Mocked analysis text",
            evidence_used=evidence_bundle.evidence,
            iterations=self.call_count,
        )


class MockAnswerAgent:
    def answer(self, question, analyst_result):
        return FinalAnswer(
            question=question,
            answer=f"Mocked answer for: {question}",
            citations=[
                Citation(ref_id=1, doc="TestDoc.pdf", page=1, score=0.9, snippet="CNN acc=92%")
            ],
            sources="[1] TestDoc.pdf, page 1 (relevance: 0.900)",
            confidence=0.875,
            metadata={"analyst_status": analyst_result.status, "iterations": analyst_result.iterations},
        )


# ── Tests ─────────────────────────────────────────────────────────────────

def test_orchestrator_happy_path():
    """Single pass — analyst returns enough_evidence immediately."""
    orch = OrchestratorGraph(
        retriever=MockRetriever(),
        analyst=MockAnalyst(status="enough_evidence"),
        answer_agent=MockAnswerAgent(),
    )
    result = orch.run("What is the CNN accuracy?")

    assert isinstance(result, FinalAnswer)
    assert "CNN accuracy" in result.question
    assert result.confidence > 0
    assert len(result.citations) >= 1


def test_orchestrator_feedback_loop():
    """Analyst returns need_more_evidence once, then enough_evidence → 2 retrieve calls."""

    call_log = []

    class CountingRetriever:
        def retrieve(self, query, filters=None, **kwargs):
            call_log.append(query)
            return EvidenceBundle(
                query=query,
                rewritten_query=query,
                evidence=[Evidence(doc="Doc.pdf", page=1, score=0.9, content="data")],
                source_count=1,
            )

    class AlternatingAnalyst:
        def __init__(self):
            self._calls = 0

        def analyze(self, question, evidence_bundle):
            self._calls += 1
            status = "need_more_evidence" if self._calls == 1 else "enough_evidence"
            return AnalystResult(
                status=status,
                analysis="analysis" if status == "enough_evidence" else None,
                evidence_used=evidence_bundle.evidence,
                follow_up_query="more specific query",
                iterations=self._calls,
            )

    orch = OrchestratorGraph(
        retriever=CountingRetriever(),
        analyst=AlternatingAnalyst(),
        answer_agent=MockAnswerAgent(),
    )
    result = orch.run("Multi-hop question", max_loops=3)

    assert isinstance(result, FinalAnswer)
    assert len(call_log) == 2  # retrieve called twice


def test_orchestrator_max_loops_respected():
    """Analyst always returns need_more_evidence — orchestrator should still terminate."""
    call_log = []

    class AlwaysNeedMoreRetriever:
        def retrieve(self, query, filters=None, **kwargs):
            call_log.append(query)
            return EvidenceBundle(
                query=query, rewritten_query=query,
                evidence=[Evidence(doc="Doc.pdf", page=1, score=0.5, content="x")],
                source_count=1,
            )

    orch = OrchestratorGraph(
        retriever=AlwaysNeedMoreRetriever(),
        analyst=MockAnalyst(status="need_more_evidence"),
        answer_agent=MockAnswerAgent(),
    )
    result = orch.run("Unanswerable question", max_loops=2)

    assert isinstance(result, FinalAnswer)
    assert len(call_log) <= 3  # initial + max_loops iterations

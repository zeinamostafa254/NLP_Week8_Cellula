"""
tests/test_answer.py
---------------------
Pytest suite for the Answer Agent and its three @tools.
All tests run without a real LLM — the AnswerAgent is monkeypatched.
"""

import json

import pytest

from document_ai.schemas.evidence import Evidence
from document_ai.schemas.analysis import AnalystResult
from document_ai.schemas.answer import FinalAnswer, Citation

from document_ai.answer.citation_formatter import build_citations, format_citations
from document_ai.answer.source_formatter import build_sources, format_sources
from document_ai.answer.response_formatter import assemble_response


# ── Sample fixtures ──────────────────────────────────────────────────────

SAMPLE_EVIDENCE = [
    {"doc": "PaperA.pdf", "page": 3, "score": 0.92, "content": "CNN achieved 92% accuracy on CIFAR-10."},
    {"doc": "PaperB.pdf", "page": 7, "score": 0.87, "content": "RNN achieved 89% accuracy on the same benchmark."},
]

SAMPLE_ANALYST_RESULT = AnalystResult(
    status="enough_evidence",
    analysis="PaperA.pdf reports CNN accuracy of 92%. PaperB.pdf reports RNN accuracy of 89%.",
    calculations=[{"operation": "average", "inputs": [92, 89], "result": 90.5}],
    evidence_used=[
        Evidence(doc="PaperA.pdf", page=3, score=0.92, content="CNN achieved 92% accuracy on CIFAR-10."),
        Evidence(doc="PaperB.pdf", page=7, score=0.87, content="RNN achieved 89% accuracy on the same benchmark."),
    ],
    iterations=1,
)


# ── CitationFormatter ────────────────────────────────────────────────────

def test_build_citations_count():
    citations = build_citations(SAMPLE_EVIDENCE)
    assert len(citations) == 2


def test_build_citations_ref_ids():
    citations = build_citations(SAMPLE_EVIDENCE)
    assert citations[0]["ref_id"] == 1
    assert citations[1]["ref_id"] == 2


def test_build_citations_snippet_truncated():
    long_evidence = [{"doc": "doc.pdf", "page": 1, "score": 0.9, "content": "x" * 300}]
    citations = build_citations(long_evidence)
    assert len(citations[0]["snippet"]) <= 204  # 200 + "…"
    assert citations[0]["snippet"].endswith("…")


def test_format_citations_tool():
    """@tool version — takes JSON string, returns JSON string."""
    result = format_citations.invoke({"evidence_json": json.dumps(SAMPLE_EVIDENCE)})
    citations = json.loads(result)
    assert len(citations) == 2
    assert citations[0]["doc"] == "PaperA.pdf"


# ── SourceFormatter ──────────────────────────────────────────────────────

def test_build_sources_format():
    citations = build_citations(SAMPLE_EVIDENCE)
    sources = build_sources(citations)
    assert "[1]" in sources
    assert "PaperA.pdf" in sources
    assert "[2]" in sources
    assert "PaperB.pdf" in sources


def test_build_sources_empty():
    assert build_sources([]) == "No sources available."


def test_format_sources_tool():
    citations = build_citations(SAMPLE_EVIDENCE)
    result = format_sources.invoke({"citations_json": json.dumps(citations)})
    assert "[1]" in result


# ── ResponseFormatter ────────────────────────────────────────────────────

def test_assemble_response_contains_sources():
    citations = build_citations(SAMPLE_EVIDENCE)
    response = assemble_response(
        "PaperA.pdf is great.",
        citations,
    )
    assert "**Sources:**" in response
    assert "[1] PaperA.pdf" in response
    assert "page 3" in response


def test_assemble_response_doc_marker():
    citations = build_citations(SAMPLE_EVIDENCE)
    response = assemble_response("PaperA.pdf achieved 92%.", citations)
    assert "PaperA.pdf[1]" in response


# ── AnswerAgent integration (LLM monkeypatched) ───────────────────────────

def test_answer_agent_returns_final_answer(monkeypatch):
    from document_ai.answer.agent import AnswerAgent

    class FakeLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            class FakeMsg:
                content = ""
                tool_calls = []
            return FakeMsg()

    monkeypatch.setattr("document_ai.answer.agent.get_llm", lambda: FakeLLM())

    agent = AnswerAgent()
    result = agent.answer("Compare CNN and RNN accuracy.", SAMPLE_ANALYST_RESULT)

    assert isinstance(result, FinalAnswer)
    assert result.question == "Compare CNN and RNN accuracy."
    assert len(result.citations) == 2
    assert result.confidence > 0
    assert result.sources != ""

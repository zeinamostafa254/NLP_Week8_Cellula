"""
tests/test_analyst.py
----------------------
Pytest suite for the Analyst Agent and its tools.
Ported from Alaa's test_agent.py (was inside member2/) with:
  - Proper pytest structure
  - Uses Zeina's Evidence/EvidenceBundle schemas
  - MockRetrieverCallback updated to return EvidenceBundle
"""

import pytest

from document_ai.schemas.evidence import Evidence, EvidenceBundle
from document_ai.analyst.agent import AnalystAgent
from document_ai.analyst.retrieve_more import MockRetrieverCallback
from document_ai.analyst.calculator import Calculator, CalculatorError
from document_ai.analyst.evidence_assessment import AssessmentConfig, EvidenceAssessor
from document_ai.analyst.table_extractor import TableExtractor


# ── Fixtures ─────────────────────────────────────────────────────────────

def _make_bundle(evidence_dicts: list) -> EvidenceBundle:
    evidence = [
        Evidence(
            doc=d.get("doc", d.get("document", "unknown")),
            page=d.get("page"),
            score=float(d.get("score", 0.5)),
            content=d.get("content", ""),
        )
        for d in evidence_dicts
    ]
    return EvidenceBundle(
        query="test",
        rewritten_query="test",
        evidence=evidence,
        source_count=len({e.doc for e in evidence}),
    )


# ── Calculator ───────────────────────────────────────────────────────────

def test_calculator_average():
    calc = Calculator()
    result = calc.average([92, 95, 89])
    assert abs(result["result"] - 92.0) < 0.01


def test_calculator_evaluate():
    calc = Calculator()
    result = calc.evaluate("(92+95+89)/3")
    assert abs(result["result"] - 92.0) < 0.01


def test_calculator_division_by_zero():
    calc = Calculator()
    with pytest.raises(CalculatorError):
        calc.evaluate("10/0")


# ── EvidenceAssessor ──────────────────────────────────────────────────────

def test_assessor_enough_evidence():
    config = AssessmentConfig(min_evidence_count=2, min_avg_score=0.5, min_source_count=1)
    assessor = EvidenceAssessor(config)
    evidence = [
        Evidence(doc="A.pdf", page=1, score=0.9, content="CNN acc=92%"),
        Evidence(doc="B.pdf", page=2, score=0.85, content="CNN acc=95%"),
    ]
    result = assessor.assess("What is the CNN accuracy?", evidence)
    assert result.enough_evidence


def test_assessor_low_score_fails():
    config = AssessmentConfig(min_avg_score=0.8)
    assessor = EvidenceAssessor(config)
    evidence = [
        Evidence(doc="A.pdf", page=1, score=0.3, content="some text"),
    ]
    result = assessor.assess("question", evidence)
    assert not result.enough_evidence
    assert result.suggested_follow_up_query is not None


# ── TableExtractor ───────────────────────────────────────────────────────

def test_table_extractor_markdown():
    extractor = TableExtractor()
    text = "| Model | Accuracy |\n|---|---|\n| CNN | 92% |\n| RNN | 89% |"
    tables = extractor.extract_from_text(text)
    assert len(tables) == 1
    assert tables[0]["headers"] == ["Model", "Accuracy"]
    assert len(tables[0]["rows"]) == 2


def test_table_extractor_no_table():
    extractor = TableExtractor()
    tables = extractor.extract_from_text("No tables in this paragraph of text.")
    assert tables == []


# ── AnalystAgent (with mock retriever, no real LLM) ───────────────────────

def test_analyst_enough_evidence_no_llm(monkeypatch):
    """Verify the assessment gate fires correctly before LLM is called."""
    bundle = _make_bundle([
        {"doc": "PaperA.pdf", "page": 4, "score": 0.93, "content": "CNN acc=92%"},
        {"doc": "PaperB.pdf", "page": 6, "score": 0.90, "content": "CNN acc=95%"},
        {"doc": "PaperC.pdf", "page": 3, "score": 0.88, "content": "CNN acc=89%"},
    ])

    from document_ai.schemas.analysis import AnalystResult
    from document_ai.analyst.evidence_assessment import Assessment

    # Mock LLM so no API key needed
    class FakeLLM:
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class R:
                content = "mocked"
                tool_calls = []
            return R()

    monkeypatch.setattr("document_ai.analyst.agent.get_llm", lambda: FakeLLM())

    def fake_tool_loop(self, question, evidence, assessment, iterations):
        return AnalystResult(
            status="enough_evidence",
            analysis="Mocked analysis",
            evidence_used=evidence,
            iterations=iterations,
        )

    monkeypatch.setattr(AnalystAgent, "_run_tool_loop", fake_tool_loop)

    agent = AnalystAgent(
        retriever_callback=MockRetrieverCallback(),
        assessment_config=AssessmentConfig(min_evidence_count=2, min_avg_score=0.5),
    )
    result = agent.analyze("What is the average CNN accuracy?", bundle)
    assert result.status == "enough_evidence"


def test_analyst_need_more_evidence_no_llm(monkeypatch):
    """Thin evidence → assessor requests more → mock has nothing → need_more_evidence."""
    bundle = _make_bundle([
        {"doc": "PaperA.pdf", "page": 1, "score": 0.1, "content": "irrelevant text"},
    ])

    from document_ai.schemas.analysis import AnalystResult

    class FakeLLM:
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class R:
                content = "mocked"
                tool_calls = []
            return R()

    monkeypatch.setattr("document_ai.analyst.agent.get_llm", lambda: FakeLLM())

    def fake_tool_loop(self, question, evidence, assessment, iterations):
        return AnalystResult(
            status="need_more_evidence",
            evidence_used=evidence,
            missing_information=assessment.missing_information,
            iterations=iterations,
        )

    monkeypatch.setattr(AnalystAgent, "_run_tool_loop", fake_tool_loop)

    agent = AnalystAgent(
        retriever_callback=MockRetrieverCallback(),
        assessment_config=AssessmentConfig(min_avg_score=0.8),
        max_iterations=1,
    )
    result = agent.analyze("Find the F1 score.", bundle)
    assert result.status == "need_more_evidence"
    assert len(result.missing_information) > 0


from src.document_ai.retriever.metadata_filter import filter_results
from src.document_ai.retriever.reranker import Reranker
from src.document_ai.retriever.context_selector import ContextSelector


def test_metadata_filter():
    results = [
        {"content": "a", "metadata": {"doc": "a.pdf", "page": 1}, "score": 0.9},
        {"content": "b", "metadata": {"doc": "b.pdf", "page": 2}, "score": 0.8},
    ]
    out = filter_results(results, {"doc": "a.pdf"})
    assert len(out) == 1
    assert out[0]["metadata"]["doc"] == "a.pdf"


def test_reranker():
    results = [{"content": "BERT base model", "metadata": {}, "score": 0.5}]
    out = Reranker().rerank("BERT", results)
    assert out[0]["score"] > 0


def test_context_selector():
    results = [
        {"content": "a", "metadata": {"doc": "x", "page": 1}, "score": 0.9},
        {"content": "b", "metadata": {"doc": "x", "page": 1}, "score": 0.8},
        {"content": "c", "metadata": {"doc": "y", "page": 2}, "score": 0.7},
    ]
    out = ContextSelector().select(results, k=2)
    assert len(out) == 2

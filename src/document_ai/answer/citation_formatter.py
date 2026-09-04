"""
answer/citation_formatter.py
------------------------------
@tool: Converts a list of Evidence objects (as JSON) into numbered
Citation objects with inline reference IDs ([1], [2], ...).

Member 3 deliverable.
"""

import json
from typing import Any, Dict, List

from langchain_core.tools import tool


@tool
def format_citations(evidence_json: str) -> str:
    """
    Convert a JSON array of evidence chunks into numbered citation objects.

    Each citation gets a sequential ref_id used for inline markers like [1], [2].
    Input: JSON array of objects with keys: doc, page, score, content.
    Output: JSON array of citations with keys: ref_id, doc, page, score, snippet.

    Example input:
      [{"doc": "paper.pdf", "page": 3, "score": 0.92, "content": "CNN achieved 92% accuracy..."}]
    Example output:
      [{"ref_id": 1, "doc": "paper.pdf", "page": 3, "score": 0.92, "snippet": "CNN achieved 92%..."}]
    """
    try:
        evidence_list: List[Dict[str, Any]] = json.loads(evidence_json)
        citations = []
        for i, ev in enumerate(evidence_list, start=1):
            content = ev.get("content", "")
            # Keep snippet short — first 200 chars
            snippet = content[:200].strip()
            if len(content) > 200:
                snippet += "…"
            citations.append({
                "ref_id": i,
                "doc": ev.get("doc", "unknown"),
                "page": ev.get("page"),
                "score": round(float(ev.get("score", 0.0)), 3),
                "snippet": snippet,
            })
        return json.dumps(citations)
    except Exception as e:
        return f"Error: {e}"


# ── Raw function for direct use inside AnswerAgent ──
def build_citations(evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Direct (non-tool) version for internal AnswerAgent use."""
    citations = []
    for i, ev in enumerate(evidence_list, start=1):
        content = ev.get("content", "")
        snippet = content[:200].strip() + ("…" if len(content) > 200 else "")
        citations.append({
            "ref_id": i,
            "doc": ev.get("doc", "unknown"),
            "page": ev.get("page"),
            "score": round(float(ev.get("score", 0.0)), 3),
            "snippet": snippet,
        })
    return citations

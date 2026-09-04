"""
answer/source_formatter.py
----------------------------
@tool: Converts a list of Citation objects into a formatted bibliography string.

Member 3 deliverable.
"""

import json
from typing import Any, Dict, List

from langchain_core.tools import tool


@tool
def format_sources(citations_json: str) -> str:
    """
    Convert a JSON array of citation objects into a numbered bibliography string.

    Input: JSON array of citation objects (ref_id, doc, page, score, snippet).
    Output: A formatted multi-line string, e.g.:
      [1] paper.pdf, page 3 (relevance: 0.920)
      [2] report.pdf, page 7 (relevance: 0.875)

    This string is suitable for appending at the end of the final answer
    as a "Sources" or "References" section.
    """
    try:
        citations: List[Dict[str, Any]] = json.loads(citations_json)
        if not citations:
            return "No sources available."
        lines = []
        for c in citations:
            ref_id = c.get("ref_id", "?")
            doc = c.get("doc", "unknown")
            page = c.get("page")
            score = c.get("score", 0.0)
            page_str = f", page {page}" if page is not None else ""
            lines.append(f"[{ref_id}] {doc}{page_str} (relevance: {score:.3f})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


# ── Raw function for direct use inside AnswerAgent ──
def build_sources(citations: List[Dict[str, Any]]) -> str:
    """Direct (non-tool) version for internal AnswerAgent use."""
    if not citations:
        return "No sources available."
    lines = []
    for c in citations:
        ref_id = c.get("ref_id", "?")
        doc = c.get("doc", "unknown")
        page = c.get("page")
        score = c.get("score", 0.0)
        page_str = f", page {page}" if page is not None else ""
        lines.append(f"[{ref_id}] {doc}{page_str} (relevance: {score:.3f})")
    return "\n".join(lines)

"""
answer/response_formatter.py
------------------------------
@tool: Weaves an analysis text with inline citation markers [1][2]
and assembles the final clean response.

Member 3 deliverable.
"""

import json
import re
from typing import Any, Dict, List

from langchain_core.tools import tool


@tool
def format_response(analysis: str, citations_json: str) -> str:
    """
    Produce the final answer text by inserting inline citation markers into
    the analysis and appending a Sources section.

    How citation markers are inserted:
      - The LLM's analysis may already contain markers like [1], [doc.pdf].
      - Any mention of a document name (e.g. "PaperA.pdf") is replaced with
        the corresponding [ref_id] marker from the citations list.
      - A "Sources:" section is appended at the end.

    Input:
      analysis:       the written analysis string from the Analyst Agent.
      citations_json: JSON array of citation objects (ref_id, doc, page, snippet, score).
      question:       the original user question (used to open the answer).

    Output: complete formatted answer string.
    """
    try:
        citations: List[Dict[str, Any]] = json.loads(citations_json)
    except Exception:
        citations = []

    # Build doc-name → [ref_id] map
    doc_ref_map = {c["doc"]: f"[{c['ref_id']}]" for c in citations}

    # Replace document name mentions in analysis text with inline markers
    formatted = analysis
    for doc_name, marker in sorted(doc_ref_map.items(), key=lambda x: -len(x[0])):
        # Replace exact doc filename mentions (case-sensitive)
        formatted = formatted.replace(doc_name, f"{doc_name}{marker}")

    # Build sources block
    if citations:
        source_lines = []
        for c in citations:
            page_str = f", page {c['page']}" if c.get("page") is not None else ""
            source_lines.append(
                f"[{c['ref_id']}] {c['doc']}{page_str} (relevance: {c['score']:.3f})"
            )
        sources_block = "\n\n**Sources:**\n" + "\n".join(source_lines)
    else:
        sources_block = ""

    return formatted + sources_block


# ── Raw function for direct use inside AnswerAgent ──
def assemble_response(analysis: str, citations: List[Dict[str, Any]]) -> str:
    """Direct (non-tool) version for internal AnswerAgent use."""
    doc_ref_map = {c["doc"]: f"[{c['ref_id']}]" for c in citations}
    formatted = analysis
    for doc_name, marker in sorted(doc_ref_map.items(), key=lambda x: -len(x[0])):
        formatted = formatted.replace(doc_name, f"{doc_name}{marker}")
    if citations:
        source_lines = [
            f"[{c['ref_id']}] {c['doc']}"
            + (f", page {c['page']}" if c.get("page") is not None else "")
            + f" (relevance: {c['score']:.3f})"
            for c in citations
        ]
        formatted += "\n\n**Sources:**\n" + "\n".join(source_lines)
    return formatted

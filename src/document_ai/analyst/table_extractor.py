"""
analyst/table_extractor.py
---------------------------
Finds structured tables inside evidence chunk text and converts them into
structured rows/columns the Analyst can reason over.

Handles:
  1. Markdown pipe tables:   | Model | Acc |\n|---|---|\n| CNN | 92% |
  2. Whitespace/tab aligned: Model     Acc\n CNN       92%

Original logic by Alaa (Member 2) — fixed import (uses Zeina's Evidence schema,
field renamed ev.document → ev.doc) and wrapped with @tool.
"""

import json
import re
from typing import Any, Dict, List

from langchain_core.tools import tool

from document_ai.schemas.evidence import Evidence


@tool
def extract_tables_from_text(text: str) -> str:
    """
    Parse markdown-style or whitespace-aligned tables from a text chunk.
    Returns a JSON string containing a list of table objects, each with
    'headers' (list of column names) and 'rows' (list of dicts).
    Returns '[]' if no tables are found.
    """
    extractor = TableExtractor()
    tables = extractor.extract_from_text(text)
    return json.dumps(tables)


# ── Raw class kept for direct use inside AnalystAgent ──
class TableExtractor:
    def extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        tables = self._extract_markdown_tables(text)
        if not tables:
            tables = self._extract_whitespace_tables(text)
        return tables

    def extract_from_evidence(self, evidence_list: List[Evidence]) -> List[Dict[str, Any]]:
        """Run extraction across a list of evidence chunks, tagging each
        found table with its source document/page for later citation."""
        results = []
        for ev in evidence_list:
            for table in self.extract_from_text(ev.content):
                # ev.doc is Zeina's field name (was ev.document in Alaa's original)
                table["source"] = {"doc": ev.doc, "page": ev.page}
                results.append(table)
        return results

    # ---- markdown pipe tables ----------------------------------------
    def _extract_markdown_tables(self, text: str) -> List[Dict[str, Any]]:
        lines = text.splitlines()
        tables = []
        i = 0
        while i < len(lines):
            if "|" in lines[i] and i + 1 < len(lines) and re.match(
                r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]
            ):
                header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                row_idx = i + 2
                rows = []
                while row_idx < len(lines) and "|" in lines[row_idx]:
                    cells = [c.strip() for c in lines[row_idx].strip().strip("|").split("|")]
                    if len(cells) == len(header):
                        rows.append(dict(zip(header, cells)))
                    row_idx += 1
                if rows:
                    tables.append({"headers": header, "rows": rows})
                i = row_idx
            else:
                i += 1
        return tables

    # ---- whitespace-aligned tables -----------------------------------
    def _extract_whitespace_tables(self, text: str) -> List[Dict[str, Any]]:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        tables = []
        block: List[List[str]] = []

        def flush():
            if len(block) >= 2:
                header, *rows = block
                if all(len(r) == len(header) for r in rows):
                    tables.append(
                        {"headers": header, "rows": [dict(zip(header, r)) for r in rows]}
                    )

        for line in lines:
            cells = re.split(r"\s{2,}|\t", line.strip())
            if len(cells) >= 2 and any(
                re.search(r"\d", c) or len(c) < 20 for c in cells
            ):
                block.append(cells)
            else:
                flush()
                block = []
        flush()
        return tables

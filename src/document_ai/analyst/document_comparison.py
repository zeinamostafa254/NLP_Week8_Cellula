"""
analyst/document_comparison.py
--------------------------------
Compares information across multiple source documents: similarities,
differences, and (when numeric) which document "wins" on a given metric.

Original logic by Alaa (Member 2) — fixed import + field name (ev.document → ev.doc)
and wrapped with @tool.
"""

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from document_ai.schemas.evidence import Evidence

_NUM_NEAR_LABEL = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z0-9_\- ]{1,30}?)\s*[:=]?\s*(?P<value>\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)


@tool
def compare_documents(evidence_json: str, metric_keywords_csv: str = "") -> str:
    """
    Compare evidence chunks across multiple documents.
    Input:
      evidence_json: JSON array of evidence objects (doc, page, score, content).
      metric_keywords_csv: optional comma-separated keywords to filter metrics
                           (e.g. 'accuracy, F1, precision'). Leave blank for all.
    Returns a JSON object with per-document summaries, shared metrics, and
    the best-performing document per metric.
    """
    try:
        raw = json.loads(evidence_json)
        evidence = [Evidence(**item) for item in raw]
        keywords = (
            [k.strip() for k in metric_keywords_csv.split(",") if k.strip()]
            if metric_keywords_csv
            else None
        )
        result = DocumentComparison().compare(evidence, keywords)
        return json.dumps(result)
    except Exception as e:
        return f"Error: {e}"


# ── Raw class kept for direct use inside AnalystAgent ──
class DocumentComparison:
    def group_by_document(self, evidence_list: List[Evidence]) -> Dict[str, List[Evidence]]:
        groups: Dict[str, List[Evidence]] = defaultdict(list)
        for ev in evidence_list:
            # ev.doc is Zeina's field name (was ev.document in Alaa's original)
            groups[ev.doc].append(ev)
        return groups

    def extract_numeric_mentions(self, text: str) -> List[Dict[str, Any]]:
        found = []
        for m in _NUM_NEAR_LABEL.finditer(text):
            found.append({"label": m.group("label").strip(), "value": float(m.group("value"))})
        return found

    def compare(
        self,
        evidence_list: List[Evidence],
        metric_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        groups = self.group_by_document(evidence_list)
        per_document: Dict[str, Any] = {}
        metric_table: Dict[str, Dict[str, float]] = defaultdict(dict)

        for doc, chunks in groups.items():
            mentions = []
            excerpts = []
            for ev in chunks:
                mentions.extend(self.extract_numeric_mentions(ev.content))
                excerpts.append(ev.content[:280])
            if metric_keywords:
                mentions = [
                    m for m in mentions
                    if any(k.lower() in m["label"].lower() for k in metric_keywords)
                ]
            per_document[doc] = {
                "evidence_count": len(chunks),
                "numeric_mentions": mentions,
                "excerpts": excerpts,
            }
            for m in mentions:
                metric_table[m["label"]].setdefault(doc, m["value"])

        best_by_metric = {}
        for label, doc_values in metric_table.items():
            if doc_values:
                best_doc = max(doc_values, key=doc_values.get)
                best_by_metric[label] = {"document": best_doc, "value": doc_values[best_doc]}

        return {
            "documents": list(groups.keys()),
            "per_document": per_document,
            "shared_metrics": dict(metric_table),
            "best_by_metric": best_by_metric,
        }

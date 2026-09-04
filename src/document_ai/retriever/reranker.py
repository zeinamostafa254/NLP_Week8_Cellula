from typing import Any
import math
import re


class Reranker:
    """
    Dependency-light lexical reranker.

    It combines the initial retrieval score with exact query-term coverage.
    This provides a real second ranking stage without requiring another paid
    API call. It can later be replaced by a CrossEncoder or hosted reranker.
    """

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        query_terms = set(re.findall(r"\b[\w.-]+\b", query.lower()))
        if not query_terms:
            return results[:top_k]

        reranked = []
        for item in results:
            text_terms = set(
                re.findall(r"\b[\w.-]+\b", item.get("content", "").lower())
            )
            exact_coverage = len(query_terms & text_terms) / len(query_terms)

            base = float(item.get("score", 0.0))
            final_score = 0.7 * base + 0.3 * exact_coverage

            updated = dict(item)
            updated["initial_score"] = base
            updated["rerank_score"] = final_score
            updated["score"] = final_score
            reranked.append(updated)

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]

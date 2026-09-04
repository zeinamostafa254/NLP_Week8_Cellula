from typing import Any

from document_ai.config import RETRIEVAL_K, FINAL_K
from document_ai.schemas.evidence import Evidence, EvidenceBundle
from document_ai.retriever.query_rewriter import QueryRewriter
from document_ai.retriever.semantic_search import SemanticSearcher
from document_ai.retriever.keyword_search import KeywordSearcher
from document_ai.retriever.metadata_filter import filter_results
from document_ai.retriever.reranker import Reranker
from document_ai.retriever.context_selector import ContextSelector


import logging
logger = logging.getLogger(__name__)

class RetrieverAgent:
    """
    End-to-end Retriever Agent.

    Flow:
        user query
          -> query rewrite
          -> semantic + keyword search
          -> metadata filtering
          -> merge/deduplicate
          -> rerank
          -> context selection
          -> EvidenceBundle
    """

    def __init__(self):
        self.query_rewriter = QueryRewriter()
        self.semantic_search = SemanticSearcher()
        self.keyword_search = KeywordSearcher()
        self.reranker = Reranker()
        self.context_selector = ContextSelector()

    @staticmethod
    def _merge_results(
        semantic: list[dict[str, Any]],
        keyword: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged = {}
        for item in semantic + keyword:
            metadata = item.get("metadata", {})
            key = (
                metadata.get("doc"),
                metadata.get("page"),
                metadata.get("chunk_id"),
                item.get("content", "")[:80],
            )

            if key not in merged:
                merged[key] = item
            else:
                # Preserve the stronger score.
                if item["score"] > merged[key]["score"]:
                    merged[key] = item

        return list(merged.values())

    def retrieve(
        self,
        query: str,
        chat_history: str = "",
        filters: dict[str, Any] | None = None,
    ) -> EvidenceBundle:
        logger.info(f"    [Retriever] Original query: '{query}'")
        rewritten = self.query_rewriter.rewrite(query, chat_history)
        logger.info(f"    [Retriever] Rewritten query: '{rewritten}'")

        semantic_results = self.semantic_search.search(rewritten, k=RETRIEVAL_K)
        keyword_results = self.keyword_search.search(rewritten, k=RETRIEVAL_K)
        logger.info(f"    [Retriever] Found {len(semantic_results)} semantic hits, {len(keyword_results)} keyword hits")

        merged = self._merge_results(semantic_results, keyword_results)
        filtered = filter_results(merged, filters)
        reranked = self.reranker.rerank(rewritten, filtered, top_k=RETRIEVAL_K)
        selected = self.context_selector.select(reranked, k=FINAL_K)
        logger.info(f"    [Retriever] After merge, filter, and rerank -> returning {len(selected)} chunks")

        evidence = [
            Evidence(
                doc=str(item.get("metadata", {}).get("doc", "unknown")),
                page=item.get("metadata", {}).get("page"),
                score=float(item.get("score", 0.0)),
                content=item.get("content", ""),
                metadata=item.get("metadata", {}),
                retrieval_method=item.get("retrieval_method", "hybrid"),
            )
            for item in selected
        ]

        source_keys = {
            (e.doc, e.page)
            for e in evidence
        }

        return EvidenceBundle(
            query=query,
            rewritten_query=rewritten,
            evidence=evidence,
            filters=filters or {},
            source_count=len(source_keys),
        )

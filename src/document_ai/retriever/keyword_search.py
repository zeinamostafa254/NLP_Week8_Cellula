import re
from typing import Any
from langchain_chroma import Chroma

from document_ai.config import COLLECTION_NAME, VECTOR_DB_DIR


class KeywordSearcher:
    """
    Lightweight lexical search over the chunks stored in Chroma.

    This is intentionally simple and dependency-light. It is useful for exact
    terms such as model names, IDs, equations, or unique phrases.
    """

    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "what", "which",
        "who", "how", "why", "when", "where", "of", "to", "in", "on",
        "for", "and", "or", "with", "from", "by", "about", "does", "do",
    }

    def __init__(self):
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTOR_DB_DIR),
        )

    @staticmethod
    def _tokens(text: str) -> list[str]:
        tokens = re.findall(r"\b[\w.-]+\b", text.lower())
        return [t for t in tokens if t not in KeywordSearcher.STOPWORDS]

    def search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        # Chroma can return all stored text/metadata. This is appropriate for
        # a small-to-medium project collection; a real large-scale system
        # should replace this with BM25/Elasticsearch/OpenSearch.
        data = self.vector_store.get(include=["documents", "metadatas"])

        query_tokens = set(self._tokens(query))
        scored = []

        for content, metadata in zip(
            data.get("documents", []),
            data.get("metadatas", []),
        ):
            doc_tokens = set(self._tokens(content or ""))
            if not query_tokens:
                continue

            overlap = len(query_tokens & doc_tokens)
            if overlap == 0:
                continue

            # Jaccard-like lexical score.
            score = overlap / len(query_tokens | doc_tokens)
            scored.append(
                {
                    "content": content,
                    "metadata": metadata or {},
                    "score": float(score),
                    "retrieval_method": "keyword",
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

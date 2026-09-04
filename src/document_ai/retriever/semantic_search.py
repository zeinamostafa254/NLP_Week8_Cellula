from typing import Any
from langchain_chroma import Chroma
from document_ai.config import COLLECTION_NAME, VECTOR_DB_DIR, RETRIEVAL_K
from document_ai.ingestion.embedder import get_embedding_model


class SemanticSearcher:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=get_embedding_model(),
        )

    def search(self, query: str, k: int = RETRIEVAL_K) -> list[dict[str, Any]]:
        """
        Semantic similarity search.

        Chroma's score is a distance in the common configuration, where lower
        is better. We convert it to a simple bounded relevance score for the
        evidence contract.
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        output = []

        for doc, distance in results:
            distance = float(distance)
            score = 1.0 / (1.0 + max(distance, 0.0))

            output.append(
                {
                    "content": doc.page_content,
                    "metadata": dict(doc.metadata),
                    "score": score,
                    "retrieval_method": "semantic",
                }
            )

        return output

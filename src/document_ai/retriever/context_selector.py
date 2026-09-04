from typing import Any
from document_ai.config import FINAL_K


class ContextSelector:
    """
    Select a compact evidence set while encouraging document/page diversity.

    The selector first favors high scores, then avoids sending many near-duplicate
    chunks from exactly the same page when other useful sources exist.
    """

    def select(
        self,
        results: list[dict[str, Any]],
        k: int = FINAL_K,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen_keys: set[tuple] = set()

        # Results are already ranked.
        for item in results:
            metadata = item.get("metadata", {})
            key = (
                metadata.get("doc"),
                metadata.get("page"),
            )

            if key not in seen_keys:
                selected.append(item)
                seen_keys.add(key)

            if len(selected) >= k:
                return selected

        # If there were not enough distinct pages, fill remaining slots.
        selected_ids = {id(x) for x in selected}
        for item in results:
            if id(item) not in selected_ids:
                selected.append(item)
            if len(selected) >= k:
                break

        return selected

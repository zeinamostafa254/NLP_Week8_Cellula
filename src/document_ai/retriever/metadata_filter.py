from typing import Any


def filter_results(
    results: list[dict[str, Any]],
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Filter evidence by exact metadata values."""
    if not filters:
        return results

    filtered = []
    for item in results:
        metadata = item.get("metadata", {})
        matches = True

        for key, expected in filters.items():
            if expected is None:
                continue

            actual = metadata.get(key)

            if isinstance(expected, list):
                if actual not in expected:
                    matches = False
                    break
            elif str(actual).lower() != str(expected).lower():
                matches = False
                break

        if matches:
            filtered.append(item)

    return filtered

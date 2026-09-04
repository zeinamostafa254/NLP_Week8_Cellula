"""
analyst/data_analysis.py
-------------------------
Higher-level statistical analysis tool for the Analyst Agent.
Turns raw extracted numbers into rankings, distributions, and trend detection.

Original logic by Alaa (Member 2) — wrapped with @tool for LangChain.
"""

import json
import statistics
from typing import Dict, List, Tuple

from langchain_core.tools import tool


@tool
def rank_items(items_json: str, descending: bool = True) -> str:
    """
    Rank named items by their numeric values.
    Input: JSON string mapping names to numbers, e.g. '{"CNN": 92, "RNN": 89, "LSTM": 94}'.
    Returns a sorted list of [name, value] pairs.
    """
    try:
        items: Dict[str, float] = json.loads(items_json)
        ranked = sorted(items.items(), key=lambda kv: kv[1], reverse=descending)
        return json.dumps({"ranked": ranked, "best": ranked[0], "worst": ranked[-1]})
    except Exception as e:
        return f"Error: {e}"


@tool
def detect_trend(values_csv: str) -> str:
    """
    Detect whether a sequence of numbers is increasing, decreasing, or flat.
    Uses the sign of the linear regression slope.
    Input: comma-separated numbers in chronological order, e.g. '70, 75, 80, 85'.
    """
    try:
        values = [float(v.strip()) for v in values_csv.split(",")]
        n = len(values)
        if n < 2:
            return "insufficient_data"
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(values) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
        den = sum((x - x_mean) ** 2 for x in xs)
        if den == 0:
            return "flat"
        slope = num / den
        if abs(slope) < 1e-9:
            return "flat"
        return "increasing" if slope > 0 else "decreasing"
    except Exception as e:
        return f"Error: {e}"


@tool
def compute_distribution(values_csv: str, bucket_count: int = 5) -> str:
    """
    Bucket a list of numbers into a histogram-style distribution.
    Input: comma-separated numbers, e.g. '55, 65, 70, 85, 92, 95'.
    Returns a JSON object mapping bucket ranges to counts.
    """
    try:
        values = [float(v.strip()) for v in values_csv.split(",")]
        if not values:
            return "{}"
        lo, hi = min(values), max(values)
        if lo == hi:
            return json.dumps({str(lo): len(values)})
        width = (hi - lo) / bucket_count
        buckets: Dict[str, int] = {}
        for i in range(bucket_count):
            label = f"{lo + i * width:.2f}-{lo + (i + 1) * width:.2f}"
            buckets[label] = 0
        for v in values:
            idx = min(int((v - lo) / width), bucket_count - 1)
            label = list(buckets.keys())[idx]
            buckets[label] += 1
        return json.dumps(buckets)
    except Exception as e:
        return f"Error: {e}"


# ── Raw class kept for direct use inside AnalystAgent ──
class DataAnalysis:
    def summary_stats(self, values: List[float]) -> Dict:
        if not values:
            return {}
        stats = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        if len(values) > 1:
            stats["stdev"] = statistics.stdev(values)
        return stats

    def rank(self, items: Dict[str, float], descending: bool = True) -> List[Tuple]:
        return sorted(items.items(), key=lambda kv: kv[1], reverse=descending)

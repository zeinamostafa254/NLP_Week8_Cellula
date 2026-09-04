"""
analyst/calculator.py
----------------------
Deterministic arithmetic tool for the Analyst Agent.
Decorated with @tool so it can be registered via llm.bind_tools().

Original logic by Alaa (Member 2) — adapted for LangChain tool pattern.
"""

import ast
import operator
import statistics
from typing import List, Union

from langchain_core.tools import tool

Number = Union[int, float]

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


class CalculatorError(ValueError):
    pass


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](
            _eval_node(node.left), _eval_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise CalculatorError("Expression contains disallowed syntax")


@tool
def calculate_expression(expression: str) -> str:
    """
    Safely evaluate a numeric expression such as '(92+95+89)/3'.
    Only numbers and +-*/%() are allowed — no variable names or function calls.
    Returns a JSON-like string with operation, inputs, and result.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except (SyntaxError, TypeError, ZeroDivisionError, CalculatorError) as e:
        return f"Error: {e}"
    return str({"operation": "evaluate", "inputs": expression, "result": result})


@tool
def calculate_average(values_csv: str) -> str:
    """
    Compute the average of a comma-separated list of numbers.
    Example input: '92, 95, 89'
    Returns a string with the mean result.
    """
    try:
        values = [float(v.strip()) for v in values_csv.split(",")]
        result = statistics.mean(values)
        return str({"operation": "average", "inputs": values, "result": result})
    except Exception as e:
        return f"Error: {e}"


@tool
def calculate_percent_change(old_value: float, new_value: float) -> str:
    """
    Compute the percentage change from old_value to new_value.
    Formula: ((new - old) / old) * 100
    """
    if old_value == 0:
        return "Error: old_value cannot be zero"
    result = ((new_value - old_value) / old_value) * 100
    return str({"operation": "percent_change", "inputs": (old_value, new_value), "result": result})


@tool
def calculate_summary_stats(values_csv: str) -> str:
    """
    Compute summary statistics (count, mean, median, min, max, stdev)
    for a comma-separated list of numbers.
    Example input: '92, 95, 89, 78, 85'
    """
    try:
        values = [float(v.strip()) for v in values_csv.split(",")]
        if not values:
            return "Error: empty list"
        stats = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        if len(values) > 1:
            stats["stdev"] = statistics.stdev(values)
        return str(stats)
    except Exception as e:
        return f"Error: {e}"


# ── Raw class kept for direct instantiation in tests / agent._auto_calculations ──
class Calculator:
    """Direct-call wrapper (non-tool) used internally by AnalystAgent."""

    def average(self, values: List[Number]) -> dict:
        if not values:
            raise CalculatorError("average() needs at least one value")
        return {"tool": "calculator", "operation": "average",
                "inputs": values, "result": statistics.mean(values)}

    def evaluate(self, expression: str) -> dict:
        try:
            tree = ast.parse(expression, mode="eval")
            result = _eval_node(tree.body)
        except (SyntaxError, TypeError, ZeroDivisionError, CalculatorError) as e:
            raise CalculatorError(f"Could not evaluate '{expression}': {e}")
        return {"tool": "calculator", "operation": "evaluate",
                "inputs": expression, "result": result}

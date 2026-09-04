"""
analyst/agent.py
-----------------
The Analyst Agent — Member 2's core deliverable, refactored to use
LangChain's @tool + llm.bind_tools() pattern.

Flow:
  EvidenceBundle
    → EvidenceAssessor gate
    → if enough: run tools (calculator, table_extractor, doc_comparison, data_analysis)
                 via LLM tool-calling loop
    → if not enough: RetrieveMoreTool feedback loop → re-assess
    → return AnalystResult

Original logic by Alaa (Member 2):
  - Fixed all bare imports → full package imports
  - Fixed Evidence field names (document → doc)
  - Replaced custom Evidence dataclass with Zeina's Pydantic Evidence
  - Replaced custom AnalystResult dataclass with schemas/analysis.py Pydantic model
  - Integrated LangChain bind_tools pattern
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from document_ai.llm.model import get_llm
from document_ai.schemas.analysis import AnalystResult
from document_ai.schemas.evidence import Evidence, EvidenceBundle

from document_ai.analyst.calculator import (
    Calculator,
    calculate_average,
    calculate_expression,
    calculate_percent_change,
    calculate_summary_stats,
)
from document_ai.analyst.data_analysis import DataAnalysis, compute_distribution, detect_trend, rank_items
from document_ai.analyst.document_comparison import DocumentComparison, compare_documents
from document_ai.analyst.evidence_assessment import Assessment, AssessmentConfig, EvidenceAssessor
from document_ai.analyst.retrieve_more import MockRetrieverCallback, RetrieveMoreTool
from document_ai.analyst.table_extractor import TableExtractor, extract_tables_from_text

import logging
logger = logging.getLogger(__name__)

# All tools the Analyst LLM can call
ANALYST_TOOLS = [
    calculate_expression,
    calculate_average,
    calculate_percent_change,
    calculate_summary_stats,
    extract_tables_from_text,
    compare_documents,
    rank_items,
    detect_trend,
    compute_distribution,
]

ANALYST_SYSTEM_PROMPT = """You are an expert document analyst.

You have been given a set of retrieved evidence chunks from academic or technical documents.
Your job is to analyze this evidence to answer the user's question accurately.

You have access to the following tools:
- calculate_expression: evaluate numeric expressions
- calculate_average / calculate_summary_stats: compute statistics over lists of numbers
- calculate_percent_change: compute percentage changes
- extract_tables_from_text: parse tables from evidence text
- compare_documents: compare metrics across multiple documents
- rank_items: rank items by numeric value
- detect_trend: detect if a series is increasing/decreasing/flat
- compute_distribution: histogram a set of numbers

Use tools as needed, then provide a clear written analysis that directly answers the question.
Always cite evidence by document name and page when possible.
"""


class AnalystAgent:
    """
    Analyst Agent with LangChain bind_tools loop.

    Usage:
        agent = AnalystAgent(retriever_callback=my_retriever)
        result = agent.analyze(question, evidence_bundle)
    """

    def __init__(
        self,
        retriever_callback,
        assessment_config: Optional[AssessmentConfig] = None,
        max_iterations: int = 3,
        max_tool_turns: int = 10,
    ):
        self.assessor = EvidenceAssessor(assessment_config)
        self.retrieve_more = RetrieveMoreTool(retriever_callback)
        self.calculator = Calculator()
        self.table_extractor = TableExtractor()
        self.document_comparison = DocumentComparison()
        self.data_analysis = DataAnalysis()
        self.max_iterations = max_iterations
        self.max_tool_turns = max_tool_turns

        # LLM bound to all analyst tools
        self._llm = get_llm().bind_tools(ANALYST_TOOLS)

        # Tool name → callable map for executing tool calls
        self._tool_map = {t.name: t for t in ANALYST_TOOLS}

    # ------------------------------------------------------------------
    def analyze(self, question: str, evidence_bundle: EvidenceBundle) -> AnalystResult:
        evidence = list(evidence_bundle.evidence)
        iterations = 0

        while True:
            iterations += 1
            logger.info(f"    [Analyst] Assessment loop {iterations}")
            assessment: Assessment = self.assessor.assess(question, evidence)

            if assessment.enough_evidence:
                logger.info(f"    [Analyst] -> 'enough_evidence'. Running LLM tool loop...")
                return self._run_tool_loop(question, evidence, assessment, iterations)
            elif iterations >= self.max_iterations:
                logger.info(f"    [Analyst] -> Max iterations reached. Forcing LLM tool loop.")
                return self._run_tool_loop(question, evidence, assessment, iterations)

            logger.info(f"    [Analyst] -> 'need_more_evidence' (Missing: {assessment.missing_information})")
            # Feedback loop: ask Retriever for more evidence
            new_evidence = self.retrieve_more.request_more_evidence(
                assessment.suggested_follow_up_query or question
            )
            if not new_evidence:
                logger.warning("    [Analyst] Retriever found no additional evidence. Stopping loop.")
                # Retriever had nothing more — stop looping
                return self._run_tool_loop(question, evidence, assessment, iterations)

            logger.info(f"    [Analyst] Retrieved {len(new_evidence)} additional chunks.")
            evidence = _merge_evidence(evidence, new_evidence)

    # ------------------------------------------------------------------
    def _run_tool_loop(
        self,
        question: str,
        evidence: List[Evidence],
        assessment: Assessment,
        iterations: int,
    ) -> AnalystResult:
        """Run the LLM tool-calling loop and return an AnalystResult."""
        logger.info(f"    [Analyst] Calling LLM tools (turns_allowed={self.max_tool_turns})")
        if not assessment.enough_evidence:
            # Give the LLM what we have, but note the evidence was thin
            status = "need_more_evidence"
        else:
            status = "enough_evidence"

        # Build the evidence context string
        evidence_text = "\n\n".join(
            f"[{e.doc}, p.{e.page}, score={e.score:.2f}]\n{e.content}"
            for e in evidence
        )

        messages = [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Question: {question}\n\n"
                    f"Evidence:\n{evidence_text}\n\n"
                    "Analyze this evidence to answer the question. "
                    "Use tools as needed, then write your final analysis."
                )
            ),
        ]

        tables: list = []
        calculations: list = []
        comparisons: dict | None = None
        analysis_text = ""

        # Tool-calling loop
        for _ in range(self.max_tool_turns):
            response: AIMessage = self._llm.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                # LLM finished — extract final text
                analysis_text = response.content or ""
                break

            # Execute each tool call and feed results back
            for tc in response.tool_calls:
                fn = self._tool_map.get(tc["name"])
                result = fn.invoke(tc["args"]) if fn else "Tool not found."
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )
                # Collect structured outputs for AnalystResult
                if tc["name"] == "extract_tables_from_text":
                    import json
                    try:
                        tables.extend(json.loads(result))
                    except Exception:
                        pass
                elif tc["name"] in ("calculate_expression", "calculate_average",
                                    "calculate_percent_change", "calculate_summary_stats"):
                    import ast as _ast
                    try:
                        calculations.append(_ast.literal_eval(result))
                    except Exception:
                        pass
                elif tc["name"] == "compare_documents":
                    import json
                    try:
                        comparisons = json.loads(result)
                    except Exception:
                        pass

        # Auto-calculations for shared numeric metrics (Alaa's original logic)
        if comparisons and not calculations:
            calculations = self._auto_calculations(comparisons)

        return AnalystResult(
            status=status,
            analysis=analysis_text or None,
            calculations=calculations,
            tables=tables,
            comparisons=comparisons,
            evidence_used=evidence,
            missing_information=assessment.missing_information,
            follow_up_query=assessment.suggested_follow_up_query,
            iterations=iterations,
        )

    def _auto_calculations(self, comparison: dict) -> list:
        """For every metric mentioned in 2+ documents, compute average."""
        calcs = []
        for label, doc_values in comparison.get("shared_metrics", {}).items():
            values = list(doc_values.values())
            if len(values) >= 2:
                calcs.append({"metric": label, **self.calculator.average(values)})
        return calcs


# ── Helpers ────────────────────────────────────────────────────────────
def _merge_evidence(existing: List[Evidence], new: List[Evidence]) -> List[Evidence]:
    seen = {(e.doc, e.page, e.content) for e in existing}
    merged = list(existing)
    for e in new:
        key = (e.doc, e.page, e.content)
        if key not in seen:
            merged.append(e)
            seen.add(key)
    return merged

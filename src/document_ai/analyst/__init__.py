"""
analyst/ — Member 2's deliverable: the Analyst Agent and its tools.

Public interface (what the Orchestrator imports):

    from document_ai.analyst import AnalystAgent
    from document_ai.schemas import AnalystResult, EvidenceBundle

    agent = AnalystAgent(retriever_callback=retriever.retrieve)
    result: AnalystResult = agent.analyze(question, evidence_bundle)
"""

from document_ai.analyst.agent import AnalystAgent
from document_ai.analyst.calculator import Calculator, CalculatorError
from document_ai.analyst.data_analysis import DataAnalysis
from document_ai.analyst.document_comparison import DocumentComparison
from document_ai.analyst.evidence_assessment import Assessment, AssessmentConfig, EvidenceAssessor
from document_ai.analyst.retrieve_more import MockRetrieverCallback, RetrieveMoreTool
from document_ai.analyst.table_extractor import TableExtractor

__all__ = [
    "AnalystAgent",
    "Calculator",
    "CalculatorError",
    "DataAnalysis",
    "DocumentComparison",
    "EvidenceAssessor",
    "Assessment",
    "AssessmentConfig",
    "RetrieveMoreTool",
    "MockRetrieverCallback",
    "TableExtractor",
]

"""
orchestrator/orchestrator.py
------------------------------
Wires all three agents into the full pipeline.

Three orchestration strategies are implemented in this file:

  ✅ Strategy 1 (ACTIVE)   — LangGraph StateGraph
  💬 Strategy 2 (COMMENTED) — LangChain AgentExecutor
  💬 Strategy 3 (COMMENTED) — DeepAgent Loop
                              (think + retrieve + analyze + answer tools,
                               LLM freely reasons across the full flow)

To switch strategy: comment out the active `Orchestrator` alias at the
bottom and uncomment a different one.

Member 3 deliverable.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from document_ai.llm.model import get_llm
from document_ai.schemas.answer import FinalAnswer
from document_ai.schemas.analysis import AnalystResult
from document_ai.schemas.evidence import EvidenceBundle

if TYPE_CHECKING:
    from document_ai.retriever.agent import RetrieverAgent
    from document_ai.analyst.agent import AnalystAgent
    from document_ai.answer.agent import AnswerAgent


# ══════════════════════════════════════════════════════════════════════
# STRATEGY 1 (ACTIVE) — LangGraph StateGraph
# ══════════════════════════════════════════════════════════════════════
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    """Shared state flowing between LangGraph nodes."""
    question: str
    filters: Dict[str, Any]
    evidence_bundle: Optional[EvidenceBundle]
    analyst_result: Optional[AnalystResult]
    final_answer: Optional[FinalAnswer]
    loop_count: int
    max_loops: int


import logging
logger = logging.getLogger(__name__)

class OrchestratorGraph:
    """
    LangGraph orchestration — explicit typed graph with conditional edges.

    Graph topology:
        [START] → retrieve → analyze → (conditional) → answer → [END]
                                 ↑_____(need_more_evidence + loop budget)__|

    Nodes:
        retrieve : calls RetrieverAgent.retrieve()
        analyze  : calls AnalystAgent.analyze()
        answer   : calls AnswerAgent.answer()

    Conditional edge after "analyze":
        "enough_evidence"  or loop_count ≥ max_loops → "answer"
        "need_more_evidence" and loop_count < max_loops → "retrieve"
    """

    def __init__(
        self,
        retriever: RetrieverAgent,
        analyst: AnalystAgent,
        answer_agent: AnswerAgent,
    ):
        self._retriever = retriever
        self._analyst = analyst
        self._answer = answer_agent
        self.graph = self._build()

    # ── Node functions ──────────────────────────────────────────────
    def _retrieve_node(self, state: AgentState) -> AgentState:
        question = state["analyst_result"].follow_up_query or state["question"] \
            if state.get("analyst_result") else state["question"]
        logger.info(f"[Orchestrator] -> Running RetrieverNode for query: '{question}'")
        bundle = self._retriever.retrieve(
            query=question,
            filters=state.get("filters") or {},
        )
        return {**state, "evidence_bundle": bundle}

    def _analyze_node(self, state: AgentState) -> AgentState:
        logger.info(f"[Orchestrator] -> Running AnalyzeNode (Loop {state['loop_count'] + 1})")
        result = self._analyst.analyze(
            question=state["question"],
            evidence_bundle=state["evidence_bundle"],
        )
        return {
            **state,
            "analyst_result": result,
            "loop_count": state["loop_count"] + 1,
        }

    def _answer_node(self, state: AgentState) -> AgentState:
        logger.info("[Orchestrator] -> Running AnswerNode")
        final = self._answer.answer(
            question=state["question"],
            analyst_result=state["analyst_result"],
        )
        return {**state, "final_answer": final}

    # ── Conditional routing ─────────────────────────────────────────
    @staticmethod
    def _route_after_analysis(state: AgentState) -> str:
        result: AnalystResult = state["analyst_result"]
        if (
            result.status == "need_more_evidence"
            and state["loop_count"] < state["max_loops"]
        ):
            logger.info("[Orchestrator] Routing back to 'retrieve' (need_more_evidence)")
            return "retrieve"
        logger.info(f"[Orchestrator] Routing to 'answer' (status={result.status}, loop={state['loop_count']})")
        return "answer"

    # ── Graph assembly ──────────────────────────────────────────────
    def _build(self) -> Any:
        graph = StateGraph(AgentState)

        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("analyze",  self._analyze_node)
        graph.add_node("answer",   self._answer_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "analyze")
        graph.add_conditional_edges(
            "analyze",
            self._route_after_analysis,
            {"retrieve": "retrieve", "answer": "answer"},
        )
        graph.add_edge("answer", END)
        return graph.compile()


    def run(self, question: str, filters: Optional[Dict[str, Any]] = None, max_loops: int = 3) -> FinalAnswer:
        logger.info(f"--- Starting Orchestrator Pipeline (max_loops={max_loops}) ---")
        initial_state: AgentState = {
            "question": question,
            "filters": filters or {},
            "evidence_bundle": None,
            "analyst_result": None,
            "final_answer": None,
            "loop_count": 0,
            "max_loops": max_loops,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state["final_answer"]


# ══════════════════════════════════════════════════════════════════════
# STRATEGY 2 (COMMENTED) — LangChain AgentExecutor
# ══════════════════════════════════════════════════════════════════════

# class OrchestratorChain:
#     """
#     Classic LangChain approach: all 3 agents exposed as @tool functions
#     and bound to a single LLM via llm.bind_tools(). Wrapped in a
#     create_tool_calling_agent + AgentExecutor.
#
#     The LLM autonomously decides the call order and whether to loop.
#
#     Pros: minimal boilerplate, flexible, easy to extend.
#     Cons: less explicit control — the LLM may skip steps or loop
#           unpredictably without careful prompting.
#     """
#
#     SYSTEM_PROMPT = """You are an orchestrator for a document Q&A system.
#     You have three tools: retrieve_evidence, analyze_evidence, produce_answer.
#     To answer the user's question:
#       1. Call retrieve_evidence(question) to get relevant document chunks.
#       2. Call analyze_evidence(question, evidence_json) to analyze them.
#       3. If the analysis says need_more_evidence, call retrieve_evidence again
#          with a refined query, then analyze again.
#       4. When analysis says enough_evidence, call produce_answer to get the
#          final response.
#     """
#
#     def __init__(self, retriever: RetrieverAgent, analyst: AnalystAgent, answer_agent: AnswerAgent):
#         from langchain.agents import AgentExecutor, create_tool_calling_agent
#         from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#
#         retriever_ref = retriever
#         analyst_ref   = analyst
#         answer_ref    = answer_agent
#
#         @tool
#         def retrieve_evidence(question: str) -> str:
#             """Retrieve relevant evidence chunks for the question."""
#             bundle = retriever_ref.retrieve(question)
#             return bundle.model_dump_json()
#
#         @tool
#         def analyze_evidence(question: str, evidence_json: str) -> str:
#             """Analyze the retrieved evidence to answer the question."""
#             bundle = EvidenceBundle.model_validate_json(evidence_json)
#             result = analyst_ref.analyze(question, bundle)
#             return result.model_dump_json()
#
#         @tool
#         def produce_answer(question: str, analysis_json: str) -> str:
#             """Produce the final formatted answer with citations."""
#             result = AnalystResult.model_validate_json(analysis_json)
#             final = answer_ref.answer(question, result)
#             return final.model_dump_json()
#
#         tools = [retrieve_evidence, analyze_evidence, produce_answer]
#         llm   = get_llm().bind_tools(tools)
#
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", self.SYSTEM_PROMPT),
#             ("human",  "{input}"),
#             MessagesPlaceholder("agent_scratchpad"),
#         ])
#
#         agent = create_tool_calling_agent(llm, tools, prompt)
#         self.executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#         self._answer_ref = answer_ref
#
#     def run(self, question: str, filters=None, max_loops: int = 3) -> FinalAnswer:
#         output = self.executor.invoke({"input": question})
#         # The final tool call to produce_answer returns a FinalAnswer JSON
#         try:
#             return FinalAnswer.model_validate_json(output["output"])
#         except Exception:
#             # Fallback if executor returns raw text
#             return FinalAnswer(
#                 question=question,
#                 answer=output["output"],
#                 citations=[],
#                 sources="",
#                 confidence=0.0,
#                 metadata={},
#             )


# ══════════════════════════════════════════════════════════════════════
# STRATEGY 3 (COMMENTED) — DeepAgent Loop
# ══════════════════════════════════════════════════════════════════════

# class OrchestratorDeepAgent:
#     """
#     DeepAgent orchestration: the LLM is given 4 tools —
#       think, retrieve, analyze, answer —
#     and loops freely (up to MAX_TURNS) until it calls `answer`.
#
#     The `think` tool is a scratchpad: the LLM reasons out loud before
#     deciding which tool to call next. Its output is echoed back into
#     the message history so the LLM builds on its own chain-of-thought
#     across turns.
#
#     Tool calling sequence (example):
#       LLM: think("I need to find papers about CNN accuracy first")
#       LLM: retrieve("CNN accuracy comparison across papers")
#       LLM: think("I have 3 papers; I should compare them numerically")
#       LLM: analyze("compare CNN accuracy", evidence_json)
#       LLM: think("Evidence is sufficient, I can now produce the answer")
#       LLM: answer("compare CNN accuracy", analysis_json)  ← STOP
#
#     Pros: maximally flexible, rich chain-of-thought, can self-correct.
#     Cons: more LLM calls, harder to guarantee termination without MAX_TURNS.
#     """
#
#     MAX_TURNS = 12
#
#     SYSTEM_PROMPT = """You are a deep research agent for document Q&A.
# You have four tools:
#   - think(thought): reason step by step before acting. Use this freely.
#   - retrieve(question): retrieve relevant evidence chunks from the document store.
#   - analyze(question, evidence_json): analyze the evidence to answer the question.
#   - answer(question, analysis_json): produce the final cited answer. Call this last.
#
# Always think before each major action. Retrieve first, then analyze.
# If the analysis says 'need_more_evidence', think about a better query, then retrieve again.
# When you are confident the evidence is sufficient, call answer to finish.
# """
#
#     def __init__(self, retriever: RetrieverAgent, analyst: AnalystAgent, answer_agent: AnswerAgent):
#         retriever_ref = retriever
#         analyst_ref   = analyst
#         answer_ref    = answer_agent
#
#         @tool
#         def think(thought: str) -> str:
#             """Use this to reason step by step before deciding the next action.
#             Your thought is recorded and influences your future decisions."""
#             return thought  # echoed back into message history
#
#         @tool
#         def retrieve(question: str) -> str:
#             """Retrieve relevant evidence chunks for the question from the document store."""
#             bundle = retriever_ref.retrieve(question)
#             return bundle.model_dump_json()
#
#         @tool
#         def analyze(question: str, evidence_json: str) -> str:
#             """Analyze retrieved evidence to answer the question.
#             Returns analysis result including status (enough_evidence / need_more_evidence)."""
#             bundle = EvidenceBundle.model_validate_json(evidence_json)
#             result = analyst_ref.analyze(question, bundle)
#             return result.model_dump_json()
#
#         @tool
#         def answer(question: str, analysis_json: str) -> str:
#             """Produce the final formatted answer with citations. Call this when ready to finish."""
#             result = AnalystResult.model_validate_json(analysis_json)
#             final  = answer_ref.answer(question, result)
#             return final.model_dump_json()
#
#         self._tools    = [think, retrieve, analyze, answer]
#         self._llm      = get_llm().bind_tools(self._tools)
#         self._tool_map = {t.name: t for t in self._tools}
#         self._stop_tool = "answer"
#
#     def run(self, question: str, filters=None, max_loops: int = 3) -> FinalAnswer:
#         messages = [
#             SystemMessage(content=self.SYSTEM_PROMPT),
#             HumanMessage(content=question),
#         ]
#
#         for turn in range(self.MAX_TURNS):
#             response: AIMessage = self._llm.invoke(messages)
#             messages.append(response)
#
#             if not response.tool_calls:
#                 # LLM stopped — wrap raw text as FinalAnswer
#                 return FinalAnswer(
#                     question=question,
#                     answer=response.content or "No answer produced.",
#                     citations=[],
#                     sources="",
#                     confidence=0.0,
#                     metadata={"turns": turn + 1, "strategy": "deepagent"},
#                 )
#
#             for tc in response.tool_calls:
#                 fn     = self._tool_map[tc["name"]]
#                 result = fn.invoke(tc["args"])
#                 messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
#
#                 if tc["name"] == self._stop_tool:
#                     try:
#                         return FinalAnswer.model_validate_json(result)
#                     except Exception:
#                         return FinalAnswer(
#                             question=question,
#                             answer=result,
#                             citations=[],
#                             sources="",
#                             confidence=0.0,
#                             metadata={"turns": turn + 1, "strategy": "deepagent"},
#                         )
#
#         raise RuntimeError(
#             f"DeepAgent exhausted {self.MAX_TURNS} turns without producing a final answer."
#         )


# ══════════════════════════════════════════════════════════════════════
# Active alias — change this line to switch strategy
# ══════════════════════════════════════════════════════════════════════
Orchestrator = OrchestratorGraph          # ← Strategy 1 active
# Orchestrator = OrchestratorChain        # ← Strategy 2
# Orchestrator = OrchestratorDeepAgent    # ← Strategy 3

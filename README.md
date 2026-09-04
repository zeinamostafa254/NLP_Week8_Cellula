# Document AI Assistant

Multi-agent RAG system for document question-answering.
Built by a 3-member team as part of an NLP project.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                         │
│            (LangGraph StateGraph — active)              │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐     │
│   │ Retrieve │───▶│ Analyze  │───▶│    Answer    │     │
│   │  Agent   │◀───│  Agent   │    │    Agent     │     │
│   └──────────┘    └──────────┘    └──────────────┘     │
│   (loop back if need_more_evidence, max 3 times)        │
└─────────────────────────────────────────────────────────┘
      │
      ▼
  FinalAnswer (answer + citations [1][2] + bibliography)
```

### Agent Responsibilities

| Agent | Member | Tools |
|---|---|---|
| **Retriever Agent** | Zeina (Member 1) | query_rewriter, semantic_search, keyword_search, metadata_filter, reranker, context_selector |
| **Analyst Agent** | Alaa (Member 2) | `@tool` calculate_expression, calculate_average, extract_tables_from_text, compare_documents, rank_items, detect_trend, retrieve_more_evidence |
| **Answer Agent** | Member 3 | `@tool` format_citations, format_sources, format_response |
| **Orchestrator** | Member 3 | LangGraph StateGraph (+ 2 alternative strategies commented) |

### Orchestration Strategies

Three strategies are implemented in [`orchestrator.py`](src/document_ai/orchestrator/orchestrator.py). Only Strategy 1 is active — swap by changing the `Orchestrator` alias at the bottom of the file.

| | Strategy | Description |
|---|---|---|
| ✅ **Active** | **LangGraph StateGraph** | Explicit typed graph with conditional edges. The graph routes between retrieve/analyze/answer nodes based on `AnalystResult.status`. |
| 💬 Commented | **LangChain AgentExecutor** | All 3 agents as `@tool` functions bound to one LLM via `bind_tools()`. The LLM decides call order. |
| 💬 Commented | **DeepAgent Loop** | 4 tools: `think` (scratchpad), `retrieve`, `analyze`, `answer`. LLM reasons freely via chain-of-thought before each action. |

---

## Project Structure

```
NLP_Week7_Cellula/
├── app.py                              # Streamlit UI
├── main.py                             # CLI entry point
├── pyproject.toml
├── .env                                # API keys (not committed)
└── src/document_ai/
    ├── config.py                       # Shared configuration
    ├── llm/model.py                    # OpenRouter LLM client
    ├── schemas/
    │   ├── evidence.py                 # Evidence, EvidenceBundle
    │   ├── analysis.py                 # AnalystResult
    │   └── answer.py                   # Citation, FinalAnswer
    ├── ingestion/                      # load→parse→clean→chunk→embed→store
    ├── retriever/                      # RetrieverAgent + 6 tools
    ├── analyst/                        # AnalystAgent + 5 @tools + evidence_assessment
    ├── answer/                         # AnswerAgent + 3 @tools
    ├── orchestrator/                   # 3-strategy orchestrator
    └── api/main.py                     # FastAPI backend
```

---

## Setup

### 1. Install dependencies

```bash
# Install uv if not already installed
pip install uv

# Install project
uv sync
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=20
FINAL_K=8
```

---

## Running

### Option A — Full stack (API + UI)

```bash
# Terminal 1: start the FastAPI backend
uv run uvicorn document_ai.api.main:app --reload

# Terminal 2: start the Streamlit UI
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**UI Tabs:**
- 📄 **Upload Documents** — drag and drop PDFs/DOCX/TXT, click Ingest
- 💬 **Ask Questions** — chat interface with citation cards and bibliography

### Option B — CLI

```bash
# Ingest documents
uv run python main.py ingest path/to/paper1.pdf path/to/paper2.pdf

# Ask a question
uv run python main.py query "What is the average CNN accuracy across all papers?"
uv run python main.py query "Compare the F1 scores in the three papers" --max-loops 5
```

### Option C — API only (Swagger UI)

```bash
uv run uvicorn document_ai.api.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/ingest` | Upload + ingest documents |
| `GET` | `/documents` | List ingested document names |
| `POST` | `/query` | Full pipeline Q&A → FinalAnswer |

---

## Running Tests

```bash
uv run pytest tests/ -v
```

Tests do **not** require an API key or a running vector DB — all LLM and retriever calls are monkeypatched.

---

## Interface Contracts

### EvidenceBundle (Member 1 → Member 2)
```json
{
  "query": "original question",
  "rewritten_query": "expanded query",
  "evidence": [
    {"doc": "paper.pdf", "page": 3, "score": 0.92, "content": "..."}
  ],
  "filters": {},
  "source_count": 2
}
```

### AnalystResult (Member 2 → Member 3)
```json
{
  "status": "enough_evidence",
  "analysis": "Written analysis text...",
  "calculations": [{"operation": "average", "inputs": [92, 95], "result": 93.5}],
  "tables": [],
  "comparisons": {},
  "evidence_used": [...],
  "iterations": 1
}
```

### FinalAnswer (Member 3 → User/API)
```json
{
  "question": "What is the average CNN accuracy?",
  "answer": "The average CNN accuracy is 92% [1][2]...\n\n**Sources:**\n[1] PaperA.pdf...",
  "citations": [{"ref_id": 1, "doc": "PaperA.pdf", "page": 3, "score": 0.92, "snippet": "..."}],
  "sources": "[1] PaperA.pdf, page 3 (relevance: 0.920)",
  "confidence": 0.895,
  "metadata": {"analyst_status": "enough_evidence", "iterations": 1, "evidence_count": 5}
}
```

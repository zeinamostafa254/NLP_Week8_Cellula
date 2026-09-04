# Member 1 Branch Handoff — Data & Retriever Agent

## Purpose

This document explains the work owned by **Member 1** and gives Members 2 and 3 the information they need to continue the project without breaking the interfaces.

Member 1 owns the complete **Data & Retriever pipeline**:

```text
Documents
   ↓
Loading → Parsing → Cleaning → Chunking → Embeddings → Vector DB
   ↓
User Question → Query Rewriting → Semantic + Keyword Search
   ↓
Metadata Filtering → Reranking → Context Selection
   ↓
EvidenceBundle
   ↓
Member 2 — Analyst Agent
```

---

## 1. Team Architecture

### Member 1 — Data & Retriever Agent

Owns:

- Document ingestion
- Loading / parsing / cleaning
- Chunking
- Embeddings
- Vector DB
- Retriever Agent
- Query Rewriter
- Semantic Search
- Keyword Search
- Metadata Filter
- Reranker
- Context Selector

**Output:** a standardized `EvidenceBundle`.

### Member 2 — Analyst Agent

Consumes the `EvidenceBundle` and owns:

- Calculator
- Table Extractor
- Document Comparison
- Data Analysis
- More-Evidence Retrieval
- Evidence Assessment

If evidence is insufficient, Member 2 requests another retrieval pass.

### Member 3 — Answer Agent & Orchestrator

Consumes Analyst output and owns:

- Citation Formatter
- Source Formatter
- Response Formatter
- Orchestrator
- End-to-end response assembly
- Integration testing

---

# 2. Member 1 Files

## `src/document_ai/config.py`

Central configuration.

Should contain configurable values such as:

- document paths
- vector DB path
- embedding model
- OpenRouter settings
- LLM model
- retrieval parameters
- chunk size / overlap

Do not hard-code the API key.

Expected LLM:

```text
google/gemma-4-26b-a4b-it:free
```

Expected OpenRouter base URL:

```text
https://openrouter.ai/api/v1
```

---

# 3. Schemas

## `src/document_ai/schemas/evidence.py`

This is the **main interface between Member 1 and Member 2**.

The Retriever should return an `EvidenceBundle` containing evidence items with at least:

```text
document name
page
score
content
```

Recommended additional fields:

```text
metadata
retrieval_method
```

Conceptually:

```json
{
  "doc": "paper.pdf",
  "page": 3,
  "score": 0.87,
  "content": "Relevant document text...",
  "metadata": {},
  "retrieval_method": "semantic"
}
```

Keep this schema stable after Members 1 and 2 agree on it.

Member 2 should not need to know how Chroma, BM25, embeddings, or reranking work.

---

# 4. Ingestion

## `src/document_ai/ingestion/loader.py`

Loads documents from the input directory / uploaded files.

Preserve useful source information:

- document name
- source path
- page number where available

---

## `src/document_ai/ingestion/parser.py`

Converts loaded documents into a consistent internal representation.

Extract text while preserving document/page metadata.

---

## `src/document_ai/ingestion/cleaner.py`

Cleans extracted text before chunking.

Typical responsibilities:

- normalize whitespace
- normalize line breaks
- remove extraction artifacts
- remove empty content
- preserve meaningful text

Expected function:

```python
clean_text(...)
```

Example:

```python
from document_ai.ingestion.cleaner import clean_text
```

---

## `src/document_ai/ingestion/chunker.py`

Splits cleaned documents into retrieval-friendly chunks.

It should:

- use configurable chunk size
- use chunk overlap
- preserve document metadata
- preserve page numbers
- avoid losing source information

Expected function:

```python
chunk_documents(...)
```

Example:

```python
from document_ai.ingestion.chunker import chunk_documents
```

Every chunk should retain enough metadata for later citation.

---

## `src/document_ai/ingestion/embedder.py`

Converts chunks into vector embeddings.

Important distinction:

```text
LLM:
google/gemma-4-26b-a4b-it:free

Embedding model:
BAAI/bge-small-en-v1.5
```

**Do not use Gemma as the embedding model.**

---

## `src/document_ai/ingestion/pipeline.py`

Coordinates:

```text
load
 ↓
parse
 ↓
clean
 ↓
chunk
 ↓
embed
 ↓
store in vector DB
```

This should be the main reusable ingestion entry point.

---

# 5. Retriever

## `src/document_ai/retriever/query_rewriter.py`

Uses the OpenRouter LLM to make the user's question retrieval-friendly.

The LLM should **not answer the question** here.

Recommended output:

```json
{
  "rewritten_query": "...",
  "keywords": ["...", "..."],
  "metadata_filters": {}
}
```

Example:

```text
Question:
"What are the disadvantages of this method?"

Rewritten:
"What are the disadvantages and limitations of the described method?"

Keywords:
["disadvantages", "limitations", "method"]

Metadata filters:
{}
```

The implementation should have a fallback to the original query if the LLM response is malformed.

---

## `src/document_ai/retriever/semantic_search.py`

Performs vector similarity search against the vector database.

Flow:

```text
query
 ↓
embedding
 ↓
Chroma
 ↓
candidate documents
```

Preserve:

- content
- document
- page
- score
- metadata

---

## `src/document_ai/retriever/keyword_search.py`

Performs lexical / keyword retrieval.

Conceptually:

```text
Semantic Search → meaning-based matches
Keyword Search  → exact / lexical matches
```

If BM25 is used, keep the matching itself deterministic.

The Query Rewriter may provide keywords, but the keyword search should not require an LLM.

---

## `src/document_ai/retriever/metadata_filter.py`

Filters candidate results using metadata.

Examples:

```text
document = "report.pdf"
page = 10
section = "Results"
```

Expected function:

```python
filter_results(...)
```

Example:

```python
from document_ai.retriever.metadata_filter import filter_results
```

If an LLM suggests a filter, Python code must validate and apply it.

---

## `src/document_ai/retriever/reranker.py`

Re-ranks candidate documents with a stronger relevance model.

Recommended architecture:

```text
Semantic / Keyword Search
          ↓
     candidates
          ↓
     CrossEncoder
          ↓
   ranked candidates
```

A possible model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Preserve the original document metadata.

---

## `src/document_ai/retriever/context_selector.py`

Selects the final evidence passed to Member 2.

Responsibilities:

- remove duplicate chunks
- remove empty content
- limit the number of chunks
- keep the strongest evidence
- preserve metadata and scores

The current project interface expects a `ContextSelector` class:

```python
from document_ai.retriever.context_selector import ContextSelector
```

If this is changed to a function, notify Members 2 and 3 and update the interface consistently.

---

## `src/document_ai/retriever/agent.py`

This is the main controller for the Retriever.

Expected flow:

```text
Query Rewriter
      ↓
Semantic Search + Keyword Search
      ↓
Merge / Deduplicate
      ↓
Metadata Filter
      ↓
Reranker
      ↓
Context Selector
      ↓
EvidenceBundle
```

The rest of the application should call the Retriever Agent rather than directly accessing Chroma or individual retrieval tools.

Conceptually:

```python
evidence = retriever.retrieve(question)
```

---

# 6. Interface With Member 2

### Member 1 receives

```text
user question
```

Potentially:

```text
conversation context
metadata constraints
```

### Member 1 returns

```text
EvidenceBundle
```

Example evidence item:

```json
{
  "doc": "document.pdf",
  "page": 5,
  "score": 0.91,
  "content": "Relevant passage...",
  "metadata": {},
  "retrieval_method": "semantic"
}
```

Member 2 should be able to assess and analyze this without knowing the retrieval internals.

---

# 7. More-Evidence Loop

The intended loop is:

```text
Evidence
   ↓
Analyst
   ↓
Evidence sufficient?
   ├── YES → Analysis → Answer Agent
   │
   └── NO → More Retrieval Request
                     ↓
                Retriever
                     ↓
              New EvidenceBundle
```

The Retriever therefore needs to be reusable and callable multiple times.

---

# 8. OpenRouter / Gemma

Use:

```text
Model:
google/gemma-4-26b-a4b-it:free

Base URL:
https://openrouter.ai/api/v1
```

Keep the key in `.env`:

```env
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Never commit `.env`.

Gemma is for LLM tasks such as query rewriting.

It should not be responsible for:

- embeddings
- Chroma storage
- deterministic keyword matching
- metadata filtering
- CrossEncoder reranking

---

# 9. Dependencies

Important Member 1 dependencies include:

```text
openai
python-dotenv
pydantic
langchain
langchain-core
langchain-community
langchain-text-splitters
langchain-chroma
chromadb
langchain-huggingface
sentence-transformers
pypdf
docx2txt
rank-bm25
pytest
```

The team's requirements file should remain the final source of truth.

---

# 10. Testing

Test Member 1 in this order:

```text
1. Cleaner
2. Chunker
3. Embedder
4. Chroma / Vector DB
5. Query Rewriter
6. Semantic Search
7. Keyword Search
8. Metadata Filter
9. Reranker
10. Context Selector
11. Retriever Agent
12. Full ingestion → retrieval flow
```

Relevant tests:

```text
tests/test_ingestion.py
tests/test_retriever.py
```

---

# 11. What Members 2 and 3 Should NOT Depend On

Do not make them directly depend on:

```text
Chroma internals
embedding implementation
BM25 implementation
CrossEncoder implementation
loaders
chunking logic
```

They should depend primarily on:

```text
RetrieverAgent
EvidenceBundle
```

This lets Member 1 change the retrieval implementation without breaking the rest of the system.

---

# 12. What Remains To Be Verified Before Member 1 Is Finished

## Ingestion

- [ ] Required document formats work
- [ ] Page/document metadata is preserved
- [ ] Cleaning works
- [ ] Chunking works
- [ ] Embeddings are generated
- [ ] Chroma storage works
- [ ] Full ingestion pipeline works

## Retriever

- [ ] Gemma Query Rewriter works through OpenRouter
- [ ] Semantic Search works
- [ ] Keyword Search works
- [ ] Metadata Filter works
- [ ] Reranker works
- [ ] Context Selector works
- [ ] Duplicate results are removed
- [ ] Scores are preserved
- [ ] Document/page metadata is preserved
- [ ] Retriever Agent returns the agreed EvidenceBundle

## Integration

- [ ] Member 2 can consume EvidenceBundle
- [ ] Retriever can be called again for more evidence
- [ ] Member 1 tests pass
- [ ] No API key is committed
- [ ] Vector DB is not unnecessarily committed
- [ ] Setup instructions are clear

---

# 13. Git Handoff

Before pushing:

```powershell
git status
git add .
git commit -m "Implement document ingestion and retriever pipeline"
git fetch origin
git pull --rebase origin main
git push -u origin YOUR_BRANCH_NAME
```

Then create a PR:

```text
base: main
compare: YOUR_BRANCH_NAME
```

The PR should explain:

- what was implemented
- how to run it
- the EvidenceBundle schema
- the LLM/model used
- known limitations
- remaining work

---

# 14. Final Architecture

```text
                         USER
                           │
                           ▼
                    ORCHESTRATOR
                           │
                           ▼
                  ┌─────────────────┐
                  │ RETRIEVER AGENT │
                  └────────┬────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        Query          Semantic       Keyword
       Rewriter         Search         Search
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    Metadata Filter
                           │
                           ▼
                       Reranker
                           │
                           ▼
                    Context Selector
                           │
                           ▼
                    EvidenceBundle
                           │
                           ▼
                   ┌──────────────┐
                   │ ANALYST AGENT│
                   └──────┬───────┘
                          │
                    Evidence OK?
                     /          \
                   YES           NO
                    │             │
                    ▼             ▼
                Analysis    More Retrieval
                    │             │
                    │             └──────→ Retriever
                    ▼
                 ANSWER AGENT
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
      Citations   Sources   Response
          │         │         │
          └─────────┼─────────┘
                    ▼
                   USER
```

---

# 15. Most Important Handoff Rule

**Member 1 owns how evidence is found.**

**Member 2 owns what the evidence means and whether it is sufficient.**

**Member 3 owns how the final answer is assembled and how the agents are orchestrated.**

The most important shared contract is:

```text
Retriever Agent
      ↓
EvidenceBundle
      ↓
Analyst Agent
```

Keep this interface stable even if the internal retrieval implementation changes.

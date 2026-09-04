$directories = @(
    "data",
    "data/documents",
    "data/documents/raw",
    "data/documents/processed",
    "data/vector_db",

    "src",
    "src/document_ai",
    "src/document_ai/schemas",

    "src/document_ai/ingestion",
    "src/document_ai/retriever",

    "src/document_ai/analyst",
    "src/document_ai/answer",
    "src/document_ai/orchestrator",
    "src/document_ai/llm",
    "src/document_ai/api",

    "tests"
)

$files = @(
    "src/document_ai/__init__.py",
    "src/document_ai/config.py",

    "src/document_ai/schemas/__init__.py",
    "src/document_ai/schemas/evidence.py",
    "src/document_ai/schemas/analysis.py",
    "src/document_ai/schemas/answer.py",

    "src/document_ai/ingestion/__init__.py",
    "src/document_ai/ingestion/loader.py",
    "src/document_ai/ingestion/parser.py",
    "src/document_ai/ingestion/cleaner.py",
    "src/document_ai/ingestion/chunker.py",
    "src/document_ai/ingestion/embedder.py",
    "src/document_ai/ingestion/pipeline.py",

    "src/document_ai/retriever/__init__.py",
    "src/document_ai/retriever/agent.py",
    "src/document_ai/retriever/query_rewriter.py",
    "src/document_ai/retriever/semantic_search.py",
    "src/document_ai/retriever/keyword_search.py",
    "src/document_ai/retriever/metadata_filter.py",
    "src/document_ai/retriever/reranker.py",
    "src/document_ai/retriever/context_selector.py",

    "src/document_ai/analyst/__init__.py",
    "src/document_ai/analyst/agent.py",
    "src/document_ai/analyst/calculator.py",
    "src/document_ai/analyst/table_extractor.py",
    "src/document_ai/analyst/document_comparison.py",
    "src/document_ai/analyst/data_analysis.py",
    "src/document_ai/analyst/evidence_assessment.py",
    "src/document_ai/analyst/retrieve_more.py",

    "src/document_ai/answer/__init__.py",
    "src/document_ai/answer/agent.py",
    "src/document_ai/answer/citation_formatter.py",
    "src/document_ai/answer/source_formatter.py",
    "src/document_ai/answer/response_formatter.py",

    "src/document_ai/orchestrator/__init__.py",
    "src/document_ai/orchestrator/orchestrator.py",

    "src/document_ai/llm/__init__.py",
    "src/document_ai/llm/model.py",

    "src/document_ai/api/__init__.py",
    "src/document_ai/api/main.py",

    "tests/test_ingestion.py",
    "tests/test_retriever.py",
    "tests/test_analyst.py",
    "tests/test_answer.py",
    "tests/test_orchestrator.py"
)

foreach ($directory in $directories) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

foreach ($file in $files) {
    New-Item -ItemType File -Path $file -Force | Out-Null
}

Write-Host "Project structure created successfully!"
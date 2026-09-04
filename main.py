"""
main.py — CLI entry point for the Document AI Assistant.

Usage:
    # Ingest documents
    uv run python main.py ingest path/to/doc1.pdf path/to/doc2.pdf

    # Ask a question
    uv run python main.py query "What is the average CNN accuracy across the papers?"

    # Start the API server
    uv run uvicorn document_ai.api.main:app --reload

    # Start the Streamlit UI (separate terminal)
    uv run streamlit run app.py
"""

import argparse
import sys
from document_ai.logger import setup_logging
from document_ai.monitoring import setup_langsmith

def cmd_ingest(paths: list[str]):
    from document_ai.ingestion.pipeline import ingest_files
    result = ingest_files(paths)
    print(f"✅ Ingested {result['chunks']} chunks from {result['documents']} document(s)")
    print(f"   Vector DB: {result['vector_db']}")
    print(f"   Collection: {result['collection']}")


def cmd_query(question: str, max_loops: int = 3):
    from document_ai.retriever.agent import RetrieverAgent
    from document_ai.analyst.agent import AnalystAgent
    from document_ai.analyst.retrieve_more import set_retriever_callback
    from document_ai.answer.agent import AnswerAgent
    from document_ai.orchestrator.orchestrator import Orchestrator

    print(f"\n--- Question: {question}\n")
    retriever = RetrieverAgent()
    set_retriever_callback(retriever.retrieve)
    analyst = AnalystAgent(retriever_callback=retriever.retrieve)
    answer_agent = AnswerAgent()
    orch = Orchestrator(retriever, analyst, answer_agent)

    result = orch.run(question, max_loops=max_loops)

    print("=" * 60)
    print(result.answer)
    print("=" * 60)
    if result.sources:
        print("\n--- Sources:")
        print(result.sources)
    print(f"\n--- Confidence: {result.confidence:.0%} | "
          f"Evidence: {result.metadata.get('evidence_count','?')} chunks | "
          f"Loops: {result.metadata.get('iterations','?')}")


def main():
    setup_logging()
    setup_langsmith()
    parser = argparse.ArgumentParser(description="Document AI Assistant CLI")
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents into the vector store")
    p_ingest.add_argument("paths", nargs="+", help="Paths to PDF/DOCX/TXT files")

    # query
    p_query = sub.add_parser("query", help="Ask a question against ingested documents")
    p_query.add_argument("question", help="The question to answer")
    p_query.add_argument("--max-loops", type=int, default=3,
                         help="Maximum retrieval feedback loops (default: 3)")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args.paths)
    elif args.command == "query":
        cmd_query(args.question, max_loops=args.max_loops)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

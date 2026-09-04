from src.document_ai.ingestion.cleaner import clean_text
from src.document_ai.ingestion.chunker import chunk_documents
from langchain_core.documents import Document


def test_clean_text():
    assert clean_text(" hello   world \n\n\n test ") == "hello world \n\n test"


def test_chunking():
    docs = [Document(page_content="A " * 100, metadata={"doc": "x.pdf", "page": 1})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1
    assert all("doc" in c.metadata for c in chunks)

from langchain_core.documents import Document


def normalize_metadata(doc: Document) -> Document:
    """Normalize page/source metadata so every downstream component sees the same keys."""
    metadata = dict(doc.metadata)

    # PyPDFLoader normally gives page as 0-based.
    if "page" in metadata and metadata["page"] is not None:
        try:
            metadata["page"] = int(metadata["page"]) + 1
        except (TypeError, ValueError):
            metadata["page"] = None

    metadata.setdefault("doc", metadata.get("source", "unknown"))
    metadata["doc"] = str(metadata["doc"]).split("/")[-1].split("\\")[-1]

    # Useful generic metadata fields; absent values stay None.
    metadata.setdefault("chapter", None)
    metadata.setdefault("section", None)
    metadata.setdefault("author", None)
    metadata.setdefault("date", None)
    metadata.setdefault("document_type", None)

    return Document(page_content=doc.page_content, metadata=metadata)


def parse_documents(documents: list[Document]) -> list[Document]:
    return [normalize_metadata(doc) for doc in documents]
from langchain_core.documents import Document


def normalize_metadata(doc: Document) -> Document:
    """Normalize page/source metadata so every downstream component sees the same keys."""
    metadata = dict(doc.metadata)

    # PyPDFLoader normally gives page as 0-based.
    if "page" in metadata and metadata["page"] is not None:
        try:
            metadata["page"] = int(metadata["page"]) + 1
        except (TypeError, ValueError):
            metadata["page"] = None

    metadata.setdefault("doc", metadata.get("source", "unknown"))
    metadata["doc"] = str(metadata["doc"]).split("/")[-1].split("\\")[-1]

    # Useful generic metadata fields; absent values stay None.
    metadata.setdefault("chapter", None)
    metadata.setdefault("section", None)
    metadata.setdefault("author", None)
    metadata.setdefault("date", None)
    metadata.setdefault("document_type", None)

    return Document(page_content=doc.page_content, metadata=metadata)


def parse_documents(documents: list[Document]) -> list[Document]:
    return [normalize_metadata(doc) for doc in documents]

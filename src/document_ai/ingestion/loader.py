from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

from document_ai.ingestion.ocr_loader import IMAGE_EXTENSIONS, load_image

TEXT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS


def load_file(path: str | Path) -> List[Document]:
    """Load a PDF, DOCX, TXT, Markdown, or image (via OCR) file into LangChain Documents."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix in IMAGE_EXTENSIONS:
        # OCR path — load_image() already returns fully-formed Documents
        # with doc/source metadata set, so return early.
        return load_image(path)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    documents = loader.load()

    # Always preserve the original filename.
    for doc in documents:
        doc.metadata["doc"] = path.name
        doc.metadata["source"] = str(path)

    return documents


def load_directory(directory: str | Path) -> List[Document]:
    """Load every supported document from a directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    documents: List[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(load_file(path))

    return documents

import re
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Lightweight cleaning that removes layout noise without destroying meaning."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_documents(documents: list[Document]) -> list[Document]:
    cleaned: list[Document] = []

    for doc in documents:
        content = clean_text(doc.page_content)
        if not content:
            continue

        cleaned.append(
            Document(
                page_content=content,
                metadata=dict(doc.metadata),
            )
        )

    return cleaned

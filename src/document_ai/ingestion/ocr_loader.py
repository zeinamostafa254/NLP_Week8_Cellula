"""
ingestion/ocr_loader.py
------------------------
Loads image files (PNG/JPG/JPEG/BMP/TIFF/WEBP) by running OCR (pytesseract)
to extract text, then wraps the result in a LangChain Document so it can
flow through the exact same parse -> clean -> chunk -> embed pipeline as
PDFs/DOCX/TXT.

Design notes:
- Imports of pytesseract/Pillow are done lazily inside functions, not at
  module import time. This means the rest of the app (and `import
  document_ai...` anywhere else) never crashes just because Tesseract
  isn't installed on a given machine — the error only surfaces when
  someone actually tries to OCR an image.
- Errors are raised as a single, clear OCRNotAvailableError /
  ValueError so callers (API, CLI, Streamlit) can catch them and show a
  friendly message instead of a stack trace.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from document_ai.config import OCR_LANGUAGE, TESSERACT_CMD

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class OCRNotAvailableError(RuntimeError):
    """Raised when pytesseract / the Tesseract binary isn't installed."""


def _get_ocr_dependencies():
    """Lazily import Pillow + pytesseract so a missing install never crashes app startup."""
    try:
        from PIL import Image
    except ImportError as e:
        raise OCRNotAvailableError(
            "Pillow is not installed. Run `uv add pillow` (or `pip install pillow`)."
        ) from e

    try:
        import pytesseract
    except ImportError as e:
        raise OCRNotAvailableError(
            "pytesseract is not installed. Run `uv add pytesseract` "
            "(or `pip install pytesseract`)."
        ) from e

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    return Image, pytesseract


def load_image(path: str | Path, lang: str | None = None) -> List[Document]:
    """
    Run OCR on a single image file and return it as a one-element list of
    LangChain Documents so it matches the return shape of load_file() for
    PDFs/DOCX/TXT.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image type: {path.suffix}. Supported: {sorted(IMAGE_EXTENSIONS)}"
        )

    Image, pytesseract = _get_ocr_dependencies()

    try:
        image = Image.open(path)
        # Convert to RGB so palette / RGBA / grayscale images all OCR cleanly.
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        text = pytesseract.image_to_string(image, lang=lang or OCR_LANGUAGE)
    except pytesseract.TesseractNotFoundError as e:
        raise OCRNotAvailableError(
            "The Tesseract OCR engine isn't installed on this machine.\n"
            "  - macOS:   brew install tesseract\n"
            "  - Ubuntu:  sudo apt-get install tesseract-ocr\n"
            "  - Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "After installing, set TESSERACT_CMD in .env if it's not on your PATH."
        ) from e
    except Exception as e:
        # Corrupt/unreadable image, unsupported format variant, etc. Don't let
        # one bad file take down a whole batch ingestion — surface a clean error.
        raise ValueError(f"Could not OCR image '{path.name}': {e}") from e

    text = (text or "").strip()

    document = Document(
        page_content=text,
        metadata={
            "doc": path.name,
            "source": str(path),
            "page": 1,
            "document_type": "image_ocr",
        },
    )
    return [document]

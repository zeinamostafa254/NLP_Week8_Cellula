import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "documents" / "raw"
PROCESSED_DOCS_DIR = DATA_DIR / "documents" / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-4o-mini",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "20"))
FINAL_K = int(os.getenv("FINAL_K", "8"))

# Optional collection name so the same Chroma directory can later be shared
# with the other members without changing the interface.
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_chunks")

# --- OCR (image upload) ---
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")  # only needed if tesseract isn't on PATH

# --- LangSmith monitoring/tracing ---
# All optional: if LANGCHAIN_API_KEY is unset, tracing simply stays off and
# nothing else in the app behaves differently or crashes.
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "document-ai-assistant")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# --- Voice query feature (faster-whisper) ---
# Model size options (smallest → largest): tiny, base, small, medium, large-v3
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
# "cpu" or "cuda"
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
# "int8"/"int8_float16" for CPU, "float16" for GPU — see faster-whisper docs
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
# Force a language code (e.g. "en", "ar") or leave empty to auto-detect
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "") or None

RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

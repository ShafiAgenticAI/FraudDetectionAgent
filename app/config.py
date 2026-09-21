import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENT_DIRECTORY = Path(
    os.getenv("DOCUMENT_DIRECTORY", str(BASE_DIR / "data" / "documents"))
)
CHROMA_PERSIST_DIRECTORY = Path(
    os.getenv("CHROMA_PERSIST_DIRECTORY", str(BASE_DIR / "data" / "chroma"))
)

# Which LLM provider to use for chat + embeddings: "openai" or "gemini".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"
)

TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

DOCUMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

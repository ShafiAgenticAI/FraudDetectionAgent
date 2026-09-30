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

# Which LLM provider to use for chat + embeddings: "openai", "gemini",
# or "bedrock".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
)

# AWS Bedrock -- only needed when LLM_PROVIDER=bedrock. Credentials are
# read from these vars if set; otherwise boto3's normal default chain
# applies (these are boto3's standard env var names, so if you already
# export them, or load them via .env, boto3 picks them up automatically).
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN", "")  # only for temporary creds
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Model IDs must be enabled for your account under Bedrock console ->
# Model access before first use, even on a free-tier/credit account.
# Titan models are usually enabled by default; Anthropic/Meta/Cohere
# models typically require a one-time access request (instant approval
# for most).
BEDROCK_CHAT_MODEL = os.getenv(
    "BEDROCK_CHAT_MODEL", "anthropic.claude-3-5-haiku-20241022-v1:0"
)
BEDROCK_EMBEDDING_MODEL = os.getenv(
    "BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"
)

TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

# How the bot describes its own scope in the router / out-of-scope refusal
# / greeting reply.
DOMAIN_LABEL = os.getenv(
    "DOMAIN_LABEL", "the regulatory compliance documents indexed in this system"
)

# How many recent exchanges (one user message + one assistant reply = one
# exchange) each chat session keeps server-side.
MAX_SESSION_TURNS = int(os.getenv("MAX_SESSION_TURNS", "10"))

# Document registry, query audit log, and analysis-run timing (KPIs) live
# in a small local SQLite database.
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "data" / "app.db")))

DOCUMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

RAGAS_GEMINI_MODEL = os.getenv("RAGAS_GEMINI_MODEL", GEMINI_CHAT_MODEL)
RAGAS_MIN_SCORE = float(os.getenv("RAGAS_MIN_SCORE", "0.70"))

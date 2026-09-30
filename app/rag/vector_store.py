import re

import chromadb

from app.config import (
    BEDROCK_EMBEDDING_MODEL,
    CHROMA_PERSIST_DIRECTORY,
    GEMINI_EMBEDDING_MODEL,
    LLM_PROVIDER,
    OPENAI_EMBEDDING_MODEL,
)

client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIRECTORY))

# Cache of collection objects, keyed by name -- see _get_collection().
_collections: dict = {}


def _active_embedding_model() -> str:
    if LLM_PROVIDER == "openai":
        return OPENAI_EMBEDDING_MODEL
    if LLM_PROVIDER == "bedrock":
        return BEDROCK_EMBEDDING_MODEL
    return GEMINI_EMBEDDING_MODEL


def _collection_name() -> str:
    """Different embedding models produce vectors of different
    dimensionality (and in different, incompatible semantic spaces), and
    a Chroma collection is permanently locked to whichever dimension its
    first vectors used. Without this, switching LLM_PROVIDER (or even
    just the embedding model) crashes on the next add/search with a
    "Collection expecting embedding with dimension of X, got Y" error.

    Giving each provider+model combo its own collection name makes a
    provider switch a clean, empty start instead of a crash -- and means
    switching back to a previously-used provider finds its old index
    still intact rather than losing it.
    """
    raw = f"regdocs_{LLM_PROVIDER}_{_active_embedding_model()}"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_-")[:63]
    slug = slug.rstrip("_-") or "regdocs_default"
    return slug


def _get_collection():
    name = _collection_name()

    if name not in _collections:
        _collections[name] = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    return _collections[name]


def add_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    if not chunks:
        return

    _get_collection().upsert(
        ids=[item["id"] for item in chunks],
        documents=[item["text"] for item in chunks],
        metadatas=[item["metadata"] for item in chunks],
        embeddings=embeddings,
    )


def delete_document(filename: str) -> int:
    """Remove every chunk belonging to one document (used before
    re-ingesting a changed file, and when a document is deleted outright,
    so stale/orphaned chunks can never be retrieved)."""
    collection = _get_collection()
    existing = collection.get(where={"document": filename}, include=[])
    removed = len(existing.get("ids", []))

    if removed:
        collection.delete(where={"document": filename})

    return removed


def search(
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    collection = _get_collection()

    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    return [
        {
            "text": document,
            "metadata": metadata,
            "distance": distance,
        }
        for document, metadata, distance in zip(
            documents, metadatas, distances
        )
    ]


def collection_stats() -> dict:
    collection = _get_collection()
    return {
        "collection": collection.name,
        "chunks": collection.count(),
    }

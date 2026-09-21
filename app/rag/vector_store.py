import chromadb

from app.config import CHROMA_PERSIST_DIRECTORY

client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIRECTORY))

collection = client.get_or_create_collection(
    name="regulatory_documents",
    metadata={"hnsw:space": "cosine"},
)


def add_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    if not chunks:
        return

    collection.upsert(
        ids=[item["id"] for item in chunks],
        documents=[item["text"] for item in chunks],
        metadatas=[item["metadata"] for item in chunks],
        embeddings=embeddings,
    )


def search(
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
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
    return {
        "collection": collection.name,
        "chunks": collection.count(),
    }

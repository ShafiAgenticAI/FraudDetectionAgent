from app.config import TOP_K
from app.llm.client import chat_completion
from app.rag.embeddings import embed_query
from app.rag.vector_store import search

SYSTEM_PROMPT = """
You are a regulatory compliance document assistant.

Your job is to answer questions using ONLY the supplied retrieved
regulatory document context.

Rules:
1. Do not invent regulatory requirements, dates, penalties, or citations.
2. If the retrieved context is insufficient, say that the indexed
   documents do not contain enough information.
3. Answer clearly and concisely.
4. Every substantive claim must be supported by the retrieved context.
5. Do not provide legal advice.
6. Use the source information supplied with each chunk when discussing
   where the answer came from.
"""


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    results = search(embed_query(question), top_k)

    if not results:
        return {
            "question": question,
            "answer": (
                "No indexed regulatory content was found. "
                "Upload and index a PDF first."
            ),
            "citations": [],
            "retrieved_chunks": 0,
        }

    context_parts = []
    citations = []

    for item in results:
        metadata = item["metadata"]

        context_parts.append(
            f"DOCUMENT: {metadata.get('document')}\n"
            f"PAGE: {metadata.get('page')}\n"
            f"CHUNK_ID: {metadata.get('chunk_id')}\n"
            f"TEXT:\n{item['text']}"
        )

        citations.append(
            {
                "document": metadata.get("document"),
                "page": metadata.get("page"),
                "chunk": metadata.get("chunk"),
                "chunk_id": metadata.get("chunk_id"),
                "distance": item.get("distance"),
            }
        )

    context = "\n\n---\n\n".join(context_parts)

    answer = chat_completion(
        SYSTEM_PROMPT,
        (
            f"Retrieved regulatory context:\n\n{context}\n\n"
            f"User question:\n{question}"
        ),
    )

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": len(results),
    }

import tiktoken

from app.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS

ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_page_text(text: str) -> list[str]:
    """
    Chunk one page using token boundaries.

    The page boundary is intentionally preserved because page-level
    citations are required by the application.
    """
    if not text.strip():
        return []

    tokens = ENCODING.encode(text)
    chunks: list[str] = []

    start = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE_TOKENS, len(tokens))

        chunk = ENCODING.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(tokens):
            break

        next_start = end - CHUNK_OVERLAP_TOKENS
        start = max(start + 1, next_start)

    return chunks


def build_chunks(
    pages: list[dict],
    filename: str,
    ingested_at: str | None = None,
) -> list[dict]:
    """Turn page records into Chroma-ready chunks with citation metadata.

    ingested_at (ISO-8601 UTC timestamp) is stamped onto every chunk's
    metadata so newer re-indexed content can later be preferred/ranked
    over stale entries if needed (data-freshness tracking).
    """
    results: list[dict] = []

    for page in pages:
        page_number = page["page"]

        for chunk_number, text in enumerate(
            chunk_page_text(page["text"]), start=1
        ):
            chunk_id = f"{filename}::p{page_number}::c{chunk_number}"

            metadata = {
                "document": filename,
                "source": filename,
                "page": page_number,
                "chunk": chunk_number,
                "chunk_id": chunk_id,
            }

            if ingested_at:
                metadata["ingested_at"] = ingested_at

            results.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                }
            )

    return results

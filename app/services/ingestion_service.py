from pathlib import Path

from app.config import DOCUMENT_DIRECTORY
from app.parsers.pdf_parser import extract_pages, pdf_stats
from app.rag.chunker import build_chunks
from app.rag.embeddings import embed_texts
from app.rag.vector_store import add_chunks


def ingest_pdf(filename: str) -> dict:
    safe_name = Path(filename).name
    pdf_path = DOCUMENT_DIRECTORY / safe_name

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be indexed.")

    pages = extract_pages(pdf_path)
    chunks = build_chunks(pages, safe_name)

    if not chunks:
        raise ValueError(
            "No readable text was extracted. "
            "If this is a scanned PDF, OCR support will be needed."
        )

    all_embeddings: list[list[float]] = []

    batch_size = 50

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_embeddings = embed_texts(
            [item["text"] for item in batch]
        )
        all_embeddings.extend(batch_embeddings)

    add_chunks(chunks, all_embeddings)

    stats = pdf_stats(pdf_path)

    return {
        **stats,
        "chunks_created": len(chunks),
        "message": "Document indexed successfully.",
    }

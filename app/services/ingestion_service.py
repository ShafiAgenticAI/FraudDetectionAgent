import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import DOCUMENT_DIRECTORY
from app.parsers.pdf_parser import extract_pages, pdf_stats
from app.rag.chunker import build_chunks
from app.rag.embeddings import embed_texts
from app.rag.vector_store import add_chunks, delete_document

# Tracks the content hash of the last-indexed version of each document, so
# re-ingesting an unchanged file is a cheap no-op instead of a full re-embed
# (see MVP question 3: "does it give yesterday's outdated policy?").
MANIFEST_PATH = DOCUMENT_DIRECTORY / ".ingestion_manifest.json"


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}

    try:
        return json.loads(MANIFEST_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest_pdf(filename: str, force: bool = False) -> dict:
    safe_name = Path(filename).name
    pdf_path = DOCUMENT_DIRECTORY / safe_name

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be indexed.")

    if not pdf_path.exists():
        raise FileNotFoundError(f"Document not found: {pdf_path}")

    manifest = _load_manifest()
    current_hash = _file_hash(pdf_path)

    if not force and manifest.get(safe_name) == current_hash:
        stats = pdf_stats(pdf_path)
        return {
            **stats,
            "chunks_created": 0,
            "message": (
                "Document content is unchanged since the last ingest; "
                "skipped re-embedding. Pass force=true to re-index anyway."
            ),
        }

    pages = extract_pages(pdf_path)
    chunks = build_chunks(
        pages,
        safe_name,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )

    if not chunks:
        raise ValueError(
            "No readable text was extracted. "
            "If this is a scanned PDF, OCR support will be needed."
        )

    # Remove any chunks left behind by a previous version of this document
    # (e.g. pages/sections that no longer exist) before adding the new set,
    # so stale content can never be retrieved alongside the fresh content.
    delete_document(safe_name)

    all_embeddings: list[list[float]] = []

    batch_size = 50

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_embeddings = embed_texts(
            [item["text"] for item in batch]
        )
        all_embeddings.extend(batch_embeddings)

    add_chunks(chunks, all_embeddings)

    manifest[safe_name] = current_hash
    _save_manifest(manifest)

    stats = pdf_stats(pdf_path)

    return {
        **stats,
        "chunks_created": len(chunks),
        "message": "Document indexed successfully.",
    }


def remove_document(filename: str) -> dict:
    """Delete a document's stored PDF, its indexed chunks, and its manifest
    entry, so it can no longer be retrieved, summarized, or queried."""
    safe_name = Path(filename).name
    pdf_path = DOCUMENT_DIRECTORY / safe_name

    if not pdf_path.exists():
        raise FileNotFoundError(f"Document not found: {pdf_path}")

    removed_chunks = delete_document(safe_name)
    pdf_path.unlink()

    manifest = _load_manifest()
    manifest.pop(safe_name, None)
    _save_manifest(manifest)

    return {
        "filename": safe_name,
        "chunks_removed": removed_chunks,
        "message": "Document and its indexed chunks were removed.",
    }

from pathlib import Path

import fitz


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text page-by-page while preserving original 1-based page numbers."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[dict] = []

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


def pdf_stats(pdf_path: Path) -> dict:
    """Return basic PDF statistics useful before ingestion."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with fitz.open(pdf_path) as document:
        return {
            "filename": pdf_path.name,
            "total_pages": len(document),
            "pages_with_text": sum(
                1 for page in document if page.get_text("text").strip()
            ),
        }

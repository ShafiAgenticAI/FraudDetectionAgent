from pathlib import Path

from app.config import DOCUMENT_DIRECTORY
from app.llm.client import chat_completion
from app.parsers.pdf_parser import extract_pages
from app.rag.chunker import build_chunks


def summarize_document(filename: str) -> dict:
    path = DOCUMENT_DIRECTORY / Path(filename).name

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    pages = extract_pages(path)
    chunks = build_chunks(pages, path.name)

    partial_summaries = []

    # Map step: summarize manageable groups of chunks.
    for start in range(0, len(chunks), 8):
        batch = chunks[start : start + 8]

        text = "\n\n".join(
            (
                f"Page {item['metadata']['page']}, "
                f"Chunk {item['metadata']['chunk']}:\n"
                f"{item['text']}"
            )
            for item in batch
        )

        summary_text = chat_completion(
            (
                "Summarize only the supplied regulatory text. "
                "Identify important requirements, changes, risks, "
                "and business implications. Do not invent facts, figures, "
                "or requirements that are not present in the text. If a "
                "section has nothing substantive to summarize, state that "
                "plainly instead of filling in plausible-sounding content."
            ),
            text,
        )

        partial_summaries.append(summary_text)

    combined = "\n\n---\n\n".join(partial_summaries)

    # Reduce step: combine intermediate summaries.
    final_summary = chat_completion(
        (
            "Create an executive regulatory summary using ONLY "
            "the supplied intermediate summaries. Use these sections:\n"
            "1. Executive Summary\n"
            "2. Key Obligations\n"
            "3. Key Changes\n"
            "4. Key Risks\n"
            "5. Business Impact\n\n"
            "Do not invent facts."
        ),
        combined,
    )

    return {
        "filename": path.name,
        "summary": final_summary,
    }

import json
from pathlib import Path

from app.config import DOCUMENT_DIRECTORY
from app.llm.client import chat_completion
from app.parsers.pdf_parser import extract_pages
from app.rag.chunker import build_chunks

PROMPT = """
You are a regulatory compliance analyst.

Extract only obligations explicitly supported by the supplied
regulatory text.

Return valid JSON in exactly this general structure:
{
  "obligations": [
    {
      "obligation": "text",
      "type": "Reporting|Retention|Deadline|Restriction|Recordkeeping|Other",
      "deadline": "text or null",
      "source": "document name",
      "page": 1,
      "chunk": 1,
      "evidence": "short supporting excerpt"
    }
  ]
}

Rules:
- Do not infer obligations that are not stated.
- Use null when a field is not available.
- Preserve the source page and chunk supplied in the context.
- Return an empty obligations list if there are no explicit obligations.
"""


def extract_obligations(filename: str) -> dict:
    path = DOCUMENT_DIRECTORY / Path(filename).name

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    pages = extract_pages(path)
    chunks = build_chunks(pages, path.name)

    extracted = []

    # Batch chunks to keep prompts manageable.
    for start in range(0, len(chunks), 5):
        batch = chunks[start : start + 5]

        context = "\n\n---\n\n".join(
            (
                f"DOCUMENT: {item['metadata']['document']}\n"
                f"PAGE: {item['metadata']['page']}\n"
                f"CHUNK: {item['metadata']['chunk']}\n"
                f"CHUNK_ID: {item['metadata']['chunk_id']}\n"
                f"TEXT:\n{item['text']}"
            )
            for item in batch
        )

        response_text = chat_completion(
            PROMPT,
            context,
            json_mode=True,
        )

        data = json.loads(response_text)
        extracted.extend(data.get("obligations", []))

    return {
        "filename": path.name,
        "obligations": extracted,
        "count": len(extracted),
    }

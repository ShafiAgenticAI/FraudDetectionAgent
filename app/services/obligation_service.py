import json
import logging
import time
from pathlib import Path

from app.config import DOCUMENT_DIRECTORY
from app.db import log_analysis_run
from app.llm.client import chat_completion
from app.parsers.pdf_parser import extract_pages
from app.rag.chunker import build_chunks

logger = logging.getLogger("compliance_copilot.analysis")

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
      "severity": "High|Medium|Low",
      "source": "document name",
      "page": 1,
      "chunk": 1,
      "evidence": "short supporting excerpt"
    }
  ]
}

Severity guidance:
- "High": explicit legal/regulatory deadline, mandatory filing, or a
  requirement whose breach is described as a violation/penalty.
- "Medium": a clear obligation without an explicit deadline or
  penalty language.
- "Low": a recommendation, best practice, or advisory statement rather
  than a binding requirement.

Rules:
- Do not infer obligations that are not explicitly stated in the text.
- Never fabricate a deadline, figure, or requirement to fill a field.
- Use null when a field is not available.
- Preserve the source page and chunk supplied in the context.
- Return an empty obligations list if there are no explicit obligations
  in the supplied text -- an empty list is the correct, expected output
  for text with no obligations, not a failure.
"""


def extract_obligations(filename: str) -> dict:
    start = time.perf_counter()
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

    response_time_ms = int((time.perf_counter() - start) * 1000)
    log_analysis_run(path.name, "obligations", response_time_ms)

    logger.info(
        "OBLIGATIONS complete | file=%s | count=%d | time_ms=%d",
        path.name,
        len(extracted),
        response_time_ms,
    )

    return {
        "filename": path.name,
        "obligations": extracted,
        "count": len(extracted),
        "response_time_ms": response_time_ms,
    }

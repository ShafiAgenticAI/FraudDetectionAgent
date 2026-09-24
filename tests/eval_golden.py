"""Lightweight RAG-triad-style regression harness.

This is a starter evaluator, not a Ragas/TruLens integration -- it reuses
the app's own chat_completion() as an LLM judge so it needs no new
dependency. Swap in Ragas/TruLens later if you want standardized metrics,
dashboards, or CI integration; the golden dataset format here (a simple
list of {question, expected_pages}) would carry over.

Usage:
    1. Ingest the document(s) this golden set is written against.
    2. Fill out tests/golden_qa.json (copy golden_qa.sample.json and
       replace the placeholder rows with real, expert-reviewed Q&A pairs
       -- the doc's target is 50-100 pairs before a release).
    3. Run:  python tests/eval_golden.py [path/to/golden_qa.json]

For every question this scores, on a 1-5 scale via LLM judge:
  - context_relevance : did retrieval pull the right material?
  - faithfulness       : did the answer stick to the retrieved context?
  - answer_relevance   : did the answer address what was actually asked?
and separately checks (exact, not LLM-judged) whether any of the
citation pages returned overlap the golden "expected_pages", when given.

Exits non-zero if the average of any metric falls below MIN_SCORE, so
this can be wired into a pre-deployment check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.llm.client import chat_completion
from app.services.rag_service import answer_question

MIN_SCORE = 3.5

JUDGE_PROMPT = """
You are grading one answer from a regulatory-compliance RAG system.

Score each of the following from 1 (worst) to 5 (best). Return ONLY a
JSON object, no other text:
{
  "context_relevance": <1-5, did the retrieved context actually relate
      to the question?>,
  "faithfulness": <1-5, does the answer only state things supported by
      the retrieved context, with no fabricated facts?>,
  "answer_relevance": <1-5, does the answer directly address the
      question asked?>
}
"""


def _judge(question: str, context: str, answer: str) -> dict:
    raw = chat_completion(
        JUDGE_PROMPT,
        (
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"System's answer:\n{answer}"
        ),
        json_mode=True,
    )

    return json.loads(raw)


def run(golden_path: Path) -> int:
    golden_set = json.loads(golden_path.read_text())

    totals = {"context_relevance": 0.0, "faithfulness": 0.0, "answer_relevance": 0.0}
    page_hits = 0
    page_checks = 0

    for row in golden_set:
        question = row["question"]
        expected_pages = set(row.get("expected_pages", []))

        result = answer_question(question)
        context = "\n\n".join(
            f"(page {c['page']}) {c['document']}" for c in result["citations"]
        )

        scores = _judge(question, context, result["answer"])

        for key in totals:
            totals[key] += scores.get(key, 0)

        cited_pages = {c["page"] for c in result["citations"]}

        if expected_pages:
            page_checks += 1
            if expected_pages & cited_pages:
                page_hits += 1

        print(
            f"- {question}\n"
            f"  scores: {scores}  grounded={result['grounded']}  "
            f"cited_pages={sorted(cited_pages)}"
        )

    count = len(golden_set) or 1
    averages = {key: round(total / count, 2) for key, total in totals.items()}

    print("\n=== Averages ===")
    for key, value in averages.items():
        print(f"{key}: {value}")

    if page_checks:
        print(f"citation page hit rate: {page_hits}/{page_checks}")

    failed = [key for key, value in averages.items() if value < MIN_SCORE]

    if failed:
        print(f"\nFAILED thresholds (< {MIN_SCORE}): {failed}")
        return 1

    print("\nAll metrics passed threshold.")
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "golden_qa.json"

    if not path.exists():
        print(
            f"{path} not found. Copy golden_qa.sample.json to golden_qa.json "
            "and fill in real, expert-reviewed Q&A pairs first."
        )
        sys.exit(1)

    sys.exit(run(path))

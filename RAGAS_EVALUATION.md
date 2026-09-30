# Ragas evaluation workflow

## Golden set
`tests/golden_qa.json` contains 40 initial questions grounded in FINRA Regulatory Notice 11-19. Each row includes:
- `question`: test question
- `reference_answer`: expected answer for Ragas reference-based metrics
- `expected_pages`: pages that should support the answer
- `expected_context`: human-readable evidence target

These are an initial engineering golden set; before a release, have compliance SMEs review the answers/pages and expand toward the MVP target of 50-100 expert-reviewed pairs.

## Run

```bash
python tests/eval_ragas.py
```

Results are written to `data/evaluations/` as CSV, JSON and Markdown.

## Analytics

Start FastAPI and Streamlit, open **Analytics**, then click **Run Ragas Evaluation**. The latest metrics and evaluation history are stored in SQLite.

## Improvement loop

1. Run evaluation.
2. Inspect low Context Precision/Recall for retrieval problems.
3. Inspect low Faithfulness for unsupported answers.
4. Inspect low Answer Relevancy for poor answer focus.
5. Inspect citation page hit rate for citation/retrieval problems.
6. Fix retrieval, chunking, prompts, or golden labels.
7. Re-index documents when chunking/embedding settings change.
8. Re-run evaluation and compare the stored run history.

Ragas is offline evaluation; it is not invoked for every user question.

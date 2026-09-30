"""Run standardized Ragas evaluation against the regulatory RAG pipeline.

Usage:
    python tests/eval_ragas.py
    python tests/eval_ragas.py tests/golden_qa.json

The evaluator runs the real answer_question() path, captures retrieved
contexts and citations, then sends the dataset to Ragas. It also calculates
an exact citation-page hit rate against expected_pages.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "evaluations"
OUT.mkdir(parents=True, exist_ok=True)

from app.services.rag_service import answer_question


def _load_metrics():
    # Ragas has changed metric class names across releases; support the
    # common 0.2/0.3 APIs rather than hard-coding one minor release.
    try:
        from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference, LLMContextRecall
        return Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference, LLMContextRecall
    except ImportError:
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        return faithfulness, answer_relevancy, context_precision, context_recall


def _build_dataset(rows):
    from datasets import Dataset
    records = []
    for row in rows:
        result = answer_question(row["question"], top_k=5)
        contexts = []
        for citation in result.get("citations", []):
            # answer_question currently returns metadata, but the text is not
            # exposed. Re-query through the public vector store for evaluation.
            pass
        # Retrieve the same query directly so Ragas receives actual text.
        from app.rag.embeddings import embed_query
        from app.rag.vector_store import search
        search_query = result.get("search_query") or row["question"]
        retrieved = search(embed_query(search_query), 5)
        contexts = [item["text"] for item in retrieved]
        records.append({
            "question": row["question"],
            "answer": result["answer"],
            "contexts": contexts,
            "ground_truth": row.get("reference_answer", ""),
        })
    return Dataset.from_list(records)


def _build_judge():
    """Build the Ragas judge LLM + embeddings from whichever provider the
    main app is configured for (LLM_PROVIDER), so evaluation doesn't
    depend on a specific provider's quota being available. Requires the
    matching LangChain integration package for that provider:
    langchain-openai / langchain-google-genai / langchain-aws.
    """
    from app.config import LLM_PROVIDER

    if LLM_PROVIDER == "openai":
        from app.config import OPENAI_API_KEY, OPENAI_CHAT_MODEL, OPENAI_EMBEDDING_MODEL
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required for Ragas evaluation")
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        return llm, embeddings

    if LLM_PROVIDER == "bedrock":
        from app.config import AWS_REGION, BEDROCK_CHAT_MODEL, BEDROCK_EMBEDDING_MODEL
        from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
        llm = ChatBedrockConverse(model=BEDROCK_CHAT_MODEL, region_name=AWS_REGION, temperature=0)
        embeddings = BedrockEmbeddings(model_id=BEDROCK_EMBEDDING_MODEL, region_name=AWS_REGION)
        return llm, embeddings

    # default: gemini
    from app.config import GEMINI_API_KEY, RAGAS_GEMINI_MODEL
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required for Ragas evaluation")
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    llm = ChatGoogleGenerativeAI(model=RAGAS_GEMINI_MODEL, google_api_key=GEMINI_API_KEY, temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)
    return llm, embeddings


def run(golden_path: Path) -> int:
    rows = json.loads(golden_path.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("Golden dataset is empty")

    dataset = _build_dataset(rows)

    llm, embeddings = _build_judge()

    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        evaluator_llm = LangchainLLMWrapper(llm)
        evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    except ImportError:
        evaluator_llm = llm
        evaluator_embeddings = embeddings

    m1, m2, m3, m4 = _load_metrics()
    metrics = [m1, m2, m3, m4]
    for metric in metrics:
        if hasattr(metric, "llm"):
            metric.llm = evaluator_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = evaluator_embeddings

    from ragas import evaluate
    result = evaluate(dataset, metrics=metrics)
    scores = result.to_pandas()

    # Exact expected-page check uses the actual answer pipeline citations.
    page_hits = []
    for row in rows:
        response = answer_question(row["question"], top_k=5)
        cited = {int(c["page"]) for c in response.get("citations", []) if c.get("page") is not None}
        expected = {int(p) for p in row.get("expected_pages", [])}
        page_hits.append(bool(cited & expected) if expected else True)
    citation_hit_rate = sum(page_hits) / len(page_hits)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = OUT / f"ragas_{stamp}.csv"
    json_path = OUT / f"ragas_{stamp}.json"
    md_path = OUT / f"ragas_{stamp}.md"
    scores.to_csv(csv_path, index=False)

    summary = {column: float(scores[column].mean()) for column in scores.columns if scores[column].dtype != "object"}
    payload = {
        "timestamp": stamp,
        "golden_questions": len(rows),
        "metrics": summary,
        "citation_page_hit_rate": citation_hit_rate,
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        "# Ragas Evaluation\n\n"
        f"Run: {stamp}\n\n"
        f"Golden questions: {len(rows)}\n\n"
        "## Metrics\n\n" +
        "\n".join(f"- **{k}**: {v:.3f}" for k, v in summary.items()) +
        f"\n- **Citation page hit rate**: {citation_hit_rate:.3f}\n",
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "golden_qa.json"
    raise SystemExit(run(path))

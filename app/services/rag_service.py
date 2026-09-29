import logging
import time

from app.config import DOMAIN_LABEL, TOP_K
from app.db import log_query
from app.llm.client import chat_completion
from app.rag.embeddings import embed_query
from app.rag.vector_store import search
from app.services import session_store
from app.services.intent_router import classify_intent

logger = logging.getLogger("compliance_copilot.chat")

# Used verbatim so callers (and tests) can reliably detect an abstention
# instead of parsing free text for something that "sounds like" a refusal.
FALLBACK_MESSAGE = (
    "I do not have sufficient information in the provided documentation "
    "to answer that."
)

OUT_OF_SCOPE_MESSAGE = (
    f"I'm designed specifically to answer questions about {DOMAIN_LABEL}. "
    "I can't help with topics outside that scope."
)

SYSTEM_PROMPT = f"""
You are a regulatory compliance document assistant.

Your job is to answer questions using ONLY the supplied retrieved
regulatory document context.

Rules:
1. Do not invent regulatory requirements, dates, penalties, or citations.
2. If the retrieved context does not contain enough information to answer
   confidently, respond with EXACTLY this sentence and nothing else:
   "{FALLBACK_MESSAGE}"
3. Answer clearly and concisely.
4. Every substantive claim must be supported by the retrieved context.
5. Do not provide legal advice.
6. Use the source information supplied with each chunk when discussing
   where the answer came from.
"""

GREETING_PROMPT = f"""
You are a friendly front door to an assistant whose only job is
answering questions about {DOMAIN_LABEL}.

The user's message is small talk (a greeting, thanks, goodbye, or a
"who are you / what can you do" question) -- NOT a substantive question.

Reply briefly and warmly, and mention in one sentence that you can
answer questions about {DOMAIN_LABEL}. Do not attempt to answer any
substantive compliance question here, even if one is implied.
"""

SESSION_PROMPT = """
You answer questions using ONLY the current chat session context supplied below.
Do not use outside knowledge and do not perform regulatory document retrieval.
If the requested information is not present in the session context, say exactly:
"I don't have that information in this chat session."
Keep the answer brief.
"""

CONDENSE_PROMPT = """
You rewrite a user's latest chat message into a standalone search query,
using the prior conversation as context.

Rules:
- Preserve the user's intent exactly; do not answer the question yourself.
- Resolve pronouns and vague references ("it", "that policy", "this
  requirement") using the conversation history.
- If the latest message is already a standalone question, return it
  unchanged.
- Return ONLY the rewritten question as plain text. No preamble, no
  quotation marks, no explanation.
"""

# Heuristic faithfulness/groundedness check: what fraction of the
# meaningful (>4 char) words in the answer also appear in the retrieved
# context. This is NOT a substitute for a real RAG-triad evaluator (e.g.
# Ragas/TruLens with an LLM judge) -- it's a cheap, zero-extra-API-call
# guardrail that catches obviously unsupported answers so the UI can flag
# them, without adding latency or cost to every request.
GROUNDEDNESS_OVERLAP_THRESHOLD = 0.25


def _condense_question(question: str, history: list[dict] | None) -> str:
    if not history:
        return question

    transcript = "\n".join(
        f"{turn.get('role', 'user')}: {turn.get('content', '')}"
        for turn in history[-6:]  # last few turns is enough context
    )

    rewritten = chat_completion(
        CONDENSE_PROMPT,
        f"Conversation history:\n{transcript}\n\nLatest message:\n{question}",
    )

    rewritten = (rewritten or "").strip()

    return rewritten or question


def _is_grounded(answer: str, context_parts: list[str]) -> bool:
    if answer.strip() == FALLBACK_MESSAGE:
        return True  # correctly abstained -- nothing to check

    answer_words = {w.lower() for w in answer.split() if len(w) > 4}

    if not answer_words:
        return True

    context_words = {
        w.lower() for w in " ".join(context_parts).split() if len(w) > 4
    }

    overlap = len(answer_words & context_words) / len(answer_words)

    return overlap >= GROUNDEDNESS_OVERLAP_THRESHOLD


def _remember(session_id: str | None, question: str, answer: str) -> None:
    if session_id:
        session_store.remember_turn(session_id, question, answer)


def _session_answer(question: str, history: list[dict], session_id: str | None) -> str:
    facts = session_store.get_facts(session_id) if session_id else {}
    name = facts.get("name")

    # Deterministic handling for the most important explicit session fact.
    normalized = question.strip().lower().rstrip("?!.")
    if normalized in {"what is my name", "whats my name", "do you know my name", "who am i"}:
        return f"Your name is {name}." if name else "I don't have your name in this chat session."

    transcript = "\n".join(
        f"{turn.get('role', 'user')}: {turn.get('content', '')}"
        for turn in history[-10:]
    )
    facts_text = "\n".join(f"{k}: {v}" for k, v in facts.items()) or "(none)"
    return chat_completion(
        SESSION_PROMPT,
        f"Session facts:\n{facts_text}\n\nConversation:\n{transcript}\n\nUser question:\n{question}",
    ).strip()


def _finish(result: dict, start: float, session_id: str | None) -> dict:
    """Stamp response_time_ms onto the result, write it to the query
    audit log, and print a live line to the terminal -- this is what
    backs the live "Response Time" and "Citation Coverage" KPIs from the
    doc's MVP Success Criteria, and lets you watch questions/citations
    go by in real time while the backend is running."""
    response_time_ms = int((time.perf_counter() - start) * 1000)
    result["response_time_ms"] = response_time_ms

    log_query(
        session_id=session_id,
        question=result["question"],
        answer=result["answer"],
        intent=result.get("intent"),
        grounded=result.get("grounded", True),
        citation_count=len(result.get("citations") or []),
        response_time_ms=response_time_ms,
    )

    citations = result.get("citations") or []
    citation_summary = (
        "; ".join(
            f"{c['document']} p.{c['page']} chunk={c['chunk_id']}"
            for c in citations
        )
        if citations
        else "none"
    )

    logger.info(
        "QUESTION=%r | intent=%s | grounded=%s | time_ms=%d | "
        "citations=[%s] | ANSWER=%r",
        result["question"],
        result.get("intent"),
        result.get("grounded"),
        response_time_ms,
        citation_summary,
        (result["answer"][:200] + "…")
        if len(result["answer"]) > 200
        else result["answer"],
    )

    return result


def answer_question(
    question: str,
    top_k: int = TOP_K,
    history: list[dict] | None = None,
    session_id: str | None = None,
) -> dict:
    start = time.perf_counter()

    # A session_id lets the server own "recent conversation" instead of
    # requiring the caller to resend the whole transcript every request.
    # An explicitly-passed history always wins (stateless API callers).
    if history is None and session_id:
        history = session_store.get_history(session_id)

    if session_id:
        session_store.remember_explicit_facts(session_id, question)

    intent = classify_intent(question, history)

    if intent == "greeting":
        answer = chat_completion(GREETING_PROMPT, question)
        _remember(session_id, question, answer)

        return _finish(
            {
                "question": question,
                "search_query": question,
                "intent": intent,
                "answer": answer,
                "citations": [],
                "retrieved_chunks": 0,
                "grounded": True,
            },
            start,
            session_id,
        )

    if intent == "session_query":
        current_history = history or (session_store.get_history(session_id) if session_id else [])
        answer = _session_answer(question, current_history, session_id)
        _remember(session_id, question, answer)

        return _finish(
            {
                "question": question,
                "search_query": question,
                "intent": intent,
                "answer": answer,
                "citations": [],
                "retrieved_chunks": 0,
                "grounded": True,
            },
            start,
            session_id,
        )

    if intent == "out_of_scope":
        _remember(session_id, question, OUT_OF_SCOPE_MESSAGE)

        return _finish(
            {
                "question": question,
                "search_query": question,
                "intent": intent,
                "answer": OUT_OF_SCOPE_MESSAGE,
                "citations": [],
                "retrieved_chunks": 0,
                "grounded": True,
            },
            start,
            session_id,
        )

    # intent == "domain_query" -- standard retrieval + generation path.
    search_query = _condense_question(question, history)

    results = search(embed_query(search_query), top_k)

    if not results:
        _remember(session_id, question, FALLBACK_MESSAGE)

        return _finish(
            {
                "question": question,
                "search_query": search_query,
                "intent": intent,
                "answer": FALLBACK_MESSAGE,
                "citations": [],
                "retrieved_chunks": 0,
                "grounded": True,
            },
            start,
            session_id,
        )

    context_parts = []
    citations = []

    for item in results:
        metadata = item["metadata"]

        context_parts.append(
            f"DOCUMENT: {metadata.get('document')}\n"
            f"PAGE: {metadata.get('page')}\n"
            f"CHUNK_ID: {metadata.get('chunk_id')}\n"
            f"TEXT:\n{item['text']}"
        )

        citations.append(
            {
                "document": metadata.get("document"),
                "page": metadata.get("page"),
                "chunk": metadata.get("chunk"),
                "chunk_id": metadata.get("chunk_id"),
                "distance": item.get("distance"),
            }
        )

    context = "\n\n---\n\n".join(context_parts)

    answer = chat_completion(
        SYSTEM_PROMPT,
        (
            f"Retrieved regulatory context:\n\n{context}\n\n"
            f"User question:\n{search_query}"
        ),
    )

    _remember(session_id, question, answer)

    return _finish(
        {
            "question": question,
            "search_query": search_query,
            "intent": intent,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": len(results),
            "grounded": _is_grounded(answer, context_parts),
        },
        start,
        session_id,
    )

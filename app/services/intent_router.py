"""Query intent router.

Classifies each incoming chat message BEFORE retrieval runs, so the app
can skip embedding + vector search entirely for greetings and
out-of-scope questions (saving latency and embedding cost on every such
message), and can give out-of-scope questions a clean, deterministic
refusal instead of letting the LLM freewheel into a general-knowledge
answer it was never grounded to give.

This uses a small LLM call with structured (JSON) output. A cheaper
alternative -- swap this for an embedding-similarity classifier (compare
the question's embedding against a few reference "in-domain" example
questions) -- would remove even this one small LLM call; left as a
straightforward follow-up if router latency/cost ever matters at volume.
"""

import json

from app.config import DOMAIN_LABEL
from app.llm.client import chat_completion

VALID_CATEGORIES = {"greeting", "domain_query", "out_of_scope"}

ROUTER_PROMPT = f"""
You classify a single user chat message for an assistant whose ONLY job
is answering questions about {DOMAIN_LABEL}.

Return ONLY a JSON object of exactly this shape, nothing else:
{{"category": "greeting" | "domain_query" | "out_of_scope"}}

Category definitions:
- "greeting": small talk, hello/thanks/goodbye, or asking who the
  assistant is / what it can do. No retrieval is needed to answer these.
- "domain_query": a question that could plausibly be answered from
  {DOMAIN_LABEL}, including a follow-up question that refers back to an
  earlier answer in the conversation.
- "out_of_scope": anything else -- general knowledge, coding help,
  recipes, personal advice, or any topic unrelated to {DOMAIN_LABEL}.

When genuinely unsure between domain_query and out_of_scope, choose
domain_query so a real question is never wrongly refused.
"""


def classify_intent(question: str, history: list[dict] | None = None) -> str:
    transcript = ""

    if history:
        transcript = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '')}"
            for turn in history[-4:]
        )

    user_prompt = (
        f"Recent conversation:\n{transcript}\n\nMessage to classify:\n{question}"
        if transcript
        else f"Message to classify:\n{question}"
    )

    category = None

    try:
        raw = chat_completion(ROUTER_PROMPT, user_prompt, json_mode=True)
        category = json.loads(raw).get("category")
    except Exception:
        category = None

    # Fail open into domain_query: a router error should never silently
    # turn into a wrongly-refused real question.
    return category if category in VALID_CATEGORIES else "domain_query"

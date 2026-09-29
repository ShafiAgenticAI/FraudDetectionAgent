"""Query intent router.

Routes greetings, session-memory questions, regulatory questions, and
out-of-scope questions before retrieval runs.
"""

import json

from app.config import DOMAIN_LABEL
from app.llm.client import chat_completion

VALID_CATEGORIES = {"greeting", "session_query", "domain_query", "out_of_scope"}

ROUTER_PROMPT = f"""
You classify a single user chat message for an assistant whose main job is
answering questions about {DOMAIN_LABEL}.

Return ONLY a JSON object of exactly this shape, nothing else:
{{"category": "greeting" | "session_query" | "domain_query" | "out_of_scope"}}

Category definitions:
- "greeting": greetings, thanks, goodbye, or "who are you / what can you do".
- "session_query": a question answerable from the current conversation/session,
  such as "what is my name?", "what did I just ask?", or "what did you say
  earlier?". Do NOT use this for regulatory follow-ups that need the document.
- "domain_query": a question that could plausibly be answered from
  {DOMAIN_LABEL}, including follow-ups referring to an earlier regulatory answer.
- "out_of_scope": unrelated general knowledge, coding, recipes, personal advice,
  or other topics outside the assistant's purpose.

When unsure between domain_query and out_of_scope, choose domain_query.
"""


def classify_intent(question: str, history: list[dict] | None = None) -> str:
    transcript = ""
    if history:
        transcript = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '')}"
            for turn in history[-6:]
        )

    user_prompt = (
        f"Recent conversation:\n{transcript}\n\nMessage to classify:\n{question}"
        if transcript
        else f"Message to classify:\n{question}"
    )

    try:
        raw = chat_completion(ROUTER_PROMPT, user_prompt, json_mode=True)
        category = json.loads(raw).get("category")
    except Exception:
        category = None

    return category if category in VALID_CATEGORIES else "domain_query"

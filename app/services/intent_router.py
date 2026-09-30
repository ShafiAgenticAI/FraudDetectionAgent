"""Pre-retrieval intent router with a session-memory category."""
import json
from app.config import DOMAIN_LABEL
from app.llm.client import chat_completion

VALID_CATEGORIES = {"greeting", "session_query", "domain_query", "out_of_scope"}

ROUTER_PROMPT = f"""
Classify one user chat message for an assistant whose main job is answering questions about {DOMAIN_LABEL}.
Return ONLY JSON: {{"category": "greeting" | "session_query" | "domain_query" | "out_of_scope"}}

- greeting: hello, thanks, goodbye, who are you, what can you do, or small talk.
- session_query: asks about information explicitly stated in this current conversation/session,
  such as the user's name, what was just discussed, or a prior answer. Do not retrieve documents.
- domain_query: a regulatory-compliance question, including follow-ups referring to an earlier regulatory answer.
- out_of_scope: unrelated general knowledge, coding, recipes, personal advice, etc.
When unsure between domain_query and out_of_scope, choose domain_query.
"""

def classify_intent(question: str, history: list[dict] | None = None, facts: dict | None = None) -> str:
    transcript = ""
    if history:
        transcript = "\n".join(f"{t.get('role','user')}: {t.get('content','')}" for t in history[-6:])
    facts_text = "\n".join(f"{k}: {v}" for k, v in (facts or {}).items()) or "none"
    prompt = f"Session facts:\n{facts_text}\n\nRecent conversation:\n{transcript or 'none'}\n\nMessage:\n{question}"
    try:
        raw = chat_completion(ROUTER_PROMPT, prompt, json_mode=True)
        category = json.loads(raw).get("category")
        if category in VALID_CATEGORIES:
            return category
    except Exception:
        pass
    return "domain_query"

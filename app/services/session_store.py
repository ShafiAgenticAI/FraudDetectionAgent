"""Process-local conversation memory for the MVP.

Stores recent turns plus a small set of explicitly stated session facts.
This is intentionally short-lived and resets when the API process restarts.
For production, move this state to Redis/PostgreSQL with a TTL.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque

from app.config import MAX_SESSION_TURNS

_sessions: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_SESSION_TURNS * 2))
_facts: dict[str, dict[str, str]] = defaultdict(dict)

_NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z .'-]{0,60})", re.I),
    re.compile(r"\bI am\s+([A-Za-z][A-Za-z .'-]{0,60})", re.I),
    re.compile(r"\bI'm\s+([A-Za-z][A-Za-z .'-]{0,60})", re.I),
]

def get_history(session_id: str) -> list[dict]:
    return list(_sessions[session_id])

def get_facts(session_id: str) -> dict[str, str]:
    return dict(_facts.get(session_id, {}))

def remember_turn(session_id: str, question: str, answer: str) -> None:
    _sessions[session_id].append({"role": "user", "content": question})
    _sessions[session_id].append({"role": "assistant", "content": answer})
    for pattern in _NAME_PATTERNS:
        match = pattern.search(question)
        if match:
            name = match.group(1).strip(" .,!?:;\n\t")
            # Avoid accidentally capturing a whole sentence.
            name = re.split(r"\b(?:and|but|because|who|what|can|please)\b", name, maxsplit=1, flags=re.I)[0].strip()
            if 1 <= len(name) <= 60:
                _facts[session_id]["name"] = name
            break

def clear_session(session_id: str) -> bool:
    existed = session_id in _sessions or session_id in _facts
    _sessions.pop(session_id, None)
    _facts.pop(session_id, None)
    return existed

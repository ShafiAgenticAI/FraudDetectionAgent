"""Process-local conversation memory for the MVP.

Stores recent chat turns plus a very small set of explicitly stated session
facts (currently the user's name). This is intentionally session-scoped: it
is cleared with the chat session and disappears when the backend restarts.
For production, move this to Redis/PostgreSQL with a TTL.
"""

from collections import defaultdict, deque
import re

from app.config import MAX_SESSION_TURNS

_sessions: dict[str, deque] = defaultdict(
    lambda: deque(maxlen=MAX_SESSION_TURNS * 2)
)
_facts: dict[str, dict[str, str]] = defaultdict(dict)

_NAME_PATTERNS = (
    re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z .'-]{0,49})", re.I),
    re.compile(r"\bcall me\s+([A-Za-z][A-Za-z .'-]{0,49})", re.I),
    re.compile(r"\bi am\s+([A-Za-z][A-Za-z .'-]{0,49})", re.I),
    re.compile(r"\bi'm\s+([A-Za-z][A-Za-z .'-]{0,49})", re.I),
)


def get_history(session_id: str) -> list[dict]:
    return list(_sessions[session_id])


def remember_turn(session_id: str, question: str, answer: str) -> None:
    _sessions[session_id].append({"role": "user", "content": question})
    _sessions[session_id].append({"role": "assistant", "content": answer})
    remember_explicit_facts(session_id, question)


def remember_explicit_facts(session_id: str, text: str) -> None:
    """Store only facts explicitly stated by the user.

    We deliberately keep this conservative; this is not a general-purpose
    long-term memory system.
    """
    for pattern in _NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            name = match.group(1).strip(" .,!?:;\t\n")
            # Avoid capturing a long sentence after "I am".
            name = re.split(r"\b(?:and|but|from|working|a|an)\b", name, maxsplit=1, flags=re.I)[0].strip()
            if 1 <= len(name) <= 50:
                _facts[session_id]["name"] = name
                return


def get_facts(session_id: str) -> dict[str, str]:
    return dict(_facts.get(session_id, {}))


def get_name(session_id: str) -> str | None:
    return _facts.get(session_id, {}).get("name")


def clear_session(session_id: str) -> bool:
    existed = session_id in _sessions or session_id in _facts
    _sessions.pop(session_id, None)
    _facts.pop(session_id, None)
    return existed

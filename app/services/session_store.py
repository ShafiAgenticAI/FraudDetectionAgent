"""In-memory chat session cache.

Keeps the most recent MAX_SESSION_TURNS exchanges (one user message + one
assistant reply = one exchange) of a conversation, keyed by a client-
supplied session_id, so callers don't have to resend the full transcript
on every request -- the server is the source of truth for "recent
conversation" instead of the client.

This is process-local, in-memory state: fine for a single-instance MVP,
but it resets on restart and won't be shared across multiple app
instances. A real deployment would move this to Redis or a small DB
table keyed by session_id with a TTL.
"""

from collections import defaultdict, deque

from app.config import MAX_SESSION_TURNS

_sessions: dict[str, deque] = defaultdict(
    lambda: deque(maxlen=MAX_SESSION_TURNS * 2)
)


def get_history(session_id: str) -> list[dict]:
    return list(_sessions[session_id])


def remember_turn(session_id: str, question: str, answer: str) -> None:
    _sessions[session_id].append({"role": "user", "content": question})
    _sessions[session_id].append({"role": "assistant", "content": answer})


def clear_session(session_id: str) -> bool:
    existed = session_id in _sessions
    _sessions.pop(session_id, None)
    return existed

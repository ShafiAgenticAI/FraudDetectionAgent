from app.services import session_store


def test_explicit_name_is_stored_and_cleared():
    sid = "test-name-session"
    session_store.clear_session(sid)
    session_store.remember_explicit_facts(sid, "Hi my name is Hemanth")
    assert session_store.get_name(sid) == "Hemanth"
    session_store.clear_session(sid)
    assert session_store.get_name(sid) is None


def test_recent_history_is_kept():
    sid = "test-history-session"
    session_store.clear_session(sid)
    session_store.remember_turn(sid, "hello", "hi")
    history = session_store.get_history(sid)
    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    session_store.clear_session(sid)

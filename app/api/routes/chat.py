from fastapi import APIRouter, HTTPException

from app.api.schemas.documents import QueryRequest
from app.services import session_store
from app.services.rag_service import answer_question

router = APIRouter()


@router.post("/query")
def query(request: QueryRequest):
    try:
        history = (
            [turn.model_dump() for turn in request.history]
            if request.history
            else None
        )
        return answer_question(
            request.question,
            request.top_k,
            history=history,
            session_id=request.session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Drop a chat session's cached conversation history (used by the
    UI's "clear conversation" action)."""
    existed = session_store.clear_session(session_id)

    return {"session_id": session_id, "cleared": existed}

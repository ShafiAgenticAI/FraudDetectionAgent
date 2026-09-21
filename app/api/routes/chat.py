from fastapi import APIRouter, HTTPException

from app.api.schemas.documents import QueryRequest
from app.services.rag_service import answer_question

router = APIRouter()


@router.post("/query")
def query(request: QueryRequest):
    try:
        return answer_question(request.question, request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

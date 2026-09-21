from fastapi import APIRouter, HTTPException

from app.api.schemas.documents import AnalysisRequest
from app.services.obligation_service import extract_obligations
from app.services.summary_service import summarize_document

router = APIRouter()


@router.post("/obligations")
def obligations(request: AnalysisRequest):
    try:
        return extract_obligations(request.filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/summary")
def summary(request: AnalysisRequest):
    try:
        return summarize_document(request.filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

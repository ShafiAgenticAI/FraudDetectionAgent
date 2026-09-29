from fastapi import APIRouter

from app.db import get_kpis, recent_queries

router = APIRouter()


@router.get("/kpis")
def kpis():
    """Live numbers for the doc's MVP Success Criteria: response time,
    citation coverage, grounded-answer rate, and document/query volume."""
    return get_kpis()


@router.get("/queries")
def queries(limit: int = 20):
    """Recent entries from the query audit log (question, answer,
    intent, groundedness, response time, timestamp)."""
    return {"queries": recent_queries(limit)}

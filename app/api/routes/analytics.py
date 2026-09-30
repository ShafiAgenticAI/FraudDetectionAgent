from fastapi import APIRouter, HTTPException
from app.db import get_kpis, recent_queries, recent_ragas_evaluations, record_ragas_evaluation

router = APIRouter()

@router.get("/kpis")
def kpis():
    return get_kpis()

@router.get("/queries")
def queries(limit: int = 20):
    return {"queries": recent_queries(limit)}

@router.get("/ragas")
def ragas_history(limit: int = 10):
    return {"evaluations": recent_ragas_evaluations(limit)}

@router.post("/ragas/run")
def run_ragas():
    """Run the offline golden-set evaluation. This is intentionally a manual
    action from Analytics rather than part of every production chat request."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    script = root / "tests" / "eval_ragas.py"
    try:
        proc = subprocess.run([sys.executable, str(script)], cwd=str(root), capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Ragas evaluation exceeded 30 minutes")
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=proc.stderr[-4000:] or proc.stdout[-4000:])
    # The script prints a final JSON payload.
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1]) if lines else {}
    record_ragas_evaluation(payload.get("timestamp", "unknown"), int(payload.get("golden_questions", 0)), payload.get("metrics", {}), float(payload.get("citation_page_hit_rate", 0)))
    return payload

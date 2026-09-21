from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas.documents import IngestRequest
from app.config import DOCUMENT_DIRECTORY, MAX_UPLOAD_MB
from app.services.ingestion_service import ingest_pdf

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    content = await file.read()

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds {MAX_UPLOAD_MB} MB limit.",
        )

    safe_name = Path(file.filename).name
    destination = DOCUMENT_DIRECTORY / safe_name
    destination.write_bytes(content)

    return {
        "filename": safe_name,
        "size_bytes": len(content),
        "message": "PDF uploaded successfully.",
    }


@router.post("/ingest")
def ingest_document(request: IngestRequest):
    try:
        return ingest_pdf(request.filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

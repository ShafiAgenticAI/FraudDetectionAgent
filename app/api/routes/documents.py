from pathlib import Path
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas.documents import IngestRequest
from app.config import DOCUMENT_DIRECTORY, MAX_UPLOAD_MB
from app.db import list_documents, record_upload
from app.services.ingestion_service import ingest_pdf, remove_document

logger = logging.getLogger("compliance_copilot.documents")

router = APIRouter()


@router.get("/")
def list_indexed_documents():
    """The document registry: every uploaded/indexed file with its
    status, chunk count, and dates (the doc's "Simple Data Model"
    Documents table)."""
    return {"documents": list_documents()}


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

    record_upload(safe_name)

    logger.info("UPLOAD complete | file=%s | size_bytes=%d", safe_name, len(content))

    return {
        "filename": safe_name,
        "size_bytes": len(content),
        "message": "PDF uploaded successfully.",
    }


@router.post("/ingest")
def ingest_document(request: IngestRequest):
    try:
        return ingest_pdf(request.filename, force=request.force)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/{filename}")
def delete_document(filename: str):
    """Remove a document's stored PDF and its indexed chunks so stale or
    retired content can never be queried, summarized, or cited again."""
    try:
        return remove_document(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

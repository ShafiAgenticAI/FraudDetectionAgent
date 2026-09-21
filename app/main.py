from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.analysis import router as analysis_router
from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router

app = FastAPI(
    title="Regulatory Compliance Copilot",
    description=(
        "Local MVP for regulatory PDF ingestion, grounded Q&A, "
        "citations, summaries, and compliance-obligation extraction."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router, prefix="/api/documents", tags=["Documents"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(analysis_router, prefix="/api/analysis", tags=["Analysis"])


@app.get("/")
def root():
    return {
        "application": "Regulatory Compliance Copilot",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}

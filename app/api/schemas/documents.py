from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    filename: str = Field(min_length=1)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class AnalysisRequest(BaseModel):
    filename: str = Field(min_length=1)

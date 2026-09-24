from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    filename: str = Field(min_length=1)
    # Re-embed even if the file's content hash matches the last indexed
    # version (data-freshness / incremental-ingestion support).
    force: bool = False


class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    # If set, the server looks up / updates this session's own cached
    # history (last MAX_SESSION_TURNS exchanges) instead of requiring the
    # caller to resend the full transcript every request.
    session_id: str | None = None
    # Explicit history always overrides the session cache -- useful for
    # stateless API callers that manage their own conversation state.
    history: list[ChatTurn] | None = None


class AnalysisRequest(BaseModel):
    filename: str = Field(min_length=1)

"""Versioned HTTP schemas for the HiAgent/ReMe boundary."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


API_VERSION = "v1"
SERVICE_NAME = "reme-v3-hiagent-api"
UPSTREAM_COMMIT = "2f37a159b72a04ac1885a7db7f1a663a833e7791"


class ErrorDetail(BaseModel):
    """Machine-readable API error."""

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Common error envelope returned by all HiAgent endpoints."""

    status: Literal["error"] = "error"
    api_version: str = API_VERSION
    error: ErrorDetail


class ComponentHealth(BaseModel):
    """Readiness result for one required service component."""

    ready: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    """Readiness response for the ReMe API and its dependencies."""

    status: Literal["ok", "error"]
    service: str = SERVICE_NAME
    api_version: str = API_VERSION
    upstream_commit: str = UPSTREAM_COMMIT
    llm: ComponentHealth
    embedding: ComponentHealth
    vector_store: ComponentHealth
    llm_ready: bool = False
    embedding_ready: bool = False
    vector_store_ready: bool = False

    @model_validator(mode="after")
    def synchronize_readiness_flags(self):
        self.llm_ready = self.llm.ready
        self.embedding_ready = self.embedding.ready
        self.vector_store_ready = self.vector_store.ready
        return self


class RetrieveRequest(BaseModel):
    """Request contract reserved for phase A1."""

    workspace_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    rerank: bool = False
    rewrite: bool = False
    max_context_chars: int = Field(default=3000, ge=0)

    @field_validator("workspace_id", "query")
    @classmethod
    def strip_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class RetrievedMemory(BaseModel):
    """One task memory returned and fully exposed in the prompt."""

    memory_id: str
    when_to_use: str
    content: str
    validation_score: float | None = None
    retrieval_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveDiagnostics(BaseModel):
    """Non-semantic retrieval diagnostics for experiment tracing."""

    candidate_count: int = 0
    returned_count: int = 0
    truncated_count: int = 0
    skipped_count: int = 0
    reranked: bool = False
    rewritten: bool = False


class RetrieveResponse(BaseModel):
    """Read-only phase-A1 retrieval response."""

    retrieval_id: str | None = None
    memory_prompt: str = ""
    memories: list[RetrievedMemory] = Field(default_factory=list)
    diagnostics: RetrieveDiagnostics = Field(default_factory=RetrieveDiagnostics)


class TrajectoryMessage(BaseModel):
    """One normalized message in a task trajectory."""

    role: Literal["user", "assistant", "system"]
    content: str


class TrialTrajectory(BaseModel):
    """Normalized trajectory submitted at the end of a trial."""

    trajectory_id: str = Field(min_length=1)
    messages: list[TrajectoryMessage]
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrialOutcome(BaseModel):
    """HiAgent's authoritative task outcome."""

    success: bool
    score: float | None = None
    progress_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class FinishTrialRequest(BaseModel):
    """Request contract reserved for phase A2 and phase C."""

    workspace_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    retrieval_id: str | None = None
    trajectory: TrialTrajectory
    outcome: TrialOutcome


class NotImplementedResponse(ErrorResponse):
    """Explicit response used while A1/A2 business logic is not installed."""

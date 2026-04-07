"""
api/schemas.py

Pydantic request/response schemas for all FastAPI endpoints.

TEACHING NOTE:
Defining schemas separately from route handlers means:
1. The Streamlit dashboard can import schemas for type safety
2. FastAPI auto-generates OpenAPI docs from these schemas
3. Adding a new field to a response is a one-line change here,
   not a search-and-replace across route handlers

PROD SCALE:
Add versioned schemas (V1EvaluationRequest, V2EvaluationRequest)
to support API version upgrades without breaking existing clients.
"""

from typing import Optional

from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    """Request body for POST /evaluate"""
    chunking_strategy: str = Field(
        default="recursive",
        description="One of: fixed, recursive, semantic, hierarchical"
    )
    top_k: int = Field(default=5, ge=1, le=20)
    num_papers: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of papers to load from data/papers/"
    )
    paper_ids: Optional[list[str]] = Field(
        default=None,
        description="Specific paper IDs to load. If None, loads first num_papers."
    )
    run_ragas: bool = True
    run_deepeval: bool = True
    run_abstention: bool = True
    run_redteam: bool = False
    notes: str = ""


class MetricScores(BaseModel):
    """RAGAS metric scores for a single run."""
    faithfulness: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_relevance: Optional[float] = None
    answer_correctness: Optional[float] = None


class DeepEvalScores(BaseModel):
    """DeepEval pass rates for a single run."""
    faithfulness_pass_rate: Optional[float] = None
    answer_relevancy_pass_rate: Optional[float] = None
    completeness_pass_rate: Optional[float] = None


class EvaluationResponse(BaseModel):
    """Response body for POST /evaluate"""
    run_id: str
    run_name: str
    status: str = "completed"
    ragas: Optional[MetricScores] = None
    deepeval: Optional[DeepEvalScores] = None
    abstention_accuracy: Optional[float] = None
    redteam_failure_rate: Optional[float] = None
    mlflow_url: str
    message: str = ""


class RunSummary(BaseModel):
    """Summary of a single MLflow run for GET /runs"""
    run_id: str
    run_name: str
    status: str
    chunking_strategy: Optional[str] = None
    llm_model: Optional[str] = None
    top_k: Optional[int] = None
    ragas_faithfulness: Optional[float] = None
    ragas_context_precision: Optional[float] = None
    abstention_accuracy: Optional[float] = None
    start_time: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body for GET /health"""
    status: str = "ok"
    mlflow_uri: str
    experiment_name: str
    papers_available: int
    index_exists: bool

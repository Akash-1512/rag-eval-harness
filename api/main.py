"""
api/main.py

FastAPI application — exposes evaluation pipeline over HTTP.

Endpoints:
  GET  /health          — health check, system status
  POST /evaluate        — trigger a full evaluation run
  GET  /runs            — list all MLflow runs
  GET  /runs/{run_id}   — get details of a specific run

TEACHING NOTE:
Why FastAPI over Flask?
1. Automatic OpenAPI docs at /docs — no extra work
2. Pydantic validation built in — request/response schemas enforced
3. Async support — evaluation runs can be made async later
4. Type hints throughout — IDE autocomplete, fewer bugs

PROD SCALE (20,000 docs / 800K pages):
# Run as async background task — evaluation takes minutes, not seconds
# from fastapi import BackgroundTasks
# @app.post("/evaluate")
# async def evaluate(request: EvaluationRequest, background_tasks: BackgroundTasks):
#     background_tasks.add_task(run_evaluation, request)
#     return {"status": "queued", "run_id": run_id}
# Client polls GET /runs/{run_id} for status updates
"""

import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.schemas import (
    DeepEvalScores,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    MetricScores,
    RunSummary,
)

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-eval-harness")
PAPERS_DIR = Path(__file__).parent.parent / "data" / "papers"
INDEX_DIR = Path(__file__).parent.parent / "data" / "indexes"

app = FastAPI(
    title="RAG Eval Harness API",
    description=(
        "REST API for the RAG Evaluation and Red-Teaming Platform. "
        "Trigger evaluation runs, retrieve results, compare across configurations."
    ),
    version="0.9.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow Streamlit dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    Returns system status: MLflow connectivity, paper availability, index status.
    Called by Streamlit dashboard on startup to verify backend is ready.
    """
    papers_available = len(list(PAPERS_DIR.glob("*.pdf"))) if PAPERS_DIR.exists() else 0
    index_exists = INDEX_DIR.exists() and any(INDEX_DIR.iterdir()) if INDEX_DIR.exists() else False

    return HealthResponse(
        status="ok",
        mlflow_uri=MLFLOW_TRACKING_URI,
        experiment_name=EXPERIMENT_NAME,
        papers_available=papers_available,
        index_exists=index_exists,
    )


@app.post("/evaluate", response_model=EvaluationResponse)
def run_evaluation(request: EvaluationRequest):
    """
    Trigger a full evaluation run.

    Loads papers, builds FAISS index, runs RAG pipeline, evaluates with
    RAGAS + DeepEval + abstention accuracy, logs everything to MLflow.

    Returns run_id and all metric scores.

    DEMO: synchronous — blocks until complete (~10-15 minutes for full run)
    PROD: make async with BackgroundTasks — return run_id immediately,
    client polls GET /runs/{run_id} for status
    """
    logger.info(f"Evaluation request: {request.model_dump()}")

    try:
        from api.rag_pipeline import run_rag_batch
        from data.qa_pairs.loader import load_qa_pairs
        from ingestion.chunker import ChunkingStrategy, chunk_documents
        from ingestion.document_loader import load_all_papers
        from retrieval.vector_store import build_index
        from tracking.experiment import EvaluationRunConfig, log_evaluation_run

        # Load papers
        docs = load_all_papers(paper_ids=request.paper_ids)
        if not docs:
            raise HTTPException(status_code=404, detail="No papers found in data/papers/")

        # Chunk
        strategy = ChunkingStrategy(request.chunking_strategy)
        chunks = chunk_documents(docs, strategy=strategy)

        # Build index
        index_name = f"{request.chunking_strategy}_api"
        vector_store = build_index(chunks, strategy_name=index_name)

        # Load QA pairs
        qa_pairs = load_qa_pairs()
        if not qa_pairs:
            raise HTTPException(status_code=404, detail="No Q&A pairs found in data/qa_pairs/")

        # Generate RAG outputs
        rag_outputs = run_rag_batch(
            qa_pairs=qa_pairs,
            vector_store=vector_store,
            top_k=request.top_k,
            chunking_strategy=request.chunking_strategy,
        )

        ragas_result = None
        deepeval_result = None
        abstention_result = None

        # Run evaluations
        if request.run_ragas:
            from evaluation.ragas_pipeline.evaluator import run_ragas_evaluation
            ragas_result = run_ragas_evaluation(rag_outputs)

        if request.run_deepeval:
            from evaluation.deepeval_tests.test_suite import run_deepeval_assertions
            deepeval_result = run_deepeval_assertions(rag_outputs)

        if request.run_abstention:
            from evaluation.custom_metrics.abstention_accuracy import (
                OUT_OF_SCOPE_QUESTIONS,
                evaluate_abstention_accuracy,
            )
            oos_qa = [{"question": q, "ground_truth": ""} for q in OUT_OF_SCOPE_QUESTIONS[:3]]
            oos_outputs = run_rag_batch(
                qa_pairs=oos_qa,
                vector_store=vector_store,
                top_k=request.top_k,
                chunking_strategy=request.chunking_strategy,
            )
            oos_dicts = [{"question": o.question, "answer": o.answer} for o in oos_outputs]
            abstention_result = evaluate_abstention_accuracy(oos_dicts)

        # Log to MLflow
        config = EvaluationRunConfig(
            chunking_strategy=request.chunking_strategy,
            top_k=request.top_k,
            llm_model=os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
            num_papers=len(set(d.metadata.get("paper_id") for d in docs)),
            num_qa_pairs=len(qa_pairs),
            run_ragas=request.run_ragas,
            run_deepeval=request.run_deepeval,
            run_abstention=request.run_abstention,
            run_redteam=request.run_redteam,
            notes=request.notes,
        )

        run_name = f"{request.chunking_strategy}_top{request.top_k}_api"
        run_id = log_evaluation_run(
            config=config,
            ragas_result=ragas_result,
            deepeval_result=deepeval_result,
            abstention_result=abstention_result,
            run_name=run_name,
        )

        # Build response
        ragas_scores = None
        if ragas_result:
            ragas_scores = MetricScores(
                faithfulness=ragas_result.faithfulness or None,
                context_precision=ragas_result.context_precision or None,
                context_recall=ragas_result.context_recall or None,
                answer_relevance=ragas_result.answer_relevance or None,
                answer_correctness=ragas_result.answer_correctness or None,
            )

        deepeval_scores = None
        if deepeval_result:
            deepeval_scores = DeepEvalScores(
                faithfulness_pass_rate=deepeval_result.faithfulness_pass_rate,
                answer_relevancy_pass_rate=deepeval_result.answer_relevancy_pass_rate,
                completeness_pass_rate=deepeval_result.completeness_pass_rate,
            )

        return EvaluationResponse(
            run_id=run_id,
            run_name=run_name,
            status="completed",
            ragas=ragas_scores,
            deepeval=deepeval_scores,
            abstention_accuracy=abstention_result.score if abstention_result else None,
            mlflow_url=f"{MLFLOW_TRACKING_URI}/#/experiments",
            message=f"Evaluation complete. View at {MLFLOW_TRACKING_URI}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs", response_model=list[RunSummary])
def list_runs(limit: int = 20):
    """
    List recent MLflow evaluation runs.
    Called by Streamlit dashboard to populate the runs table.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        experiments = client.search_experiments(
            filter_string=f"name = '{EXPERIMENT_NAME}'"
        )
        if not experiments:
            return []

        experiment_id = experiments[0].experiment_id
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=limit,
            order_by=["start_time DESC"],
        )

        summaries = []
        for run in runs:
            params = run.data.params
            metrics = run.data.metrics
            summaries.append(RunSummary(
                run_id=run.info.run_id,
                run_name=run.info.run_name or run.info.run_id[:8],
                status=run.info.status,
                chunking_strategy=params.get("chunking_strategy"),
                llm_model=params.get("llm_model"),
                top_k=int(params["top_k"]) if "top_k" in params else None,
                ragas_faithfulness=metrics.get("ragas_faithfulness"),
                ragas_context_precision=metrics.get("ragas_context_precision"),
                abstention_accuracy=metrics.get("abstention_accuracy"),
                start_time=str(run.info.start_time),
            ))

        return summaries

    except Exception as e:
        logger.error(f"Failed to list runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs/{run_id}", response_model=dict)
def get_run(run_id: str):
    """
    Get full details of a specific MLflow run.
    Called by Streamlit dashboard when user clicks on a run.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)

        return {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "start_time": str(run.info.start_time),
        }

    except Exception as e:
        logger.error(f"Failed to get run {run_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

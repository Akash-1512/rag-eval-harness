"""
tracking/experiment.py

MLflow experiment tracking for all evaluation runs.

WHAT: Logs every evaluation run as an MLflow experiment with:
- Parameters: chunking strategy, top_k, model name, num questions
- Metrics: all RAGAS scores, DeepEval pass rates, abstention accuracy,
  red-team failure rates
- Artifacts: per-question DataFrames as CSV, attack results as JSON

WHY MLflow:
Without tracking, you run evaluations and see numbers in a terminal.
With MLflow, you answer production questions:
  "Which chunking strategy produces best faithfulness?"
  "Did the 70b model improve answer correctness over 8b?"
  "Is red-team failure rate trending up or down across runs?"
"""

import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import mlflow
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-eval-harness")


@dataclass
class EvaluationRunConfig:
    """
    Configuration for a single evaluation run.
    Logged as MLflow parameters — every config difference creates
    a new comparable data point.
    """

    chunking_strategy: str = "recursive"
    chunk_size: int = 2048
    chunk_overlap: int = 256
    top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama-3.3-70b-versatile"
    num_papers: int = 3
    num_qa_pairs: int = 3
    run_ragas: bool = True
    run_deepeval: bool = True
    run_abstention: bool = True
    run_redteam: bool = False
    notes: str = ""


@contextmanager
def _tmp_csv(df):
    """Write DataFrame to a temp CSV, yield path, then delete."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        tmp_path = f.name
        df.to_csv(f, index=False)
    try:
        yield tmp_path
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@contextmanager
def _tmp_json(data):
    """Write dict/list to a temp JSON, yield path, then delete."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        tmp_path = f.name
        json.dump(data, f, indent=2)
    try:
        yield tmp_path
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def setup_mlflow() -> None:
    """Configure MLflow tracking URI and experiment."""
    # FIXME: concurrent calls may race on experiment creation — needs lock for multi-process use

    # ──────────────────────────────────────────────────────────────
    # LOCAL DEMO — MLflow local server (zero cost)
    # Run in a separate terminal: mlflow ui --port 5000
    # Data persists in ./mlruns/ — not shared across machines
    # ──────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ──────────────────────────────────────────────────────────────
    # [PRODUCTION] Azure ML MLflow tracking — uncomment
    # Requires: AZURE_ML_TRACKING_URI in .env
    # Get URI from: Azure ML workspace -> MLflow tracking URI
    # Persists across machines, searchable, enterprise audit log
    # ──────────────────────────────────────────────────────────────
    # mlflow.set_tracking_uri(os.getenv("AZURE_ML_TRACKING_URI"))
    # mlflow.set_experiment("rag-eval-harness-production")

    logger.info(
        f"MLflow tracking: {MLFLOW_TRACKING_URI} | experiment: {EXPERIMENT_NAME}"
    )


def log_evaluation_run(
    config: EvaluationRunConfig,
    ragas_result=None,
    deepeval_result=None,
    abstention_result=None,
    redteam_result=None,
    run_name: Optional[str] = None,
) -> str:
    """
    Log a complete evaluation run to MLflow.

    Args:
        config: Run configuration (logged as parameters)
        ragas_result: RAGEvaluationResult from M5
        deepeval_result: DeepEvalResult from M6
        abstention_result: AbstentionAccuracyResult from M6
        redteam_result: RedTeamResult from M7
        run_name: Human-readable run name for MLflow UI

    Returns:
        MLflow run_id string
    """
    setup_mlflow()

    if run_name is None:
        run_name = (
            f"{config.chunking_strategy}_"
            f"{config.llm_model.split('/')[-1]}_"
            f"top{config.top_k}"
        )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_name} (id={run_id[:8]}...)")

        # Parameters
        params = {
            "chunking_strategy": config.chunking_strategy,
            "chunk_size": str(config.chunk_size),
            "chunk_overlap": str(config.chunk_overlap),
            "top_k": str(config.top_k),
            "embedding_model": config.embedding_model,
            "llm_model": config.llm_model,
            "num_papers": str(config.num_papers),
            "num_qa_pairs": str(config.num_qa_pairs),
            "notes": config.notes,
        }
        mlflow.log_params(params)
        logger.info(f"  Logged {len(params)} parameters")

        # RAGAS metrics
        if ragas_result is not None:
            ragas_metrics = {
                "ragas_faithfulness": ragas_result.faithfulness,
                "ragas_context_precision": ragas_result.context_precision,
                "ragas_context_recall": ragas_result.context_recall,
                "ragas_answer_relevance": ragas_result.answer_relevance,
                "ragas_answer_correctness": ragas_result.answer_correctness,
            }
            # Filter NaN — MLflow cannot log NaN
            ragas_metrics = {
                k: v
                for k, v in ragas_metrics.items()
                if v is not None and v == v  # NaN != NaN
            }
            mlflow.log_metrics(ragas_metrics)
            logger.info(f"  Logged RAGAS metrics: {ragas_metrics}")

            if (
                ragas_result.per_question_df is not None
                and not ragas_result.per_question_df.empty
            ):
                with _tmp_csv(ragas_result.per_question_df) as path:
                    mlflow.log_artifact(path, artifact_path="ragas")
                logger.info("  Logged RAGAS per-question CSV artifact")

        # DeepEval metrics
        if deepeval_result is not None:
            deepeval_metrics = deepeval_result.to_dict()
            mlflow.log_metrics(deepeval_metrics)
            logger.info(f"  Logged DeepEval metrics: {deepeval_metrics}")

            if deepeval_result.test_results:
                with _tmp_json(deepeval_result.test_results) as path:
                    mlflow.log_artifact(path, artifact_path="deepeval")
                logger.info("  Logged DeepEval per-question JSON artifact")

        # Abstention accuracy
        if abstention_result is not None:
            mlflow.log_metrics(
                {
                    "abstention_accuracy": abstention_result.score,
                    "abstention_correct": float(abstention_result.num_correct),
                    "abstention_total": float(abstention_result.num_total),
                }
            )
            logger.info(
                f"  Logged abstention accuracy: {abstention_result.score:.3f}"
            )

        # Red-team metrics
        if redteam_result is not None:
            redteam_metrics = redteam_result.to_dict()
            mlflow.log_metrics(redteam_metrics)
            logger.info(
                f"  Logged red-team metrics: "
                f"failure_rate={redteam_result.failure_rate:.3f}"
            )
            if redteam_result.attack_results:
                with _tmp_json(redteam_result.attack_results) as path:
                    mlflow.log_artifact(path, artifact_path="redteam")
                logger.info("  Logged red-team attack results JSON artifact")

        logger.success(
            f"MLflow run complete: {run_name} | "
            f"View at {MLFLOW_TRACKING_URI}/#/experiments"
        )
        return run_id
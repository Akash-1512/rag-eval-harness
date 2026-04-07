"""evaluation.ragas_pipeline package."""

from evaluation.ragas_pipeline.evaluator import (
    RAGEvaluationResult,
    run_ragas_evaluation,
)

__all__ = ["run_ragas_evaluation", "RAGEvaluationResult"]

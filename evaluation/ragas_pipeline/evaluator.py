"""
evaluation/ragas_pipeline/evaluator.py

RAGAS evaluation pipeline — scores all 5 core metrics against
real RAG pipeline outputs.

METRICS IMPLEMENTED:
1. Faithfulness        — hallucination detection
2. Context Precision   — retrieval ranking quality
3. Context Recall      — retrieval coverage
4. Answer Relevance    — response relevance to question
5. Answer Correctness  — factual accuracy vs ground truth

TEACHING NOTE:
RAGAS uses an LLM-as-Judge pattern for most metrics. It makes
multiple LLM calls per question per metric to decompose answers
into atomic claims, then verify each claim against context.

At 50 questions x 5 metrics x ~3 LLM calls each = ~750 LLM calls
per full evaluation run. On Groq free tier (14,400 req/day) this
is well within limits. On Azure OpenAI this would cost ~$0.50-1.00.

PROD SCALE (20,000 docs / 800K pages):
# Use stratified sampling — evaluate 10% of questions per run
# Rotate sample each run for full coverage over 10 runs
# Track sample indices in MLflow to avoid duplicate evaluation
# Use async batch evaluation to parallelise LLM calls
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from loguru import logger
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from api.rag_pipeline import RAGOutput

load_dotenv()


# ── Result schema ─────────────────────────────────────────────────────────────

@dataclass
class RAGEvaluationResult:
    """
    Structured result from a RAGAS evaluation run.
    Consumed by MLflow (M8) and Streamlit dashboard (M10).

    TEACHING NOTE:
    We store both aggregate scores (for MLflow radar charts) and
    per-question scores (for Streamlit drill-down table). Most
    evaluation frameworks only give you aggregates — storing per-question
    results is what lets you answer "which specific questions failed
    faithfulness and why?"
    """
    # Aggregate scores (0.0 to 1.0)
    faithfulness: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_relevance: float = 0.0
    answer_correctness: float = 0.0

    # Per-question breakdown (DataFrame for easy display)
    per_question_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Run metadata
    num_questions: int = 0
    chunking_strategy: str = ""
    top_k: int = 5
    model: str = ""
    run_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Flat dict for MLflow logging."""
        return {
            "faithfulness": self.faithfulness,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_relevance": self.answer_relevance,
            "answer_correctness": self.answer_correctness,
            "num_questions": self.num_questions,
            "chunking_strategy": self.chunking_strategy,
            "top_k": self.top_k,
            "model": self.model,
        }

    def summary(self) -> str:
        """Human-readable summary for logging."""
        return (
            f"RAGAS Evaluation Results ({self.num_questions} questions)\n"
            f"  Faithfulness:       {self.faithfulness:.3f}\n"
            f"  Context Precision:  {self.context_precision:.3f}\n"
            f"  Context Recall:     {self.context_recall:.3f}\n"
            f"  Answer Relevance:   {self.answer_relevance:.3f}\n"
            f"  Answer Correctness: {self.answer_correctness:.3f}\n"
            f"  Strategy: {self.chunking_strategy} | top_k: {self.top_k}"
        )


# ── LLM + Embedder setup for RAGAS ────────────────────────────────────────────

class SingleGenerationGroq(ChatGroq):
    """
    Groq wrapper that caps n=1 for models that reject n>1.
    RAGAS answer_relevancy sends n=3 by default to generate
    multiple hypothetical questions. qwen3-32b rejects this.
    This subclass forces n=1 on every call.

    PROD (Azure OpenAI): supports n>1 natively, no wrapper needed.
    """
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("n", None)  # remove n parameter entirely
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("n", None)
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _get_ragas_llm() -> LangchainLLMWrapper:
    """
    Returns RAGAS-wrapped LLM for metric computation.

    RAGAS uses this LLM to:
    - Decompose answers into atomic claims (faithfulness)
    - Generate hypothetical questions from answers (answer relevance)
    - Compare answer claims to ground truth (answer correctness)

    DEMO (zero budget): Groq llama-3.1-70b-versatile
    PROD (paid):
    # from langchain_openai import AzureChatOpenAI
    # llm = AzureChatOpenAI(
    #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    #     temperature=0,
    # )
    """
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL_NAME", "llama-3.1-70b-versatile")

    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")

    # DEMO: SingleGenerationGroq caps n=1 for models that reject n>1
    # PROD (Azure OpenAI): use standard ChatOpenAI — supports n>1 natively
    llm = SingleGenerationGroq(
        api_key=api_key,
        model=model,
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def _get_ragas_embedder() -> LangchainEmbeddingsWrapper:
    """
    Returns RAGAS-wrapped embedder for answer relevance metric.
    Answer relevance uses embeddings to compute semantic similarity
    between the original question and LLM-generated reverse questions.

    DEMO (zero budget): local sentence-transformers
    PROD (paid):
    # from langchain_openai import AzureOpenAIEmbeddings
    # embeddings = AzureOpenAIEmbeddings(
    #     azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    # )
    """
    from retrieval.embedder import LocalEmbedder
    embedder = LocalEmbedder()
    return LangchainEmbeddingsWrapper(embedder)


# ── Core evaluation function ───────────────────────────────────────────────────

def run_ragas_evaluation(
    rag_outputs: list[RAGOutput],
    metrics: Optional[list] = None,
    run_id: Optional[str] = None,
) -> RAGEvaluationResult:
    """
    Run RAGAS evaluation on a list of RAG pipeline outputs.

    Args:
        rag_outputs: List of RAGOutput from run_rag_batch()
        metrics: Optional subset of metrics to run.
                 Defaults to all 5 core metrics.
        run_id: Optional MLflow run ID for tracking

    Returns:
        RAGEvaluationResult with aggregate and per-question scores

    Raises:
        ValueError: If rag_outputs is empty or missing ground_truth
                    for metrics that require it
    """
    if not rag_outputs:
        raise ValueError("rag_outputs list is empty — run RAG pipeline first")

    # Filter out any errored outputs
    valid_outputs = [o for o in rag_outputs if "error" not in o.metadata]
    if not valid_outputs:
        raise ValueError("All RAG outputs contain errors — check RAG pipeline")

    if len(valid_outputs) < len(rag_outputs):
        logger.warning(
            f"Skipping {len(rag_outputs) - len(valid_outputs)} errored outputs"
        )

    # Default to all 5 metrics
    if metrics is None:
        metrics = [
            faithfulness,
            context_precision,
            context_recall,
            answer_relevancy,
            answer_correctness,
        ]

    # Check ground truth availability for metrics that need it
    needs_ground_truth = [context_recall, answer_correctness]
    if any(m in metrics for m in needs_ground_truth):
        missing_gt = [o for o in valid_outputs if not o.ground_truth]
        if missing_gt:
            logger.warning(
                f"{len(missing_gt)} outputs missing ground_truth. "
                f"context_recall and answer_correctness will be unreliable."
            )

    logger.info(
        f"Running RAGAS evaluation: {len(valid_outputs)} questions, "
        f"{len(metrics)} metrics"
    )

    # Build RAGAS dataset
    # RAGAS 0.2.x expects a HuggingFace Dataset with these exact column names
    dataset_dict = {
        "question": [o.question for o in valid_outputs],
        "answer": [o.answer for o in valid_outputs],
        "contexts": [o.contexts for o in valid_outputs],
        "ground_truth": [o.ground_truth for o in valid_outputs],
    }
    dataset = Dataset.from_dict(dataset_dict)

    logger.info("Dataset prepared — starting RAGAS scoring...")

    # Configure LLM and embedder for RAGAS
    ragas_llm = _get_ragas_llm()
    ragas_embedder = _get_ragas_embedder()

    # Inject LLM and embedder into each metric
    for metric in metrics:
        if hasattr(metric, "llm"):
            metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embedder

    # Run evaluation
    # RunConfig controls concurrency and timeout — critical for Groq free tier
    # DEMO: max_workers=1 prevents rate limit timeouts on Groq free tier
    # PROD: increase to max_workers=4 with Azure OpenAI which handles concurrency
    from ragas import RunConfig
    run_config = RunConfig(
        max_workers=1,      # sequential to avoid Groq rate limits
        timeout=120,        # 2 min per metric call
        max_retries=3,
    )
    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            run_config=run_config,
        )
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        raise

    # Extract scores
    scores_df = results.to_pandas()

    logger.info("RAGAS scoring complete")

    # Build result object with safe .get() for missing metrics
    def safe_mean(col: str) -> float:
        if col in scores_df.columns:
            return float(scores_df[col].mean())
        return 0.0

    # Extract run metadata from first valid output
    meta = valid_outputs[0].metadata

    result = RAGEvaluationResult(
        faithfulness=safe_mean("faithfulness"),
        context_precision=safe_mean("context_precision"),
        context_recall=safe_mean("context_recall"),
        answer_relevance=safe_mean("answer_relevancy"),
        answer_correctness=safe_mean("answer_correctness"),
        per_question_df=scores_df,
        num_questions=len(valid_outputs),
        chunking_strategy=meta.get("chunking_strategy", "unknown"),
        top_k=meta.get("top_k", 5),
        model=meta.get("model", "unknown"),
        run_id=run_id,
    )

    logger.success(f"\n{result.summary()}")
    return result






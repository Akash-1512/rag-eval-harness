"""
tracking/smoke_test.py

Runs a full evaluation pipeline and logs everything to MLflow.
First end-to-end run: ingest -> retrieve -> RAG -> RAGAS ->
DeepEval -> abstention -> MLflow.

Usage:
    # Terminal 1: start MLflow server
    mlflow ui --port 5000

    # Terminal 2: run this
    python -m tracking.smoke_test

    # Then open http://127.0.0.1:5000 to see logged results
"""

from loguru import logger
from dotenv import load_dotenv

from ingestion.document_loader import load_all_papers
from ingestion.chunker import ChunkingStrategy, chunk_documents
from retrieval.vector_store import build_index
from api.rag_pipeline import run_rag_batch
from evaluation.ragas_pipeline.evaluator import run_ragas_evaluation
from evaluation.deepeval_tests.test_suite import run_deepeval_assertions
from evaluation.custom_metrics.abstention_accuracy import (
    evaluate_abstention_accuracy,
    OUT_OF_SCOPE_QUESTIONS,
)
from tracking.experiment import log_evaluation_run, EvaluationRunConfig

load_dotenv()

QA_PAIRS = [
    {
        "question": "How many attention heads does the Transformer base model use and what is the dimensionality of each head?",
        "ground_truth": "The base Transformer uses 8 attention heads. The model dimensionality is 512 so each head has a dimensionality of 64.",
    },
    {
        "question": "What are the two RAG formulations introduced in the original RAG paper?",
        "ground_truth": "RAG-Sequence conditions the full answer on a single retrieved document and RAG-Token can draw from different documents for each generated token.",
    },
    {
        "question": "According to the RAGAS paper what does the faithfulness metric specifically measure?",
        "ground_truth": "Faithfulness measures whether all claims in the answer can be inferred from the context. It is computed as the ratio of supported claims to total claims.",
    },
]

ABSTENTION_QA = [
    {"question": q, "ground_truth": ""}
    for q in OUT_OF_SCOPE_QUESTIONS[:3]
]


def run_smoke_test():
    logger.info("=" * 60)
    logger.info("MLFLOW TRACKING SMOKE TEST")
    logger.info("Make sure: mlflow ui --port 5000 is running")
    logger.info("=" * 60)

    # Build pipeline
    logger.info("\nStep 1: Building pipeline")
    docs = load_all_papers(paper_ids=[
        "01_attention_is_all_you_need",
        "06_rag",
        "08_ragas",
    ])
    chunks = chunk_documents(docs, strategy=ChunkingStrategy.RECURSIVE)
    vector_store = build_index(chunks, strategy_name="recursive_smoke")
    logger.success(f"  {len(chunks)} chunks indexed")

    # Generate RAG outputs
    logger.info("\nStep 2: Generating RAG outputs")
    rag_outputs = run_rag_batch(
        qa_pairs=QA_PAIRS,
        vector_store=vector_store,
        top_k=5,
        chunking_strategy="recursive",
    )
    logger.success(f"  {len(rag_outputs)} outputs generated")

    # Run RAGAS
    logger.info("\nStep 3: Running RAGAS evaluation")
    ragas_result = run_ragas_evaluation(rag_outputs)

    # Run DeepEval
    logger.info("\nStep 4: Running DeepEval assertions")
    deepeval_result = run_deepeval_assertions(rag_outputs, threshold=0.5)

    # Run abstention accuracy
    logger.info("\nStep 5: Running abstention accuracy")
    oos_outputs = run_rag_batch(
        qa_pairs=ABSTENTION_QA,
        vector_store=vector_store,
        top_k=5,
        chunking_strategy="recursive",
    )
    oos_dicts = [{"question": o.question, "answer": o.answer} for o in oos_outputs]
    abstention_result = evaluate_abstention_accuracy(oos_dicts)

    # Log everything to MLflow
    logger.info("\nStep 6: Logging to MLflow")
    import os
    config = EvaluationRunConfig(
        chunking_strategy="recursive",
        chunk_size=2048,
        chunk_overlap=256,
        top_k=5,
        embedding_model="all-MiniLM-L6-v2",
        llm_model=os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
        num_papers=3,
        num_qa_pairs=len(QA_PAIRS),
        run_ragas=True,
        run_deepeval=True,
        run_abstention=True,
        run_redteam=False,
        notes="M8 smoke test — recursive chunking baseline",
    )

    run_id = log_evaluation_run(
        config=config,
        ragas_result=ragas_result,
        deepeval_result=deepeval_result,
        abstention_result=abstention_result,
        run_name="recursive_baseline_smoke",
    )

    logger.info("\n" + "=" * 60)
    logger.success(f"Run logged: {run_id}")
    logger.info("Open http://127.0.0.1:5000 to view results")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_smoke_test()

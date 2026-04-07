"""
evaluation/ragas_pipeline/smoke_test.py

Runs RAGAS evaluation on 3 real Q&A pairs against real RAG outputs.
First real RAGAS metric scores in the project.

Usage:
    python -m evaluation.ragas_pipeline.smoke_test

Expected output:
    - 5 metric scores between 0.0 and 1.0
    - Per-question breakdown DataFrame
    - Matches manual predictions from M4 analysis
"""

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from api.rag_pipeline import run_rag_batch
from evaluation.ragas_pipeline.evaluator import run_ragas_evaluation
from ingestion.chunker import ChunkingStrategy, chunk_documents
from ingestion.document_loader import load_all_papers
from retrieval.vector_store import build_index

load_dotenv()

# 3 Q&A pairs — subset of qa_pairs.csv for quick smoke test
SMOKE_QA = [
    {
        "question": "How many attention heads does the Transformer base model use and what is the dimensionality of each head?",
        "ground_truth": "The base Transformer uses 8 attention heads. The model dimensionality is 512 so each head has a dimensionality of 64 (512/8).",
    },
    {
        "question": "What are the two RAG formulations introduced in the original RAG paper?",
        "ground_truth": "RAG-Sequence conditions the full answer on a single retrieved document and RAG-Token can draw from different documents for each generated token.",
    },
    {
        "question": "According to the RAGAS paper what does the faithfulness metric specifically measure?",
        "ground_truth": "Faithfulness measures whether all claims made in the generated answer can be inferred from the provided context. It is computed as the ratio of claims in the answer that are supported by the context to the total number of claims in the answer.",
    },
]


def run_smoke_test():
    logger.info("=" * 60)
    logger.info("RAGAS EVALUATION SMOKE TEST")
    logger.info("=" * 60)

    # Step 1: Build pipeline
    logger.info("\nStep 1: Loading papers and building index")
    docs = load_all_papers(paper_ids=[
        "01_attention_is_all_you_need",
        "06_rag",
        "08_ragas",
    ])
    chunks = chunk_documents(docs, strategy=ChunkingStrategy.RECURSIVE)
    vector_store = build_index(chunks, strategy_name="recursive_smoke")
    logger.success(f"  Pipeline ready: {len(chunks)} chunks indexed")

    # Step 2: Generate RAG outputs
    logger.info("\nStep 2: Generating RAG outputs for 3 questions")
    rag_outputs = run_rag_batch(
        qa_pairs=SMOKE_QA,
        vector_store=vector_store,
        top_k=5,
        chunking_strategy="recursive",
    )
    logger.success(f"  Generated {len(rag_outputs)} RAG outputs")

    # Step 3: Run RAGAS evaluation
    logger.info("\nStep 3: Running RAGAS evaluation (making LLM judge calls...)")
    logger.info("  This takes 1-3 minutes — RAGAS makes ~15 LLM calls")

    result = run_ragas_evaluation(rag_outputs)

    # Step 4: Display results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.success(f"\n{result.summary()}")

    # Per-question breakdown
    logger.info("\nPer-question scores:")
    display_cols = [
        col for col in [
            "question", "faithfulness", "context_precision",
            "context_recall", "answer_relevancy", "answer_correctness"
        ]
        if col in result.per_question_df.columns
    ]
    if display_cols:
        pd.set_option("display.max_colwidth", 50)
        pd.set_option("display.float_format", "{:.3f}".format)
        print(result.per_question_df[display_cols].to_string(index=False))

    logger.info("\n" + "=" * 60)
    logger.info("Compare these scores to the manual predictions from M4:")
    logger.info("  Q1 (attention heads): Faithfulness ~1.0, Correctness ~0.9")
    logger.info("  Q2 (RAG formulations): Faithfulness ~1.0, Correctness ~0.4")
    logger.info("  Q3 (RAGAS faithfulness): Faithfulness ~0.9, Correctness ~0.7")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_smoke_test()

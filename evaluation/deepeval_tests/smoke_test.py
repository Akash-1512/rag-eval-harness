"""
evaluation/deepeval_tests/smoke_test.py

Runs DeepEval assertions AND abstention accuracy on real RAG outputs.
First time both custom metrics run against actual paper Q&A pairs.

Usage:
    python -m evaluation.deepeval_tests.smoke_test

Expected output:
    - Faithfulness, relevancy, hallucination pass rates
    - Abstention accuracy on 3 out-of-scope questions
    - Per-question pass/fail breakdown
"""

from dotenv import load_dotenv
from loguru import logger

from api.rag_pipeline import run_rag_batch
from evaluation.custom_metrics.abstention_accuracy import (
    OUT_OF_SCOPE_QUESTIONS,
    evaluate_abstention_accuracy,
)
from evaluation.deepeval_tests.test_suite import run_deepeval_assertions
from ingestion.chunker import ChunkingStrategy, chunk_documents
from ingestion.document_loader import load_all_papers
from retrieval.vector_store import build_index

load_dotenv()

# In-scope Q&A for DeepEval assertions
IN_SCOPE_QA = [
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

# 3 out-of-scope questions for abstention accuracy
ABSTENTION_QA = [q for q in OUT_OF_SCOPE_QUESTIONS[:3]]


def run_smoke_test():
    logger.info("=" * 60)
    logger.info("DEEPEVAL + ABSTENTION ACCURACY SMOKE TEST")
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
    logger.success(f"  Pipeline ready: {len(chunks)} chunks")

    # Step 2: Generate in-scope RAG outputs
    logger.info("\nStep 2: Generating in-scope RAG outputs (3 questions)")
    in_scope_outputs = run_rag_batch(
        qa_pairs=IN_SCOPE_QA,
        vector_store=vector_store,
        top_k=5,
        chunking_strategy="recursive",
    )
    logger.success(f"  {len(in_scope_outputs)} outputs generated")

    # Step 3: Run DeepEval assertions
    logger.info("\nStep 3: Running DeepEval G-Eval assertions")
    deepeval_result = run_deepeval_assertions(in_scope_outputs)

    # Step 4: Generate out-of-scope outputs for abstention test
    logger.info("\nStep 4: Generating out-of-scope outputs (3 questions)")
    abstention_qa = [{"question": q, "ground_truth": ""} for q in ABSTENTION_QA]
    oos_outputs = run_rag_batch(
        qa_pairs=abstention_qa,
        vector_store=vector_store,
        top_k=5,
        chunking_strategy="recursive",
    )

    # Step 5: Run abstention accuracy
    logger.info("\nStep 5: Running abstention accuracy evaluation")
    oos_dicts = [
        {"question": o.question, "answer": o.answer}
        for o in oos_outputs
    ]
    abstention_result = evaluate_abstention_accuracy(oos_dicts)

    # Step 6: Summary
    logger.info("\n" + "=" * 60)
    logger.info("COMBINED RESULTS")
    logger.info("=" * 60)
    logger.success(f"\n{deepeval_result.summary()}")
    logger.success(f"\n{abstention_result.summary()}")

    logger.info("\nPer-question DeepEval breakdown:")
    for r in deepeval_result.test_results:
        logger.info(f"\n  Q: {r['question'][:60]}...")
        for metric, data in r.items():
            if metric in ("question", "answer"):
                continue
            status = "PASS" if data["passed"] else "FAIL"
            logger.info(f"    {metric}: {status} ({data['score']:.3f})")

    logger.info("\nAbstention per-question:")
    for r in abstention_result.per_question:
        status = "✓" if r.is_correct else "✗"
        logger.info(f"  {status} {r.question[:60]}...")
        logger.info(f"    → {r.reason}")

    logger.info("\n" + "=" * 60)
    logger.success("Smoke test complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_smoke_test()

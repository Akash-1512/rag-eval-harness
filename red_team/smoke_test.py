"""
red_team/smoke_test.py

Runs the LangGraph red-team agent on 6 attacks (one per type)
against the real RAG pipeline.

Usage:
    python -m red_team.smoke_test
"""

from dotenv import load_dotenv
from loguru import logger

from ingestion.chunker import ChunkingStrategy, chunk_documents
from ingestion.document_loader import load_all_papers
from red_team.agent import run_red_team_agent
from red_team.attack_types import (
    CROSS_DOC_SYNTHESIS_ATTACKS,
    NUMERICAL_PROBE_ATTACKS,
    OUT_OF_SCOPE_ATTACKS,
    PREMISE_INJECTION_ATTACKS,
    TEMPORAL_VERSION_ATTACKS,
    VERSION_CONFUSION_ATTACKS,
)
from retrieval.vector_store import build_index

load_dotenv()

SMOKE_ATTACKS = [
    VERSION_CONFUSION_ATTACKS[0],
    NUMERICAL_PROBE_ATTACKS[0],
    PREMISE_INJECTION_ATTACKS[0],
    CROSS_DOC_SYNTHESIS_ATTACKS[0],
    OUT_OF_SCOPE_ATTACKS[0],
    TEMPORAL_VERSION_ATTACKS[0],
]


def run_smoke_test():
    logger.info("=" * 60)
    logger.info("RED-TEAM AGENT SMOKE TEST")
    logger.info("6 attacks — one per attack type")
    logger.info("=" * 60)

    logger.info("\nStep 1: Loading papers and building index")
    docs = load_all_papers(paper_ids=[
        "01_attention_is_all_you_need",
        "03_gpt3",
        "06_rag",
        "08_ragas",
        "04_llama",
        "05_llama2",
    ])
    chunks = chunk_documents(docs, strategy=ChunkingStrategy.RECURSIVE)
    vector_store = build_index(chunks, strategy_name="recursive_redteam")
    logger.success(f"  Index ready: {len(chunks)} chunks from 6 papers")

    logger.info("\nStep 2: Running LangGraph red-team agent")
    result = run_red_team_agent(
        vector_store=vector_store,
        attacks=SMOKE_ATTACKS,
    )

    logger.info("\n" + "=" * 60)
    logger.info("ATTACK-BY-ATTACK RESULTS")
    logger.info("=" * 60)
    for r in result.attack_results:
        status = "FAIL x" if r["failed"] else "PASS v"
        logger.info(f"\n[{r['attack_type']}] {status}")
        logger.info(f"  Q: {r['question'][:70]}...")
        logger.info(f"  A: {r['response'][:100]}...")
        logger.info(f"  Judge: {r['reason'][:100]}")

    logger.info("\n" + "=" * 60)
    logger.success(f"\n{result.summary()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_smoke_test()

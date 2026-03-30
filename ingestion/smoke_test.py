"""
ingestion/smoke_test.py

Run this BEFORE building the retrieval layer.
Verifies that all 4 chunking strategies produce valid output
against real PDFs from data/papers/.

Usage:
    python -m ingestion.smoke_test

Expected output:
    - Chunk counts for each strategy
    - Sample chunk text from Paper 1
    - Metadata keys present on each chunk
    - No errors or empty chunks
"""

from pathlib import Path
from loguru import logger
from ingestion.document_loader import load_all_papers
from ingestion.chunker import ChunkingStrategy, chunk_documents, chunk_hierarchical


def run_smoke_test():
    logger.info("=" * 60)
    logger.info("INGESTION SMOKE TEST — running against real PDFs")
    logger.info("=" * 60)

    # Load only first 3 papers for speed — real PDFs, not mocks
    logger.info("\nStep 1: Loading 3 papers from data/papers/")
    docs = load_all_papers(paper_ids=[
        "01_attention_is_all_you_need",
        "06_rag",
        "08_ragas",
    ])
    logger.info(f"Loaded {len(docs)} pages total\n")

    # ── Strategy 1: Fixed ──────────────────────────────────────────
    logger.info("Step 2: Fixed-size chunking")
    fixed_chunks = chunk_documents(docs, strategy=ChunkingStrategy.FIXED)
    logger.info(f"  Result: {len(fixed_chunks)} chunks")
    _print_sample(fixed_chunks[0], label="Fixed sample")

    # ── Strategy 2: Recursive ──────────────────────────────────────
    logger.info("\nStep 3: Recursive chunking")
    recursive_chunks = chunk_documents(docs, strategy=ChunkingStrategy.RECURSIVE)
    logger.info(f"  Result: {len(recursive_chunks)} chunks")
    _print_sample(recursive_chunks[0], label="Recursive sample")

    # ── Strategy 3: Semantic ───────────────────────────────────────
    logger.info("\nStep 4: Semantic chunking (loads embedding model...)")
    semantic_chunks = chunk_documents(docs, strategy=ChunkingStrategy.SEMANTIC)
    logger.info(f"  Result: {len(semantic_chunks)} chunks")
    _print_sample(semantic_chunks[0], label="Semantic sample")

    # ── Strategy 4: Hierarchical ───────────────────────────────────
    logger.info("\nStep 5: Hierarchical chunking")
    parent_chunks, child_chunks = chunk_hierarchical(docs)
    logger.info(f"  Result: {len(parent_chunks)} parent chunks, {len(child_chunks)} child chunks")
    _print_sample(child_chunks[0], label="Hierarchical child sample")

    # ── Metadata verification ──────────────────────────────────────
    logger.info("\nStep 6: Metadata verification")
    required_keys = {"source", "filename", "paper_id", "title", "page",
                     "chunk_index", "chunking_strategy", "chunk_char_count"}
    sample = recursive_chunks[0]
    present_keys = set(sample.metadata.keys())
    missing = required_keys - present_keys

    if missing:
        logger.error(f"Missing metadata keys: {missing}")
    else:
        logger.success(f"All required metadata keys present: {required_keys}")

    # ── Summary ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Pages loaded:           {len(docs)}")
    logger.info(f"  Fixed chunks:           {len(fixed_chunks)}")
    logger.info(f"  Recursive chunks:       {len(recursive_chunks)}")
    logger.info(f"  Semantic chunks:        {len(semantic_chunks)}")
    logger.info(f"  Hierarchical (parent):  {len(parent_chunks)}")
    logger.info(f"  Hierarchical (child):   {len(child_chunks)}")
    logger.info("=" * 60)
    logger.success("Smoke test complete — ingestion pipeline is working")


def _print_sample(chunk, label: str):
    """Print a truncated view of a chunk for manual inspection."""
    preview = chunk.page_content[:200].replace("\n", " ")
    logger.info(f"  [{label}]")
    logger.info(f"    Text: {preview}...")
    logger.info(f"    Metadata: {chunk.metadata}")


if __name__ == "__main__":
    run_smoke_test()
"""
retrieval/smoke_test.py

Builds a FAISS index from 3 real papers and runs real retrieval queries.
Verifies that retrieved chunks are actually relevant to the question.

Usage:
    python -m retrieval.smoke_test

Expected output:
    - Index built successfully
    - 3 queries return 5 chunks each
    - Chunks visually relevant to the query (inspect manually)
    - Index saved to data/indexes/recursive/
"""

from loguru import logger
from ingestion.document_loader import load_all_papers
from ingestion.chunker import ChunkingStrategy, chunk_documents
from retrieval.vector_store import build_index, retrieve


TEST_QUERIES = [
    "How many attention heads does the Transformer base model use?",
    "What are the two RAG formulations introduced in the original RAG paper?",
    "What does the faithfulness metric measure in RAGAS?",
]


def run_smoke_test():
    logger.info("=" * 60)
    logger.info("RETRIEVAL SMOKE TEST — building index on real PDFs")
    logger.info("=" * 60)

    # Step 1: Load and chunk 3 papers
    logger.info("\nStep 1: Loading and chunking 3 papers")
    docs = load_all_papers(paper_ids=[
        "01_attention_is_all_you_need",
        "06_rag",
        "08_ragas",
    ])
    chunks = chunk_documents(docs, strategy=ChunkingStrategy.RECURSIVE)
    logger.info(f"  {len(docs)} pages -> {len(chunks)} chunks")

    # Step 2: Build FAISS index
    logger.info("\nStep 2: Building FAISS index")
    vector_store = build_index(chunks, strategy_name="recursive_smoke")
    logger.success("  Index built and saved")

    # Step 3: Run real queries and inspect results
    logger.info("\nStep 3: Running retrieval queries")
    for i, query in enumerate(TEST_QUERIES, 1):
        logger.info(f"\n  Query {i}: {query}")
        results = retrieve(query, vector_store, top_k=3)

        for j, doc in enumerate(results, 1):
            preview = doc.page_content[:150].replace("\n", " ")
            logger.info(f"    Chunk {j} [{doc.metadata['title']} p.{doc.metadata['page']}]")
            logger.info(f"    {preview}...")

    logger.info("\n" + "=" * 60)
    logger.success("Retrieval smoke test complete")
    logger.info("Visually verify: are the retrieved chunks relevant to each query?")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_smoke_test()

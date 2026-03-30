# Retrieval — Notes

## What this folder does
Embeds document chunks into dense vectors and stores them in FAISS.
Exposes a single retrieve() function used by the RAG pipeline.

## Why retrieval quality is the ceiling on all RAGAS metrics
No metric can score higher than the quality of what is retrieved:
- Faithfulness: LLM can only be faithful to what it receives as context
- Context Precision: Directly measures retrieval quality (relevant chunks ranked high)
- Context Recall: Directly measures retrieval coverage (right chunks retrieved at all)
- Answer Relevance: Affected if noisy retrieval causes off-topic LLM responses
- Answer Correctness: Cannot be correct if retrieved context is wrong

## Embedding model choice
all-MiniLM-L6-v2: 384 dimensions, ~80MB, runs on CPU, zero cost.
Good enough for English academic text at demo scale.

## FAISS index persistence
Indexes are saved to data/indexes/{strategy_name}/.
This means you only embed once per strategy — subsequent evaluation
runs load from disk in ~1 second instead of re-embedding in ~3 minutes.

## top_k parameter
Default top_k=5. This is an experiment variable tracked in MLflow.
- top_k=2: High precision, low recall (misses multi-part answers)
- top_k=5: Balanced (recommended default)
- top_k=10: High recall, low precision (too much noise for LLM)

## PROD SCALE (20,000 docs / 800K pages)
Switch to Azure AI Search hybrid (BM25 + dense vectors).
Key benefits over FAISS:
1. Hybrid retrieval: keyword matching catches exact terms (model names,
   numbers, acronyms) that dense embeddings miss
2. Metadata filtering: search only within specific papers or date ranges
3. Incremental updates: add new documents without rebuilding full index
4. Re-ranking: cross-encoder reranker improves top-k precision significantly

## Interview explanation
"The retrieval layer wraps FAISS behind a consistent interface — build,
save, load, retrieve. The key design decision is that each chunking strategy
gets its own named index on disk. This lets us run MLflow experiments
comparing recursive vs semantic chunking without re-embedding the corpus
each time — we just swap which index we load."

## Real retrieval quality observations (smoke test, 3 papers)

Query: "What are the two RAG formulations?"
- Rank 1 retrieved Hemingway literary text (from RAG paper experiment examples)
- Root cause: all-MiniLM-L6-v2 matched "document passages used in RAG paper"
  to "RAG formulations" — semantically adjacent but semantically wrong
- RAGAS Context Precision will catch and quantify this
- Fix at prod scale: Azure AI Search hybrid BM25+dense. BM25 would rank
  "RAG-Sequence" and "RAG-Token" highly by exact keyword match, overriding
  the noisy dense match.

Query: "What does faithfulness measure in RAGAS?"
- Top 3 chunks describe the framework broadly, none define faithfulness directly
- Root cause: faithfulness definition is likely in a short dense paragraph
  that recursive chunking split across two chunks, neither scoring high enough
- RAGAS Context Recall will catch this
- Fix: smaller chunk size (256 chars) or semantic chunking for this corpus

## Key insight for interviews
"We ran retrieval before evaluation and found two failure modes manually:
semantic noise from in-paper example documents, and recall failure from
over-large chunks splitting the answer. RAGAS then quantified these
observations as Context Precision=0.6 and Context Recall=0.7 — turning
qualitative observations into tracked experiment metrics."

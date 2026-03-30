# Ingestion — Notes

## What this folder does
Loads raw PDF documents and splits them into chunks that the vector store can index.
This is the first step in the pipeline and one of the most consequential — a bad 
chunking decision here propagates errors into every downstream metric.

## Why chunking strategy matters
If chunks are too large: the retrieved context is noisy, Context Precision drops.
If chunks are too small: a single answer spans multiple chunks that retrieval misses,
Context Recall drops.

## Four strategies implemented here
1. **Fixed-size** — split every N tokens regardless of content boundaries. Fast, 
   predictable, bad for structured content like papers with sections.
2. **Recursive** — split by paragraph → sentence → word. Respects natural boundaries.
   Best default for research papers.
3. **Semantic** — embed adjacent sentences, split where cosine similarity drops.
   Most expensive but highest quality for long-form academic text.
4. **Hierarchical** — chunk at section level AND sentence level, store both.
   Enables parent-document retrieval patterns.

## PROD SCALE note
At 20,000 docs, fixed and recursive chunking are still valid but must run as 
async batch jobs. Semantic chunking at scale requires a dedicated embedding service 
(Azure AI Services) because the local embedding call volume becomes prohibitive.

## Interview explanation
"The ingestion layer implements four chunking strategies with a common interface —
a senior engineer can swap strategies via config and immediately see the downstream 
impact on RAGAS Context Precision and Recall scores logged to MLflow."
## Smoke test results (3 papers: Attention, RAG, RAGAS — 42 pages)

| Strategy     | Chunks | Chunks/page | Notes                                      |
|--------------|--------|-------------|--------------------------------------------|
| Fixed        | 96     | 2.3         | Baseline                                   |
| Recursive    | 96     | 2.3         | Diverges from fixed at smaller chunk sizes |
| Semantic     | 908    | 21.6        | Sentence-level splits — fine grain         |
| Hier. parent | 42     | 1.0         | One parent per page at default size        |
| Hier. child  | 96     | 2.3         | Matches recursive at default size          |

## Key tradeoff observed
Semantic produces 9x more chunks than recursive. Expected RAGAS outcome:
- Semantic: higher Context Precision (tighter chunks), lower Context Recall
- Recursive: balanced Precision and Recall
- This tradeoff is the core experiment MLflow will track in Milestone 8.

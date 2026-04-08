"""
retrieval/vector_store.py

FAISS vector index — build, persist, and query.

TEACHING NOTE:
FAISS (Facebook AI Similarity Search) stores all vectors in memory and
performs exact nearest-neighbour search using cosine similarity.

Why FAISS for the demo:
- Zero infrastructure — no server, no docker, no API key
- Exact search — guaranteed to find the true nearest neighbours
- Persistence — save/load from disk so we do not re-embed on every run
- Fast enough for 20 docs (~800 pages → ~2000 chunks)

Why FAISS fails at production scale:
- In-memory: 20,000 docs at 384 dimensions = ~3GB RAM minimum
- No hybrid search: cannot combine BM25 keyword matching with dense vectors
- No filtering: cannot filter by metadata (e.g. "only search paper_id=03_gpt3")
- No incremental updates: adding one doc requires re-building the full index

TODO: FAISS IndexFlatL2 does exact search — switch to IndexIVFFlat for >50K vectors
"""

from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger

from retrieval.embedder import get_embedder

INDEX_DIR = Path(__file__).parent.parent / "data" / "indexes"


def build_index(
    chunks: list[Document],
    strategy_name: str = "recursive",
    index_dir: Optional[Path] = None,
) -> FAISS:
    """
    Build a FAISS index from document chunks and save to disk.

    Each chunking strategy gets its own index directory so we can
    compare retrieval quality across strategies in MLflow.

    Args:
        chunks: Document chunks from ingestion pipeline
        strategy_name: Used to name the index directory
        index_dir: Override default index location

    Returns:
        FAISS vector store ready for retrieval

    Raises:
        ValueError: If chunks list is empty
        RuntimeError: If embedding or index construction fails
    """
    if not chunks:
        raise ValueError(
            "Cannot build index from empty chunk list. "
            "Run ingestion pipeline first."
        )

    target_dir = (index_dir or INDEX_DIR) / strategy_name
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Building FAISS index: {len(chunks)} chunks "
        f"(strategy={strategy_name})"
    )

    # ──────────────────────────────────────────────────────────────
    # LOCAL DEMO — FAISS with sentence-transformers all-MiniLM-L6-v2
    # Zero cost, zero infrastructure, persisted to data/indexes/
    # Exact L2 search — finds true nearest neighbours every time
    # ──────────────────────────────────────────────────────────────
    embedder = get_embedder()

    try:
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedder,
        )
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        raise RuntimeError(f"Index construction failed: {e}") from e

    # ──────────────────────────────────────────────────────────────
    # [PRODUCTION] Azure AI Search hybrid index — uncomment
    # Requires: AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY,
    #           AZURE_SEARCH_INDEX_NAME, AZURE_OPENAI_* in .env
    # Handles: incremental updates, metadata filtering,
    #          hybrid BM25+dense retrieval, re-ranking, horizontal scaling
    # ──────────────────────────────────────────────────────────────
    # from langchain_community.vectorstores import AzureSearch
    # from langchain_openai import AzureOpenAIEmbeddings
    # embedder = AzureOpenAIEmbeddings(
    #     azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #     api_key=os.getenv("AZURE_OPENAI_KEY"),
    # )
    # vector_store = AzureSearch(
    #     azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    #     azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
    #     index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    #     embedding_function=embedder.embed_query,
    #     search_type="hybrid",  # BM25 + dense vectors — 15-25% better recall
    # )
    # return vector_store  # Azure Search does not need save_index()

    save_index(vector_store, strategy_name, index_dir)

    logger.success(
        f"FAISS index built and saved: {target_dir} "
        f"({len(chunks)} vectors, dim=384)"
    )
    return vector_store


def save_index(
    vector_store: FAISS,
    strategy_name: str = "recursive",
    index_dir: Optional[Path] = None,
) -> None:
    """Save FAISS index to disk for reuse across evaluation runs."""
    target_dir = (index_dir or INDEX_DIR) / strategy_name
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        vector_store.save_local(str(target_dir))
        logger.info(f"Index saved: {target_dir}")
    except Exception as e:
        logger.error(f"Failed to save index to {target_dir}: {e}")
        raise


def load_index(
    strategy_name: str = "recursive",
    index_dir: Optional[Path] = None,
) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Use this instead of build_index() when running repeated
    evaluation experiments — avoids re-embedding the entire corpus.

    Args:
        strategy_name: Which strategy's index to load
        index_dir: Override default index location

    Returns:
        FAISS vector store

    Raises:
        FileNotFoundError: If index does not exist on disk
    """
    target_dir = (index_dir or INDEX_DIR) / strategy_name

    if not target_dir.exists():
        raise FileNotFoundError(
            f"No index found at {target_dir}\n"
            f"Run build_index() first to create the index."
        )

    embedder = get_embedder()

    logger.info(f"Loading FAISS index from {target_dir}")

    try:
        vector_store = FAISS.load_local(
            str(target_dir),
            embeddings=embedder,
            allow_dangerous_deserialization=True,
        )
        logger.success(f"Index loaded: {target_dir}")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to load index from {target_dir}: {e}")
        raise


def retrieve(
    query: str,
    vector_store: FAISS,
    top_k: int = 5,
    score_threshold: Optional[float] = None,
) -> list[Document]:
    """
    Retrieve top-k most relevant chunks for a query.

    Args:
        query: User question string
        vector_store: Loaded FAISS index
        top_k: Number of chunks to retrieve (default 5)
        score_threshold: Optional minimum similarity score (0-1).

    Returns:
        List of Document objects ordered by relevance

    RAGAS CONTEXT NOTE:
    top_k=5 is the recommended default. Too few (top_k=2) hurts Context
    Recall. Too many (top_k=10) hurts Context Precision with noise.
    MLflow tracks top_k as a parameter so you can experiment with it.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    logger.debug(f"Retrieving top-{top_k} chunks for: {query[:80]}...")

    # ──────────────────────────────────────────────────────────────
    # LOCAL DEMO — FAISS dense similarity search
    # Exact L2 distance — no approximation, no BM25 component
    # ──────────────────────────────────────────────────────────────
    try:
        if score_threshold is not None:
            results = vector_store.similarity_search_with_relevance_scores(
                query, k=top_k
            )
            retrieved_chunks = [
                doc for doc, score in results
                if score >= score_threshold
            ]
            logger.debug(
                f"Retrieved {len(retrieved_chunks)} chunks "
                f"(filtered from {len(results)} by threshold={score_threshold})"
            )
        else:
            retrieved_chunks = vector_store.similarity_search(query, k=top_k)
            logger.debug(f"Retrieved {len(retrieved_chunks)} chunks")

        # ──────────────────────────────────────────────────────────
        # [PRODUCTION] Azure AI Search hybrid retrieval — uncomment
        # 15-25% better recall than dense-only on domain-specific corpora
        # ──────────────────────────────────────────────────────────
        # retrieved_chunks = vector_store.similarity_search(
        #     query, k=top_k,
        #     search_type="hybrid",   # BM25 + dense combined via RRF scoring
        # )

        return retrieved_chunks

    except Exception as e:
        logger.error(f"Retrieval failed for query '{query[:50]}...': {e}")
        raise
"""
retrieval/embedder.py

Embedding model wrapper for converting text chunks and queries
into dense vectors for FAISS similarity search.

TEACHING NOTE:
The embedding model is the bridge between text and vector space.
Two texts are "similar" if their vectors are close in cosine distance.
The quality of this similarity directly determines retrieval quality.

all-MiniLM-L6-v2 facts:
- 384-dimensional vectors
- ~80MB model, runs on CPU
- Trained on 1B+ sentence pairs
- Good for English academic text
- Speed: ~2000 sentences/second on CPU

WHY embedding quality matters for RAGAS:
- Context Precision: Low-quality embeddings return topically adjacent but
  semantically wrong chunks. A question about "attention heads" retrieves
  chunks about "attention mechanisms in biology" if embeddings are weak.
- Context Recall: If the correct chunk has unusual phrasing, weak embeddings
  fail to match it to the query. The answer exists in the corpus but is
  never retrieved.

PROD SCALE (20,000 docs / 800K pages):
# Use Azure text-embedding-3-large (3072 dimensions, much higher quality)
# from langchain_openai import AzureOpenAIEmbeddings
# embeddings = AzureOpenAIEmbeddings(
#     azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
# )
# At scale, batch embedding calls and cache results to avoid re-embedding
# identical chunks across evaluation runs.
"""

import os
from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings
from loguru import logger


class LocalEmbedder(Embeddings):
    """
    LangChain-compatible wrapper around sentence-transformers.
    Implements the Embeddings interface so it can be swapped with
    AzureOpenAIEmbeddings without changing any downstream code.

    DEMO (zero budget): all-MiniLM-L6-v2 — local CPU, zero API cost.
    PROD (paid): Azure text-embedding-3-large via AzureOpenAIEmbeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        logger.info(f"LocalEmbedder initialised (model={model_name})")

    @property
    def model(self):
        """Lazy load — model only downloads on first embed call."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.success(f"Embedding model loaded: {self.model_name}")
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document chunks.
        Called during index construction.

        Args:
            texts: List of chunk text strings

        Returns:
            List of embedding vectors (each a list of floats)
        """
        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 100,
            batch_size=64,
            normalize_embeddings=True,
        )
        logger.success(f"Embedded {len(texts)} chunks -> shape {embeddings.shape}")
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        Called at retrieval time for every user question.

        Args:
            text: Query string

        Returns:
            Single embedding vector
        """
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
        )
        return embedding[0].tolist()


@lru_cache(maxsize=1)
def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> LocalEmbedder:
    """
    Cached embedder factory. Returns the same instance on repeated calls.
    Prevents re-loading the model on every retrieval call.

    Usage:
        embedder = get_embedder()
        vector = embedder.embed_query("How does attention work?")
    """
    return LocalEmbedder(model_name=model_name)

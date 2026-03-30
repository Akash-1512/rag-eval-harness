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

WHY embedding quality matters for RAGAS:
- Context Precision: weak embeddings return topically adjacent but
  semantically wrong chunks
- Context Recall: if correct chunk has unusual phrasing, weak embeddings
  fail to match it to the query

IMPORTANT: The property is named _st_model (not model) to avoid collision
with RAGAS 0.2.x telemetry which expects .model to be a string.

PROD SCALE (20,000 docs / 800K pages):
# from langchain_openai import AzureOpenAIEmbeddings
# embeddings = AzureOpenAIEmbeddings(
#     azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
# )
"""

from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings
from loguru import logger


class LocalEmbedder(Embeddings):
    """
    LangChain-compatible wrapper around sentence-transformers.
    
    DEMO (zero budget): all-MiniLM-L6-v2 — local CPU, zero API cost.
    PROD (paid): AzureOpenAIEmbeddings with text-embedding-3-large.

    NOTE: model_name is a string attribute (not a property returning the
    SentenceTransformer instance) so RAGAS telemetry serialises it correctly.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # model_name as plain string — RAGAS reads this for telemetry
        self.model_name = model_name
        self._st_model = None  # lazy loaded on first call
        logger.info(f"LocalEmbedder initialised (model={model_name})")

    @property
    def _loaded_model(self):
        """Lazy load — model only downloads on first embed call."""
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._st_model = SentenceTransformer(self.model_name)
            logger.success(f"Embedding model loaded: {self.model_name}")
        return self._st_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document chunks.
        Called during index construction.
        """
        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self._loaded_model.encode(
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
        """
        embedding = self._loaded_model.encode(
            [text],
            normalize_embeddings=True,
        )
        return embedding[0].tolist()


@lru_cache(maxsize=1)
def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> LocalEmbedder:
    """
    Cached embedder factory. Returns same instance on repeated calls.
    Prevents re-loading the model on every retrieval call.
    """
    return LocalEmbedder(model_name=model_name)

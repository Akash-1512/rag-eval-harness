"""retrieval package — embedding and FAISS vector store."""

from retrieval.embedder import LocalEmbedder, get_embedder
from retrieval.vector_store import build_index, load_index, retrieve, save_index

__all__ = [
    "get_embedder",
    "LocalEmbedder",
    "build_index",
    "load_index",
    "retrieve",
    "save_index",
]

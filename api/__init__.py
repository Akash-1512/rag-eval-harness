"""api package — RAG pipeline and FastAPI backend."""

from api.rag_pipeline import run_rag, run_rag_batch, RAGOutput, get_llm

__all__ = [
    "run_rag",
    "run_rag_batch",
    "RAGOutput",
    "get_llm",
]

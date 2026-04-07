"""api package — RAG pipeline and FastAPI backend."""

from api.rag_pipeline import RAGOutput, get_llm, run_rag, run_rag_batch

__all__ = [
    "run_rag",
    "run_rag_batch",
    "RAGOutput",
    "get_llm",
]

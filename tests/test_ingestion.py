"""
tests/test_ingestion.py
Unit tests for document loader and chunker.
These run in CI — no LLM calls, no network, no PDFs required.
"""

from langchain_core.documents import Document

from ingestion.chunker import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    ChunkingStrategy,
    chunk_documents,
    chunk_fixed,
    chunk_hierarchical,
    chunk_recursive,
)


def _make_test_docs(n: int = 3) -> list[Document]:
    """Create synthetic Documents for unit testing chunker logic."""
    return [
        Document(
            page_content=(
                f"This is test document {i}. " * 50  # ~250 words
            ),
            metadata={
                "source": f"test_{i}.pdf",
                "filename": f"test_{i}.pdf",
                "paper_id": f"test_{i}",
                "title": f"Test Paper {i}",
                "page": 1,
                "total_pages": 1,
            },
        )
        for i in range(n)
    ]


def test_fixed_chunking_produces_chunks():
    docs = _make_test_docs()
    chunks = chunk_fixed(docs)
    assert len(chunks) > 0, "Fixed chunking produced no chunks"


def test_recursive_chunking_produces_chunks():
    docs = _make_test_docs()
    chunks = chunk_recursive(docs)
    assert len(chunks) > 0, "Recursive chunking produced no chunks"


def test_chunk_metadata_keys_present():
    docs = _make_test_docs()
    chunks = chunk_recursive(docs)
    required = {"chunk_index", "chunking_strategy", "chunk_char_count"}
    for chunk in chunks:
        assert required.issubset(set(chunk.metadata.keys())), (
            f"Missing metadata keys: {required - set(chunk.metadata.keys())}"
        )


def test_chunking_strategy_enum_in_metadata():
    docs = _make_test_docs()
    chunks = chunk_fixed(docs)
    assert all(
        c.metadata["chunking_strategy"] == "fixed" for c in chunks
    ), "Strategy metadata incorrect on fixed chunks"


def test_hierarchical_returns_tuple():
    docs = _make_test_docs()
    result = chunk_hierarchical(docs)
    assert isinstance(result, tuple), "Hierarchical should return tuple"
    assert len(result) == 2, "Hierarchical should return (parent, child) tuple"
    parent_chunks, child_chunks = result
    assert len(parent_chunks) > 0
    assert len(child_chunks) > 0


def test_child_chunks_have_parent_id():
    docs = _make_test_docs()
    _, child_chunks = chunk_hierarchical(docs)
    assert all(
        "parent_id" in c.metadata for c in child_chunks
    ), "Child chunks missing parent_id"


def test_chunk_documents_dispatcher():
    docs = _make_test_docs()
    for strategy in [ChunkingStrategy.FIXED, ChunkingStrategy.RECURSIVE]:
        chunks = chunk_documents(docs, strategy=strategy)
        assert len(chunks) > 0, f"No chunks for strategy {strategy}"
        assert all(
            c.metadata["chunking_strategy"] == strategy.value
            for c in chunks
        )

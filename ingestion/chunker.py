"""
ingestion/chunker.py

Four chunking strategies with a unified interface.
All strategies accept List[Document] and return List[Document].
The strategy is selected via the ChunkingStrategy enum.

TEACHING NOTE:
Why does chunking strategy affect RAGAS metrics?

1. Context Precision: If chunks are too large, retrieved chunks contain
   both relevant and irrelevant content. The relevant signal is diluted.
   Score drops because the LLM has to work through noise.

2. Context Recall: If chunks are too small, a single answer may span
   multiple chunks. If retrieval returns top-3 and the answer is split
   across chunks 4 and 7, recall drops to zero for that question.

3. Faithfulness: Noisy chunks (too large) tempt the LLM to hallucinate
   connections between unrelated passages in the same chunk.

PROD SCALE (20,000 docs / 800K pages):
- Fixed + Recursive: Parallelise with concurrent.futures
- Semantic: Requires dedicated embedding service (Azure AI Services)
- Hierarchical: Requires dual-index in Azure AI Search (parent + child)
"""

from enum import Enum
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
CHARS_PER_TOKEN_APPROX = 4

CHUNK_SIZE_CHARS = DEFAULT_CHUNK_SIZE * CHARS_PER_TOKEN_APPROX
CHUNK_OVERLAP_CHARS = DEFAULT_CHUNK_OVERLAP * CHARS_PER_TOKEN_APPROX


def chunk_fixed(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[Document]:
    """
    Strategy 1: Fixed-size chunking.
    Splits at exactly chunk_size characters regardless of boundaries.
    Used as the baseline for RAGAS metric comparison.

    WEAKNESS: Cuts mid-sentence and mid-equation in research papers.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    chunks = _attach_chunk_metadata(chunks, strategy=ChunkingStrategy.FIXED)
    logger.info(
        f"Fixed chunking: {len(documents)} pages -> {len(chunks)} chunks "
        f"(size={chunk_size} chars, overlap={chunk_overlap} chars)"
    )
    return chunks


def chunk_recursive(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[Document]:
    """
    Strategy 2: Recursive character splitting.
    Splits by paragraph first, then sentence, then word, then character.
    Preserves natural language boundaries. Recommended default for papers.

    INTERVIEW EXPLANATION: Recursive splitting respects the natural structure
    of academic writing — each chunk tends to contain one complete thought,
    which improves context precision and answer faithfulness.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    chunks = _attach_chunk_metadata(chunks, strategy=ChunkingStrategy.RECURSIVE)
    logger.info(
        f"Recursive chunking: {len(documents)} pages -> {len(chunks)} chunks "
        f"(size={chunk_size} chars, overlap={chunk_overlap} chars)"
    )
    return chunks


def chunk_semantic(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
    similarity_threshold: float = 0.85,
) -> list[Document]:
    """
    Strategy 3: Semantic chunking.
    Groups sentences together while embedding similarity stays above threshold.
    Splits at genuine topic shifts rather than arbitrary character counts.

    DEMO (zero budget): Uses local sentence-transformers all-MiniLM-L6-v2.
    PROD (paid):
    # from langchain_openai import AzureOpenAIEmbeddings
    # model = AzureOpenAIEmbeddings(
    #     azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    # )
    """
    logger.info(
        f"Semantic chunking: {len(documents)} pages "
        f"(threshold={similarity_threshold}) -- loading embedding model..."
    )

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError as e:
        logger.error(f"sentence-transformers not installed: {e}")
        raise

    all_chunks = []

    for doc in documents:
        sentences = [
            s.strip()
            for s in doc.page_content.split(". ")
            if len(s.strip()) > 20
        ]
        if not sentences:
            continue

        embeddings = model.encode(sentences, show_progress_bar=False)

        current_sentences = [sentences[0]]
        current_chars = len(sentences[0])

        for i in range(1, len(sentences)):
            sim = float(
                np.dot(embeddings[i - 1], embeddings[i]) /
                (np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]) + 1e-8)
            )
            would_exceed = (current_chars + len(sentences[i])) > chunk_size
            topic_shift = sim < similarity_threshold

            if would_exceed or topic_shift:
                all_chunks.append(Document(
                    page_content=". ".join(current_sentences),
                    metadata={**doc.metadata},
                ))
                current_sentences = [sentences[i]]
                current_chars = len(sentences[i])
            else:
                current_sentences.append(sentences[i])
                current_chars += len(sentences[i])

        if current_sentences:
            all_chunks.append(Document(
                page_content=". ".join(current_sentences),
                metadata={**doc.metadata},
            ))

    all_chunks = _attach_chunk_metadata(all_chunks, strategy=ChunkingStrategy.SEMANTIC)
    logger.info(
        f"Semantic chunking: {len(documents)} pages -> {len(all_chunks)} chunks"
    )
    return all_chunks


def chunk_hierarchical(
    documents: list[Document],
    parent_chunk_size: int = CHUNK_SIZE_CHARS * 4,
    child_chunk_size: int = CHUNK_SIZE_CHARS,
    child_chunk_overlap: int = CHUNK_OVERLAP_CHARS,
) -> tuple[list[Document], list[Document]]:
    """
    Strategy 4: Hierarchical chunking.
    Produces parent chunks (section-level) and child chunks (sentence-level).
    Index child chunks for retrieval precision, return parent chunks as context.

    RAGAS IMPACT: Improves Context Recall (parent has full answer) while
    keeping embedding precision high (child is the retrieval unit).

    PROD SCALE (20,000 docs / 800K pages):
    # Requires two Azure AI Search indexes:
    # - child_index: searched during retrieval
    # - parent_index: fetched by parent_id after child match
    # Store parent_id in child metadata to enable parent lookup.

    Returns:
        Tuple of (parent_chunks, child_chunks)
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". "],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_chunks = parent_splitter.split_documents(documents)
    parent_chunks = _attach_chunk_metadata(
        parent_chunks, strategy=ChunkingStrategy.HIERARCHICAL, level="parent"
    )

    child_chunks = []
    for parent_idx, parent in enumerate(parent_chunks):
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent_idx
            child.metadata["level"] = "child"
        child_chunks.extend(children)

    child_chunks = _attach_chunk_metadata(
        child_chunks, strategy=ChunkingStrategy.HIERARCHICAL, level="child"
    )

    logger.info(
        f"Hierarchical chunking: {len(documents)} pages -> "
        f"{len(parent_chunks)} parent chunks, {len(child_chunks)} child chunks"
    )
    return parent_chunks, child_chunks


def chunk_documents(
    documents: list[Document],
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[Document]:
    """
    Unified chunking interface. Select strategy via enum.
    All strategies return List[Document] with consistent metadata.
    Called by RAG pipeline and MLflow tracking.
    """
    logger.info(f"Chunking {len(documents)} documents with strategy: {strategy}")

    if strategy == ChunkingStrategy.FIXED:
        return chunk_fixed(documents, chunk_size, chunk_overlap)
    elif strategy == ChunkingStrategy.RECURSIVE:
        return chunk_recursive(documents, chunk_size, chunk_overlap)
    elif strategy == ChunkingStrategy.SEMANTIC:
        return chunk_semantic(documents, chunk_size, chunk_overlap)
    elif strategy == ChunkingStrategy.HIERARCHICAL:
        _, child_chunks = chunk_hierarchical(documents, child_chunk_size=chunk_size)
        return child_chunks
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Choose from: {[s.value for s in ChunkingStrategy]}"
        )


def _attach_chunk_metadata(
    chunks: list[Document],
    strategy: ChunkingStrategy,
    level: Optional[str] = None,
) -> list[Document]:
    """Attach chunk-level metadata to every Document."""
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunking_strategy"] = strategy.value
        chunk.metadata["chunk_char_count"] = len(chunk.page_content)
        if level:
            chunk.metadata["level"] = level
    return chunks

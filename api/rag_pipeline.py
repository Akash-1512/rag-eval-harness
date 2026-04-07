"""
api/rag_pipeline.py

Core RAG pipeline: question -> retrieve -> prompt -> LLM -> answer.

Output format is the RAGAS evaluation dict:
{
    "question":     str,           # original question
    "answer":       str,           # LLM generated answer
    "contexts":     list[str],     # retrieved chunk texts
    "ground_truth": str,           # from Q&A CSV (empty string if not provided)
    "metadata": {
        "sources":          list[str],   # paper titles of retrieved chunks
        "pages":            list[int],   # page numbers of retrieved chunks
        "chunking_strategy": str,        # which strategy produced these chunks
        "top_k":            int,         # how many chunks were retrieved
        "model":            str,         # which LLM was used
    }
}

TEACHING NOTE:
Why does output format matter so much?
RAGAS, DeepEval, MLflow, and the Streamlit dashboard all consume this dict.
If we change the key names here, all four break simultaneously.
Defining the schema once in a Pydantic model (RAGOutput below) means
any format change is caught at runtime with a clear error message,
not a silent KeyError three components later.

PROD SCALE (20,000 docs / 800K pages):
# Replace Groq with Azure OpenAI:
# from langchain_openai import AzureChatOpenAI
# llm = AzureChatOpenAI(
#     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     temperature=0,
# )
# Also add async batch processing for running 50+ Q&A pairs in parallel.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


# ── Output schema ─────────────────────────────────────────────────────────────

class RAGOutput(BaseModel):
    """
    Structured output from the RAG pipeline.
    This is the exact format consumed by RAGAS, DeepEval, and MLflow.
    Pydantic validates the structure on every call — no silent format errors.
    """
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""
    metadata: dict = {}


# ── Prompt template ───────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precise research assistant answering questions about AI and ML research papers.

Rules:
1. Answer ONLY based on the provided context passages.
2. If the answer is not in the context, say exactly: "The provided context does not contain information to answer this question."
3. Be specific — include exact numbers, names, and technical terms from the context.
4. Do not add information from your general training knowledge.
5. Keep answers concise and directly responsive to the question.""",
    ),
    (
        "human",
        """Context passages:
{context}

Question: {question}

Answer:""",
    ),
])


# ── LLM factory ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Cached LLM factory. Returns same instance on repeated calls.
    Prevents re-initialising the client on every RAG call.

    DEMO (zero budget): Groq llama-3.1-70b-versatile
    PROD (paid):
    # from langchain_openai import AzureChatOpenAI
    # return AzureChatOpenAI(
    #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    #     temperature=0,
    # )
    """
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL_NAME", "llama-3.1-70b-versatile")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set in .env\n"
            "Get a free key at https://console.groq.com"
        )

    logger.info(f"Initialising LLM: Groq {model}")

    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=0,       # Deterministic — critical for reproducible eval
        max_tokens=1024,
    )


# ── Core pipeline ─────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_llm_with_retry(chain, inputs: dict) -> str:
    """
    Call LLM chain with automatic retry on transient errors.
    Groq free tier occasionally rate-limits or returns 503 — retry handles this.
    tenacity retries up to 3 times with exponential backoff (2s, 4s, 8s).
    """
    return chain.invoke(inputs)


def run_rag(
    question: str,
    vector_store: FAISS,
    ground_truth: str = "",
    top_k: int = 5,
    chunking_strategy: str = "recursive",
) -> RAGOutput:
    """
    Run the full RAG pipeline for a single question.

    Args:
        question:          The question to answer
        vector_store:      Loaded FAISS index
        ground_truth:      Reference answer from Q&A CSV (for RAGAS scoring)
        top_k:             Number of chunks to retrieve
        chunking_strategy: Name of strategy used to build this index

    Returns:
        RAGOutput with question, answer, contexts, ground_truth, metadata

    Raises:
        ValueError: If question is empty
        RuntimeError: If LLM call fails after 3 retries
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    logger.info(f"RAG query: {question[:80]}...")

    # Step 1: Retrieve relevant chunks
    from retrieval.vector_store import retrieve
    retrieved_docs = retrieve(question, vector_store, top_k=top_k)

    if not retrieved_docs:
        logger.warning(f"No chunks retrieved for: {question[:80]}")
        return RAGOutput(
            question=question,
            answer="The provided context does not contain information to answer this question.",
            contexts=[],
            ground_truth=ground_truth,
            metadata={
                "sources": [],
                "pages": [],
                "chunking_strategy": chunking_strategy,
                "top_k": top_k,
                "model": os.getenv("GROQ_MODEL_NAME", "llama-3.1-70b-versatile"),
            },
        )

    # Step 2: Format context for prompt
    context_text = "\n\n---\n\n".join([
        f"[{doc.metadata.get('title', 'Unknown')} p.{doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    # Step 3: Build and run LLM chain
    llm = get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()

    try:
        answer = _call_llm_with_retry(chain, {
            "context": context_text,
            "question": question,
        })
        logger.success(f"Answer generated ({len(answer)} chars)")

    except Exception as e:
        logger.error(f"LLM call failed after retries: {e}")
        raise RuntimeError(f"RAG pipeline failed for question: {question[:80]}") from e

    # Step 4: Build structured output
    output = RAGOutput(
        question=question,
        answer=answer.strip(),
        contexts=[doc.page_content for doc in retrieved_docs],
        ground_truth=ground_truth,
        metadata={
            "sources": [doc.metadata.get("title", "Unknown") for doc in retrieved_docs],
            "pages": [doc.metadata.get("page", -1) for doc in retrieved_docs],
            "chunking_strategy": chunking_strategy,
            "top_k": top_k,
            "model": os.getenv("GROQ_MODEL_NAME", "llama-3.1-70b-versatile"),
        },
    )

    logger.debug(f"Sources: {output.metadata['sources']}")
    return output


def run_rag_batch(
    qa_pairs: list[dict],
    vector_store: FAISS,
    top_k: int = 5,
    chunking_strategy: str = "recursive",
) -> list[RAGOutput]:
    """
    Run RAG pipeline over a list of Q&A pairs.
    Used by RAGAS evaluation pipeline to process all 50 questions.

    Args:
        qa_pairs: List of dicts with keys "question" and "ground_truth"
        vector_store: Loaded FAISS index
        top_k: Number of chunks to retrieve per question
        chunking_strategy: Strategy name for metadata

    Returns:
        List of RAGOutput objects, one per question

    Example input:
        qa_pairs = [
            {"question": "How many attention heads...", "ground_truth": "8 heads..."},
            {"question": "What are the two RAG formulations...", "ground_truth": "RAG-Sequence..."},
        ]
    """
    if not qa_pairs:
        raise ValueError("qa_pairs list is empty")

    logger.info(f"Running RAG batch: {len(qa_pairs)} questions")
    results = []

    for i, pair in enumerate(qa_pairs, 1):
        question = pair.get("question", "")
        ground_truth = pair.get("ground_truth", "")

        if not question:
            logger.warning(f"Skipping empty question at index {i}")
            continue

        logger.info(f"  [{i}/{len(qa_pairs)}] {question[:60]}...")

        try:
            output = run_rag(
                question=question,
                vector_store=vector_store,
                ground_truth=ground_truth,
                top_k=top_k,
                chunking_strategy=chunking_strategy,
            )
            results.append(output)

        except Exception as e:
            logger.error(f"  Failed question {i}: {e}")
            # Append a failed result rather than crashing the entire batch
            results.append(RAGOutput(
                question=question,
                answer=f"ERROR: {str(e)}",
                contexts=[],
                ground_truth=ground_truth,
                metadata={"error": str(e)},
            ))
            continue

    logger.success(
        f"Batch complete: {sum(1 for r in results if 'error' not in r.metadata)}"
        f"/{len(qa_pairs)} successful"
    )
    return results

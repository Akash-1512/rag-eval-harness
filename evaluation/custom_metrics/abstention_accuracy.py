"""
evaluation/custom_metrics/abstention_accuracy.py

Custom LLM-as-Judge metric for abstention accuracy.

WHAT: Measures whether the RAG system correctly refuses to answer
questions that are outside its knowledge base (corpus).

WHY this matters in production:
A RAG system that confidently answers out-of-scope questions is MORE
dangerous than one that gives wrong answers to in-scope questions.
Wrong in-scope answers are caught by faithfulness scoring. Confident
hallucination on out-of-scope questions is invisible to RAGAS.

RAGAS has no abstention metric. This is a gap we fill with a custom
LLM-as-Judge evaluator.

HOW it works:
1. We define a set of out-of-scope questions (topics not in our 20 papers)
2. We run them through the RAG pipeline
3. An LLM judge scores whether each response is a correct abstention
   (says "I don't know / not in my context") or a hallucination

PROD SCALE (20,000 docs / 800K pages):
# Sample out-of-scope queries from production logs (Azure App Insights)
# rather than hand-crafting them. Real users generate the best adversarial
# out-of-scope questions naturally.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from loguru import logger

load_dotenv()

# Questions that are NOT in our 20-paper corpus
# Each tests a different type of out-of-scope failure
OUT_OF_SCOPE_QUESTIONS = [
    # Version confusion — model exists but version not in corpus
    "What are the key architectural improvements in LLaMA 3 compared to LLaMA 2?",
    # Completely absent topic
    "How does the Mamba state space model architecture work?",
    # Post-corpus publication
    "What does the DeepSeek R1 paper say about chain-of-thought reasoning?",
    # Adjacent domain not covered
    "What is the CLIP model and how does it connect vision and language?",
    # Specific benchmark not discussed
    "What score did GPT-4o achieve on the MMMU benchmark?",
]

ABSTENTION_JUDGE_PROMPT = """You are evaluating whether an AI assistant correctly abstained from answering a question that was outside its knowledge base.

Question: {question}

AI Response: {response}

A CORRECT abstention contains phrases like:
- "not in the provided context"
- "I don't have information about"
- "the context does not contain"
- "cannot find this in the documents"
- "outside the scope of"

An INCORRECT response confidently answers the question using information not from the corpus, or gives a partial answer without clearly flagging the limitation.

Did the AI correctly abstain? Answer with ONLY: CORRECT or INCORRECT
Then on a new line explain why in one sentence."""


@dataclass
class AbstentionResult:
    """Result for a single out-of-scope query."""
    question: str
    response: str
    judgment: str        # CORRECT or INCORRECT
    reason: str
    is_correct: bool


@dataclass
class AbstentionAccuracyResult:
    """Aggregate abstention accuracy result."""
    score: float                        # 0.0 to 1.0
    num_correct: int
    num_total: int
    per_question: list[AbstentionResult]

    def summary(self) -> str:
        return (
            f"Abstention Accuracy: {self.score:.3f} "
            f"({self.num_correct}/{self.num_total} correct abstentions)"
        )


def _get_judge_llm() -> ChatGroq:
    """
    LLM used to judge abstention quality.

    DEMO (zero budget): Groq llama-3.3-70b-versatile
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
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
        temperature=0,
        max_tokens=200,
    )


def evaluate_abstention_accuracy(
    rag_outputs: list[dict],
) -> AbstentionAccuracyResult:
    """
    Evaluate abstention accuracy on out-of-scope RAG outputs.

    Args:
        rag_outputs: List of dicts with "question" and "answer" keys
                     These should be responses to OUT_OF_SCOPE_QUESTIONS

    Returns:
        AbstentionAccuracyResult with score and per-question breakdown
    """
    if not rag_outputs:
        raise ValueError("rag_outputs is empty")

    judge = _get_judge_llm()
    results = []

    for output in rag_outputs:
        question = output.get("question", "")
        response = output.get("answer", "")

        prompt = ABSTENTION_JUDGE_PROMPT.format(
            question=question,
            response=response,
        )

        try:
            judgment_response = judge.invoke(prompt).content.strip()
            lines = judgment_response.split("\n", 1)
            judgment = lines[0].strip().upper()
            reason = lines[1].strip() if len(lines) > 1 else "No reason provided"

            is_correct = judgment == "CORRECT"

            results.append(AbstentionResult(
                question=question,
                response=response,
                judgment=judgment,
                reason=reason,
                is_correct=is_correct,
            ))

            status = "✓" if is_correct else "✗"
            logger.info(f"  {status} Abstention [{judgment}]: {question[:60]}...")

        except Exception as e:
            logger.error(f"Judge call failed for: {question[:60]}... — {e}")
            results.append(AbstentionResult(
                question=question,
                response=response,
                judgment="ERROR",
                reason=str(e),
                is_correct=False,
            ))

    num_correct = sum(1 for r in results if r.is_correct)
    score = num_correct / len(results) if results else 0.0

    result = AbstentionAccuracyResult(
        score=score,
        num_correct=num_correct,
        num_total=len(results),
        per_question=results,
    )

    logger.success(result.summary())
    return result

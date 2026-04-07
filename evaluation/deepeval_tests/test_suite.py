"""
evaluation/deepeval_tests/test_suite.py

DeepEval G-Eval assertion suite using Groq as the judge LLM.

WHAT: Unit-test-style pass/fail assertions on RAG outputs.
Unlike RAGAS continuous scores, these give binary results per question.

WHY DeepEval on top of RAGAS:
RAGAS: "average faithfulness = 0.93"
DeepEval: "Q2 FAILED faithfulness assertion — answer added uncited claims"
Both needed — averages hide individual failures.

DEMO (zero budget): Groq llama-3.3-70b via DeepEvalBaseLLM wrapper.
PROD (paid):
# Remove GroqDeepEvalLLM entirely — set OPENAI_API_KEY to Azure key.
# DeepEval natively supports OpenAI/Azure. GPT-4o gives most consistent scores.

PROD SCALE (20,000 docs / 800K pages):
# Run assertions on 10% sample per run, rotating sample each time.
# Focus on high-stakes categories: numerical answers, compliance queries.
"""

import os
from dataclasses import dataclass, field

from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from loguru import logger

from api.rag_pipeline import RAGOutput

load_dotenv()

DEFAULT_THRESHOLD = float(os.getenv("DEEPEVAL_CONFIDENCE_THRESHOLD", "0.5"))


# ── Groq adapter for DeepEval ─────────────────────────────────────────────────

class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """
    Wraps Groq LLM as a DeepEval-compatible judge.
    DeepEval calls this instead of OpenAI for all G-Eval scoring.

    DEMO (zero budget): routes through Groq free tier.
    PROD (paid): remove this class entirely — set OPENAI_API_KEY and
    DeepEval will use GPT-4o natively with better consistency.
    """

    def __init__(self):
        self.model = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
            temperature=0,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str, schema=None) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")


# ── Result schema ─────────────────────────────────────────────────────────────

@dataclass
class DeepEvalResult:
    """
    Result from DeepEval G-Eval assertions.
    Consumed by MLflow (M8) and Streamlit dashboard (M10).
    """
    faithfulness_pass_rate: float = 0.0
    answer_relevancy_pass_rate: float = 0.0
    completeness_pass_rate: float = 0.0
    test_results: list[dict] = field(default_factory=list)
    num_questions: int = 0
    threshold: float = DEFAULT_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "deepeval_faithfulness_pass_rate": self.faithfulness_pass_rate,
            "deepeval_answer_relevancy_pass_rate": self.answer_relevancy_pass_rate,
            "deepeval_completeness_pass_rate": self.completeness_pass_rate,
            "deepeval_num_questions": self.num_questions,
            "deepeval_threshold": self.threshold,
        }

    def summary(self) -> str:
        return (
            f"DeepEval Assertion Results ({self.num_questions} questions)\n"
            f"  Faithfulness pass rate:    {self.faithfulness_pass_rate:.3f}\n"
            f"  Answer relevancy pass rate: {self.answer_relevancy_pass_rate:.3f}\n"
            f"  Completeness pass rate:    {self.completeness_pass_rate:.3f}\n"
            f"  Threshold: {self.threshold}"
        )


# ── Core evaluation ───────────────────────────────────────────────────────────

def run_deepeval_assertions(
    rag_outputs: list[RAGOutput],
    threshold: float = DEFAULT_THRESHOLD,
) -> DeepEvalResult:
    """
    Run G-Eval assertions using Groq as the judge LLM.

    G-Eval works by prompting the LLM with a natural-language rubric
    and asking it to score the output on a 0-10 scale. This is more
    flexible than RAGAS — any criterion can be tested by writing a rubric.

    Three criteria tested here:
    1. Faithfulness — is every claim grounded in retrieved context?
    2. Answer relevancy — does the response directly address the question?
    3. Completeness — does the answer cover the key points in ground truth?
    """
    if not rag_outputs:
        raise ValueError("rag_outputs is empty")

    valid_outputs = [o for o in rag_outputs if "error" not in o.metadata]
    if not valid_outputs:
        raise ValueError("All RAG outputs contain errors")

    logger.info(
        f"Running DeepEval G-Eval assertions: {len(valid_outputs)} questions, "
        f"threshold={threshold}"
    )

    judge = GroqDeepEvalLLM()

    # Define G-Eval criteria with natural language rubrics
    faithfulness_metric = GEval(
        name="Faithfulness",
        criteria=(
            "Determine whether every factual claim in the 'actual output' "
            "is directly supported by information in the 'retrieval context'. "
            "Penalise any claim that cannot be traced to the context."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=threshold,
        model=judge,
    )

    relevancy_metric = GEval(
        name="Answer Relevancy",
        criteria=(
            "Determine whether the 'actual output' directly and completely "
            "addresses the 'input' question. Penalise tangential or off-topic content."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
        model=judge,
    )

    completeness_metric = GEval(
        name="Completeness",
        criteria=(
            "Compare the 'actual output' against the 'expected output'. "
            "Determine whether the actual output covers all key claims "
            "present in the expected output. Penalise missing key information."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=threshold,
        model=judge,
    )

    faithfulness_passes = []
    relevancy_passes = []
    completeness_passes = []
    test_results = []

    for i, output in enumerate(valid_outputs, 1):
        logger.info(f"  [{i}/{len(valid_outputs)}] {output.question[:60]}...")

        test_case = LLMTestCase(
            input=output.question,
            actual_output=output.answer,
            retrieval_context=output.contexts if output.contexts else [""],
            expected_output=output.ground_truth if output.ground_truth else "",
        )

        case_result = {"question": output.question, "answer": output.answer[:150]}

        for metric, passes_list, name in [
            (faithfulness_metric, faithfulness_passes, "Faithfulness"),
            (relevancy_metric, relevancy_passes, "AnswerRelevancy"),
            (completeness_metric, completeness_passes, "Completeness"),
        ]:
            try:
                metric.measure(test_case)
                passed = metric.is_successful()
                score = getattr(metric, "score", 0.0) or 0.0
                reason = getattr(metric, "reason", "") or ""

                status = "PASS" if passed else "FAIL"
                logger.info(f"    {name}: {status} ({score:.3f}) — {reason[:80]}")
                passes_list.append(passed)
                case_result[name] = {"passed": passed, "score": score, "reason": reason}

            except Exception as e:
                logger.error(f"    {name} failed: {e}")
                passes_list.append(False)
                case_result[name] = {"passed": False, "score": 0.0, "reason": str(e)}

        test_results.append(case_result)

    def pass_rate(passes: list) -> float:
        return sum(passes) / len(passes) if passes else 0.0

    result = DeepEvalResult(
        faithfulness_pass_rate=pass_rate(faithfulness_passes),
        answer_relevancy_pass_rate=pass_rate(relevancy_passes),
        completeness_pass_rate=pass_rate(completeness_passes),
        test_results=test_results,
        num_questions=len(valid_outputs),
        threshold=threshold,
    )

    logger.success(f"\n{result.summary()}")
    return result

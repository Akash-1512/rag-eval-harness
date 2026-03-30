# Evaluation — Notes

## What this folder does
Contains three evaluation sub-systems that run in parallel against the same 
RAG pipeline output: RAGAS (reference metrics), DeepEval (assertion-based), 
and custom metrics (abstention accuracy, numerical faithfulness).

## Why three evaluation frameworks?
No single framework is complete:
- RAGAS gives you grounded metric scores but is weak on custom business logic.
- DeepEval gives you assertion-based pass/fail tests like a unit test suite.
- Custom metrics capture domain-specific failure modes RAGAS doesn't model 
  (e.g. whether the system correctly refuses out-of-scope questions).

## Key design decision
All three evaluators receive the same input triple:
  (question, generated_answer, retrieved_contexts, ground_truth)
This ensures metric comparisons are apples-to-apples across frameworks.

## PROD SCALE note
At 20,000 docs, running all 5 RAGAS metrics on every question is 
cost-prohibitive (each metric = 2-4 LLM calls). Implement stratified sampling:
evaluate 10% of questions per run, rotate the sample each run so full 
coverage is achieved over 10 runs. Track sample indices in MLflow.

## Interview explanation
"The evaluation layer is framework-agnostic — it wraps RAGAS, DeepEval, 
and custom LLM-as-Judge evaluators behind a common interface so new metrics 
can be added without touching the pipeline or tracking code."


## Zero-budget vs prod stack

| Component | Demo (free) | Prod (paid) |
|---|---|---|
| LLM | Groq llama-3.1-70b-versatile | Azure OpenAI GPT-4o |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | text-embedding-3-large |
| Vector store | FAISS (in-memory) | Azure AI Search hybrid |
| MLflow | Local server | Azure ML workspace |
| LLM-as-Judge | Groq llama-3.1-70b | GPT-4o |

All paid alternatives are present in the codebase as commented blocks
tagged `# PROD (paid)` so switching is a one-line config change.

## Groq free tier limits (as of 2024)
- 14,400 requests/day on llama-3.1-8b-instant
- 30 requests/minute
- Context window: 131,072 tokens
- This is sufficient for: full RAGAS evaluation run (~200 LLM calls),
  full red-team run (~120 LLM calls), dashboard queries.
  Stay within limits by not running all three simultaneously.
## Groq model selection (as of March 2026)
llama-3.1-70b-versatile has been decommissioned by Groq.
Current recommended model: qwen/qwen3-32b
Check available models with:
    python -c "from groq import Groq; import os; from dotenv import load_dotenv; load_dotenv(); [print(m.id) for m in Groq(api_key=os.getenv('GROQ_API_KEY')).models.list().data]"

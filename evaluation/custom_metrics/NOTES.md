# Custom Metrics — Teaching Notes

## What this folder does
Implements evaluation metrics that RAGAS and DeepEval do not cover.
Currently: abstention_accuracy — whether the system correctly refuses
out-of-scope questions.

## Why abstention accuracy matters
A RAG system has two failure modes:
1. Wrong answer on in-scope question — caught by RAGAS faithfulness
2. Confident hallucination on out-of-scope question — NOT caught by RAGAS

The second failure mode is more dangerous in production because the
system appears confident. A user asking about LLaMA 3 (not in corpus)
should get "I don't have information about this" not a fabricated answer.

## How it is measured
LLM-as-Judge pattern: a judge LLM reads the question and response,
then decides whether the response is a correct abstention or a
hallucination. The judge is prompted with explicit criteria for what
a correct abstention looks like.

## Out-of-scope question taxonomy
Five categories of out-of-scope questions we test:
1. Version confusion (LLaMA 3 — only 1/2 in corpus)
2. Completely absent topic (Mamba SSM)
3. Post-corpus publication (DeepSeek R1)
4. Adjacent domain (CLIP — not in corpus)
5. Specific benchmark not discussed (MMMU scores)

## PROD SCALE note
At production scale, out-of-scope questions come from real user logs.
Use Azure Application Insights to sample queries that returned low
confidence scores, then run abstention accuracy on those queries.
This gives you a live measure of where your system is hallucinating.

## Interview explanation
"We implemented a custom abstention accuracy metric because RAGAS has
no way to measure whether the system correctly refuses out-of-scope
questions — one of the most important safety properties in production.
The metric uses an LLM judge with explicit criteria, and we track it
in MLflow alongside RAGAS scores to get full coverage of the failure
surface."

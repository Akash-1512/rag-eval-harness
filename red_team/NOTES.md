# Red Team — Notes

## What this folder does
A LangGraph agent that autonomously generates adversarial prompts, runs them 
through the RAG pipeline, and measures failure rate across 6 attack types.

## Why red-team a RAG system?
Standard evaluation metrics only measure quality on expected inputs.
Real users ask unexpected, adversarial, or ambiguous questions.
Red-teaming reveals the failure surface that RAGAS scores hide.

## Six attack vectors
1. Version confusion — ask about model versions not in the corpus
2. Numerical hallucination — probe exact benchmark numbers
3. Premise injection — embed false facts in the question
4. Cross-paper contradiction — force synthesis of conflicting claims
5. Out-of-scope query — ask about topics absent from the corpus
6. Temporal version — treat historical corpus data as current

## LangGraph loop design
The agent runs a cycle: Generate attack → Run through RAG → Judge response → 
Record result → Generate next attack (conditioned on previous failures).
Cycle detection prevents infinite loops if the agent gets stuck.

## PROD SCALE note
At 20,000 docs the attack surface is too large for exhaustive coverage.
Implement importance sampling: bias attack generation toward high-traffic 
query patterns from production logs (Azure Application Insights).

## Interview explanation
"The red-team agent is a LangGraph loop that generates adversarial prompts 
conditioned on previous failures — it gets smarter about where the system 
breaks as it runs, which is fundamentally different from a static test suite."
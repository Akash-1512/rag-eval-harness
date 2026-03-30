# rag-eval-harness

**Production-grade RAG Evaluation and Red-Teaming Platform**

A self-service platform that ingests any real document corpus, runs comprehensive 
RAG evaluation using RAGAS + DeepEval, and autonomously red-teams the system using 
a LangGraph adversarial agent. Every evaluation run is tracked in MLflow.

Built as a portfolio project and teaching resource — every design decision is 
documented inline with production-scale alternatives.

---

## Current State

| Milestone | Status | Description |
|-----------|--------|-------------|
| M1 — Scaffold | ✅ Complete | Repo structure, environment, dependencies |
| M2 — Ingestion | 🔜 Next | Document loading, 4 chunking strategies |
| M3 — Retrieval | ⬜ Planned | FAISS index, hybrid retrieval |
| M4 — RAG pipeline | ⬜ Planned | Query → retrieve → GPT-4o |
| M5 — RAGAS | ⬜ Planned | 5 core evaluation metrics |
| M6 — DeepEval | ⬜ Planned | G-Eval, custom metrics |
| M7 — Red-Team | ⬜ Planned | LangGraph adversarial agent |
| M8 — MLflow | ⬜ Planned | Experiment tracking |
| M9 — FastAPI | ⬜ Planned | REST API backend |
| M10 — Streamlit | ⬜ Planned | Evaluation dashboard |

---

## Corpus

20 foundational AI/ML research papers (~780 pages total):
- Transformers, BERT, GPT-3, LLaMA 1/2, RAG, Self-RAG, RAGAS, and more
- All sourced from arXiv — publicly available, no IP concerns
- Stored in `data/papers/` as numbered PDFs

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Azure OpenAI GPT-4o |
| Embeddings | text-embedding-3-large |
| Vector store | FAISS (local) → Azure AI Search (prod) |
| RAG framework | LangChain + LangGraph |
| Evaluation | RAGAS + DeepEval + custom metrics |
| Tracking | MLflow (local) → Azure ML (prod) |
| API | FastAPI |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |

---

## Setup
```bat
git clone https://github.com/Akash-1512/rag-eval-harness.git
cd rag-eval-harness
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Fill in your Azure OpenAI credentials in .env
```

---

## Repository Structure
```
rag-eval-harness/
  ingestion/           # Document loading, 4 chunking strategies
  retrieval/           # Vector index and hybrid retrieval
  evaluation/
    ragas_pipeline/    # RAGAS 5-metric evaluation
    deepeval_tests/    # DeepEval assertions and G-Eval
    custom_metrics/    # Abstention accuracy, numerical faithfulness
  red_team/            # LangGraph adversarial agent
  tracking/            # MLflow experiment configuration
  ui/                  # Streamlit dashboard
  api/                 # FastAPI backend
  data/
    papers/            # 20 AI/ML research PDFs
    qa_pairs/          # Real Q&A CSV files
  notebooks/           # Benchmark analysis
  .github/workflows/   # CI/CD pipelines
```

---

## Teaching Philosophy

This repo is built as both a working system and a learning resource.
Every folder contains a `NOTES.md` explaining:
- Why this design decision was made
- What breaks if you do it the naive way  
- How this connects to the rest of the system
- What changes at production scale (20,000+ documents)

---

## Downloading the corpus

PDFs are not stored in this repository (binary files, ~780 pages total).
Download each paper from arXiv and place in `data/papers/`:

| File | arXiv URL |
|------|-----------|
| 01_attention_is_all_you_need.pdf | https://arxiv.org/abs/1706.03762 |
| 02_bert.pdf | https://arxiv.org/abs/1810.04805 |
| 03_gpt3.pdf | https://arxiv.org/abs/2005.14165 |
| 04_llama.pdf | https://arxiv.org/abs/2302.13971 |
| 05_llama2.pdf | https://arxiv.org/abs/2307.09288 |
| 06_rag.pdf | https://arxiv.org/abs/2005.11401 |
| 07_self_rag.pdf | https://arxiv.org/abs/2310.11511 |
| 08_ragas.pdf | https://arxiv.org/abs/2309.15217 |
| 09_instructgpt_rlhf.pdf | https://arxiv.org/abs/2203.02155 |
| 10_constitutional_ai.pdf | https://arxiv.org/abs/2212.08073 |
| 11_chain_of_thought_prompting.pdf | https://arxiv.org/abs/2201.11903 |
| 12_react.pdf | https://arxiv.org/abs/2210.03629 |
| 13_hyde.pdf | https://arxiv.org/abs/2212.10496 |
| 14_lost_in_the_middle.pdf | https://arxiv.org/abs/2307.03172 |
| 15_mixtral_of_experts.pdf | https://arxiv.org/abs/2401.04088 |
| 16_mistral_7b.pdf | https://arxiv.org/abs/2310.06825 |
| 17_flare.pdf | https://arxiv.org/abs/2305.06983 |
| 18_toolformer.pdf | https://arxiv.org/abs/2302.04761 |
| 19_sparks_of_agi_gpt4_technical_report.pdf | https://arxiv.org/abs/2303.12528 |
| 20_rag_survey.pdf | https://arxiv.org/abs/2312.10997 |

*Built by Akash Chaudhari — Agentic AI Engineer, Deloitte*
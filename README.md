# rag-eval-harness

> Production-grade RAG Evaluation and Red-Teaming Platform — built on 20 foundational AI/ML research papers.

Most RAG systems are shipped without systematic evaluation. This project builds the infrastructure that production teams skip — a self-service harness that ingests any document corpus, scores it across five RAGAS metrics, runs DeepEval G-Eval assertions, autonomously red-teams the system with a LangGraph agent, and tracks every experiment in MLflow.

**Status: v1.0.0 — Complete**

---

## Live Results

Evaluated against 10 real Q&A pairs from 20 foundational AI/ML research papers.

| Metric | Score | Framework |
|--------|-------|-----------|
| Faithfulness | 0.933 | RAGAS |
| Context Precision | 0.778 | RAGAS |
| Context Recall | 1.000 | RAGAS |
| Answer Correctness | 0.735 | RAGAS |
| Faithfulness pass rate | 0.667 | DeepEval G-Eval |
| Answer relevancy pass rate | 1.000 | DeepEval G-Eval |
| Abstention accuracy | 1.000 | Custom LLM-as-Judge |
| Red-team failure rate | 0.333 | LangGraph Agent |

**Key finding:** The temporal version attack caught a genuine RAG failure — the system presented 2023 MMLU benchmark numbers as current state-of-the-art with no temporal caveat. This failure mode is invisible to RAGAS and DeepEval but visible to the red-team agent. That is the point.

---

## What It Does

Upload any PDF corpus and a set of question-answer pairs. The system:

1. Ingests documents using four configurable chunking strategies
2. Embeds and indexes with FAISS (local) or Azure AI Search (production)
3. Generates answers via Groq LLM (demo) or Azure OpenAI GPT-4o (production)
4. Scores across five RAGAS metrics — faithfulness, context precision, context recall, answer relevance, answer correctness
5. Runs DeepEval G-Eval assertions with custom natural language rubrics
6. Measures abstention accuracy — whether the system correctly refuses out-of-scope questions
7. Launches a LangGraph red-team agent across six adversarial attack types
8. Logs every run to MLflow with full parameter and artifact tracking
9. Displays results in a Streamlit dashboard with radar charts and per-question tables

---

## Architecture
```
PDFs (20 papers, ~780 pages)          Q&A pairs (CSV)
        │                                    │
        └──────────────┬─────────────────────┘
                       │
              ┌────────▼────────┐
              │   Ingestion     │
              │  4 strategies   │
              │ fixed·recursive │
              │semantic·hierarch│
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  FAISS Index    │
              │ all-MiniLM-L6   │
              │ [PROD: AzSearch]│
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  RAG Pipeline   │
              │ retrieve→LLM→   │
              │   RAGOutput     │
              └──┬──────┬───┬───┘
                 │      │   │
        ┌────────▼┐ ┌───▼──┐ ┌▼──────────┐
        │  RAGAS  │ │Deep  │ │ Red-Team  │
        │5 metrics│ │Eval  │ │ LangGraph │
        │         │ │G-Eval│ │ 6 attacks │
        └────┬────┘ └──┬───┘ └─────┬─────┘
             └─────────┴───────────┘
                        │
              ┌─────────▼─────────┐
              │      MLflow       │
              │ params·metrics·   │
              │    artifacts      │
              └─────────┬─────────┘
                        │
             ┌──────────▼──────────┐
             │   FastAPI Backend   │◄──── Streamlit Dashboard
             │ /health /evaluate   │      radar charts
             │ /runs /runs/{id}    │      runs table
             └─────────────────────┘      new eval form
```

---

## Tech Stack

| Layer | Demo — zero budget | Production |
|-------|-------------------|------------|
| LLM | Groq llama-3.3-70b-versatile | Azure OpenAI GPT-4o |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | text-embedding-3-large |
| Vector store | FAISS (in-memory) | Azure AI Search hybrid BM25+dense |
| Evaluation | RAGAS 0.2 + DeepEval + custom | Same |
| Agent | LangGraph StateGraph | Same |
| Tracking | MLflow local server | Azure ML workspace |
| API | FastAPI | FastAPI |
| Dashboard | Streamlit | React + FastAPI |
| CI/CD | GitHub Actions | GitHub Actions |

Every paid component has a free alternative running in demo. Switching is a single `.env` change — all prod alternatives are present as commented code tagged `# PROD (paid)`.

---

## Corpus

20 foundational AI/ML research papers (~780 pages) from arXiv.

| # | Paper | Year |
|---|-------|------|
| 01 | Attention Is All You Need | 2017 |
| 02 | BERT | 2018 |
| 03 | GPT-3 | 2020 |
| 04 | LLaMA | 2023 |
| 05 | LLaMA 2 | 2023 |
| 06 | Retrieval-Augmented Generation | 2020 |
| 07 | Self-RAG | 2023 |
| 08 | RAGAS | 2023 |
| 09 | InstructGPT / RLHF | 2022 |
| 10 | Constitutional AI | 2022 |
| 11 | Chain-of-Thought Prompting | 2022 |
| 12 | ReAct | 2022 |
| 13 | HyDE | 2022 |
| 14 | Lost in the Middle | 2023 |
| 15 | Mixtral of Experts | 2024 |
| 16 | Mistral 7B | 2023 |
| 17 | FLARE | 2023 |
| 18 | Toolformer | 2023 |
| 19 | Sparks of AGI (GPT-4) | 2023 |
| 20 | RAG Survey | 2023 |

PDFs are not tracked in git. Download from arXiv into `data/papers/` using filenames `01_attention_is_all_you_need.pdf` through `20_rag_survey.pdf`.

---

## Red-Team Attack Taxonomy

| Attack Type | Target Failure Mode | Result |
|-------------|--------------------|--------------------|
| Version confusion | Hallucinated version attributes | 1.000 failure rate |
| Numerical probe | Fabricated numbers | 0.000 — exact recall working |
| Premise injection | Accepting false premises | 0.000 — rejected correctly |
| Cross-doc synthesis | Fabricated comparisons | 0.000 — abstained correctly |
| Out-of-scope | Failure to abstain | 0.000 — abstained correctly |
| Temporal version | Stale data as current | 1.000 failure rate — genuine bug caught |

---

## Setup
```bat
git clone https://github.com/Akash-1512/rag-eval-harness.git
cd rag-eval-harness
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Set GROQ_API_KEY in .env (free at console.groq.com)
```

Download the 20 papers from arXiv into `data/papers/`.

---

## Running the System
```bat
# Terminal 1 — MLflow
mlflow ui --port 5000

# Terminal 2 — FastAPI
uvicorn api.main:app --port 8000 --reload

# Terminal 3 — Streamlit
streamlit run ui/dashboard.py

# Run evaluation smoke tests individually
python -m ingestion.smoke_test
python -m retrieval.smoke_test
python -m api.smoke_test
python -m evaluation.ragas_pipeline.smoke_test
python -m evaluation.deepeval_tests.smoke_test
python -m red_team.smoke_test
python -m tracking.smoke_test
python -m api.server_smoke_test
```

---

## Repository Structure
```
rag-eval-harness/
  ingestion/
    document_loader.py      # PDF loading, per-page error isolation
    chunker.py              # 4 strategies: fixed, recursive, semantic, hierarchical
    NOTES.md
  retrieval/
    embedder.py             # LocalEmbedder wrapping sentence-transformers
    vector_store.py         # FAISS build/save/load/retrieve
    NOTES.md
  evaluation/
    ragas_pipeline/
      evaluator.py          # 5 RAGAS metrics, RAGEvaluationResult
    deepeval_tests/
      test_suite.py         # G-Eval via GroqDeepEvalLLM wrapper
    custom_metrics/
      abstention_accuracy.py # LLM-as-judge out-of-scope refusal metric
    NOTES.md
  red_team/
    agent.py                # LangGraph StateGraph, RedTeamResult
    attack_types.py         # 6 attack types, 13 adversarial prompts
    judge.py                # Attack-type-specific judge prompts
    NOTES.md
  tracking/
    experiment.py           # MLflow logging — params, metrics, artifacts
  api/
    main.py                 # FastAPI — /health /evaluate /runs
    schemas.py              # Pydantic request/response models
    rag_pipeline.py         # Core RAG: retrieve → LLM → RAGOutput
  ui/
    dashboard.py            # Streamlit — radar charts, runs table, eval form
  data/
    papers/                 # 20 AI/ML PDFs (download from arXiv)
    qa_pairs/
      qa_pairs.csv          # 10 real Q&A pairs
  tests/
    test_ingestion.py       # Unit tests for chunking strategies
  .github/workflows/
    ci.yml                  # GitHub Actions CI
```

---

## Teaching Philosophy

Every folder contains a `NOTES.md` explaining the design decision, what breaks if you do it the naive way, how it connects to the system, and what changes at 20,000 documents. Every paid component has a zero-budget alternative. Every scale change is a single commented block tagged `# PROD SCALE`.

---

## Milestones

| Milestone | Component | Tag |
|-----------|-----------|-----|
| M1 | Scaffold — repo structure, requirements, CI | v0.1.0 |
| M2 | Ingestion — 4 chunking strategies | v0.2.0 |
| M3 | Retrieval — FAISS index | v0.3.0 |
| M4 | RAG pipeline — Groq LLM + RAGOutput | v0.4.0 |
| M5 | RAGAS evaluation — 5 core metrics | v0.5.0 |
| M6 | DeepEval + abstention accuracy | v0.6.0 |
| M7 | LangGraph red-team agent | v0.7.0 |
| M8 | MLflow experiment tracking | v0.8.0 |
| M9 | FastAPI backend | v0.9.0 |
| M10 | Streamlit dashboard | v1.0.0 |

---

*Built by [Akash Chaudhari](https://akashchaudhari.netlify.app) — Agentic AI Engineer, Deloitte South Asia LLP*
*[GitHub](https://github.com/Akash-1512) · [LinkedIn](https://linkedin.com/in/akash1512)*

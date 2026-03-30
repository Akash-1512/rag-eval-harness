# 🚀 rag-eval-harness

> **Production-grade RAG Evaluation & Red-Teaming Platform for LLM Applications**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🧠 Why This Project Matters

Most RAG systems in production are **never evaluated properly**.

They:

* Ship without measuring hallucinations
* Ignore retrieval failures
* Lack adversarial testing
* Have no experiment tracking

👉 This project solves that.

It is a **self-service evaluation harness** that lets teams:

* Quantify RAG performance
* Detect failure modes automatically
* Compare design decisions (chunking, retrieval, etc.)
* Track experiments like real ML systems

---

## ⚡ What It Does

Upload a document corpus + Q&A dataset → get **production-grade evaluation**

### 🔍 Pipeline Overview

1. 📄 Ingest PDFs with 4 chunking strategies
2. 🔎 Perform hybrid retrieval (FAISS / Azure AI Search)
3. 🤖 Generate answers using LLM
4. 📊 Evaluate with **RAGAS (5 metrics)**
5. 🧪 Run **DeepEval + custom metrics**
6. 🛡️ Launch **LangGraph red-team agent (6 attack types)**
7. 📈 Track experiments in **MLflow**
8. 📊 Visualize results in **Streamlit dashboard**

---

## 🏗️ System Architecture

```
PDF Corpus → Chunking → Vector Store → RAG Pipeline
                               │
        ┌───────────────┬───────────────┬───────────────┐
        ▼               ▼               ▼
     RAGAS         DeepEval        Red-Team Agent
   (5 metrics)    (G-Eval)        (LangGraph)
        └───────────────┴───────────────┘
                        ▼
                  MLflow Tracking
                        ▼
                 Streamlit Dashboard
```

---

## 📊 Real Evaluation Insights

| Metric             | Score | Insight              |
| ------------------ | ----- | -------------------- |
| Faithfulness       | 1.000 | No hallucinations    |
| Context Precision  | 0.712 | Some retrieval noise |
| Context Recall     | 0.833 | Good coverage        |
| Answer Correctness | 0.530 | Missing definitions  |

💡 **Key Insight:**
Chunking strategy directly impacts retrieval quality.

* Recursive chunking failed to retrieve split definitions
* MLflow experiments captured this failure quantitatively
* Confirms real-world issue: **retrieval ≠ solved problem**

---

## 🧪 Red-Team Testing (Unique Feature)

Unlike typical RAG systems, this project includes **automated adversarial testing**

### Attack Types

* Version Confusion
* Numerical Hallucination
* Premise Injection
* Cross-Document Contradiction
* Out-of-Scope Queries
* Temporal Errors

👉 The agent **learns from failures and adapts attacks dynamically**

---

## ⚙️ Tech Stack

| Layer      | Demo (Free)           | Production             |
| ---------- | --------------------- | ---------------------- |
| LLM        | Groq LLaMA 3          | Azure OpenAI GPT-4o    |
| Embeddings | MiniLM                | text-embedding-3-large |
| Vector DB  | FAISS                 | Azure AI Search        |
| Framework  | LangChain + LangGraph | Same                   |
| Eval       | RAGAS + DeepEval      | Same                   |
| Tracking   | MLflow local          | Azure ML               |
| UI         | Streamlit             | Streamlit              |

---

## 🧠 Key Design Decisions

### 1. Chunking = Experiment Variable (Not Config)

Each strategy creates its own index → tracked in MLflow
➡️ Enables **quantitative comparison**

---

### 2. Evaluation is Multi-Layered

* RAGAS → Retrieval + generation quality
* DeepEval → LLM-based judgments
* Custom → Abstention accuracy

➡️ Covers **what single metric cannot**

---

### 3. Red-Teaming is First-Class

Most systems:
❌ Evaluate only "happy path"
✅ This system actively **breaks itself**

---

### 4. Zero → Production Switch

* Free stack for demo
* Azure stack for production
* Switch using `.env`

---

## 📂 Repository Structure

```
rag-eval-harness/
├── ingestion/        # Chunking strategies
├── retrieval/        # Embeddings + FAISS
├── evaluation/       # RAGAS + DeepEval
├── red_team/         # LangGraph adversarial agent
├── tracking/         # MLflow experiments
├── api/              # RAG pipeline
├── ui/               # Streamlit dashboard
├── data/             # Papers + QA pairs
└── tests/            # Unit tests
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/Akash-1512/rag-eval-harness.git
cd rag-eval-harness

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
copy .env.example .env
```

### Add your Groq API key

```
GROQ_API_KEY=your_key_here
```

---

## ▶️ Run the System

```bash
# Ingestion
python -m ingestion.smoke_test

# Retrieval
python -m retrieval.smoke_test

# RAG pipeline
python -m api.smoke_test

# Evaluation
python -m evaluation.ragas_pipeline.smoke_test
```

---

## 📚 Dataset

* 20 foundational AI/ML papers (~780 pages)
* 10 real Q&A pairs
* Fully non-synthetic evaluation

---

## 👨‍💻 Author

**Akash Chaudhari**
Agentic AI Engineer @ Deloitte

* 🌐 [https://akashchaudhari.netlify.app](https://akashchaudhari.netlify.app)
* 💼 [https://linkedin.com/in/akash1512](https://linkedin.com/in/akash1512)
* 💻 [https://github.com/Akash-1512](https://github.com/Akash-1512)

---

## ⭐ Support

If you found this useful, consider giving it a star ⭐

"""
ui/dashboard.py

Streamlit dashboard for the RAG Evaluation and Red-Teaming Platform.

Pages:
  1. Overview     — system health, latest run summary, key metrics
  2. Runs         — table of all evaluation runs, sortable/filterable
  3. Run Detail   — deep dive into a single run: radar chart + per-question table
  4. Run Eval     — trigger a new evaluation run from the UI

Usage:
    # Make sure FastAPI and MLflow are running first:
    # Terminal 1: mlflow ui --port 5000
    # Terminal 2: uvicorn api.main:app --port 8000 --reload
    # Terminal 3:
    streamlit run ui/dashboard.py

TEACHING NOTE:
Why Streamlit over a React frontend?
1. Zero JavaScript — entire dashboard in Python
2. Built-in charts, tables, forms — no HTML/CSS needed
3. Fast iteration — changes visible on save without rebuild
4. Perfect for ML dashboards — data scientists can extend it

PROD SCALE:
Replace with a React frontend calling the FastAPI backend.
Streamlit is not suitable for multi-user production deployment —
it runs a single Python process per user, which does not scale.
FastAPI + React scales horizontally; Streamlit does not.
"""

import os
import httpx
import mlflow
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = "http://127.0.0.1:8000"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-eval-harness")

st.set_page_config(
    page_title="RAG Eval Harness",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_health():
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=30)
def fetch_runs(limit: int = 50):
    try:
        r = httpx.get(f"{API_BASE}/runs?limit={limit}", timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


@st.cache_data(ttl=60)
def fetch_run_detail(run_id: str):
    try:
        r = httpx.get(f"{API_BASE}/runs/{run_id}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def make_radar_chart(metrics: dict, title: str = "RAGAS Metrics") -> go.Figure:
    """
    Create a radar/spider chart for RAGAS metrics.
    This is the signature visualisation of the dashboard.
    """
    labels = list(metrics.keys())
    values = [v if v is not None else 0.0 for v in metrics.values()]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        name=title,
        line_color="#4F8EF7",
        fillcolor="rgba(79, 142, 247, 0.2)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=title,
        height=400,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🔬 RAG Eval Harness")
st.sidebar.caption("v1.0.0 — Production-grade RAG Evaluation")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "All Runs", "Run Detail", "New Evaluation"],
)

health = fetch_health()
if health:
    st.sidebar.success(f"✅ API: {API_BASE}")
    st.sidebar.info(f"📊 MLflow: {MLFLOW_URI}")
    st.sidebar.metric("Papers available", health.get("papers_available", 0))
else:
    st.sidebar.error("❌ API not reachable")
    st.sidebar.warning("Start: uvicorn api.main:app --port 8000")


# ── Page: Overview ────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("RAG Evaluation Dashboard")
    st.markdown(
        "Production-grade RAG evaluation across **5 RAGAS metrics**, "
        "**DeepEval G-Eval assertions**, **custom abstention accuracy**, "
        "and **LangGraph red-team attacks**."
    )

    if not health:
        st.error("API server not running. Start with: `uvicorn api.main:app --port 8000`")
        st.stop()

    runs = fetch_runs(limit=10)
    if not runs:
        st.info("No evaluation runs yet. Go to **New Evaluation** to run your first evaluation.")
        st.stop()

    latest = runs[0]

    # Top metrics row
    st.subheader("Latest Run — " + latest["run_name"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Faithfulness", f"{latest.get('ragas_faithfulness', 0):.3f}" if latest.get('ragas_faithfulness') else "—")
    col2.metric("Context Precision", f"{latest.get('ragas_context_precision', 0):.3f}" if latest.get('ragas_context_precision') else "—")
    col3.metric("Abstention Accuracy", f"{latest.get('abstention_accuracy', 0):.3f}" if latest.get('abstention_accuracy') else "—")
    col4.metric("Strategy", latest.get("chunking_strategy", "—"))

    # Radar chart from latest run
    detail = fetch_run_detail(latest["run_id"])
    if detail:
        metrics_map = {
            "Faithfulness": detail["metrics"].get("ragas_faithfulness"),
            "Context Precision": detail["metrics"].get("ragas_context_precision"),
            "Context Recall": detail["metrics"].get("ragas_context_recall"),
            "Answer Correctness": detail["metrics"].get("ragas_answer_correctness"),
            "Abstention": detail["metrics"].get("abstention_accuracy"),
        }
        metrics_map = {k: v for k, v in metrics_map.items() if v is not None}

        if metrics_map:
            col_chart, col_info = st.columns([2, 1])
            with col_chart:
                fig = make_radar_chart(metrics_map, title=f"RAGAS Profile — {latest['run_name']}")
                st.plotly_chart(fig, use_container_width=True)
            with col_info:
                st.subheader("Run Config")
                params = detail.get("params", {})
                for k, v in params.items():
                    st.text(f"{k}: {v}")

    # Recent runs table
    st.subheader("Recent Runs")
    df = pd.DataFrame(runs)
    display_cols = [c for c in [
        "run_name", "chunking_strategy", "llm_model",
        "ragas_faithfulness", "ragas_context_precision", "abstention_accuracy"
    ] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)


# ── Page: All Runs ────────────────────────────────────────────────────────────

elif page == "All Runs":
    st.title("All Evaluation Runs")

    runs = fetch_runs(limit=50)
    if not runs:
        st.info("No runs found. Run an evaluation first.")
        st.stop()

    df = pd.DataFrame(runs)

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        strategies = ["All"] + sorted(df["chunking_strategy"].dropna().unique().tolist())
        selected_strategy = st.selectbox("Filter by chunking strategy", strategies)
    with col2:
        sort_col = st.selectbox(
            "Sort by",
            ["start_time", "ragas_faithfulness", "ragas_context_precision", "abstention_accuracy"]
        )

    if selected_strategy != "All":
        df = df[df["chunking_strategy"] == selected_strategy]

    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position="last")

    display_cols = [c for c in [
        "run_name", "status", "chunking_strategy", "llm_model", "top_k",
        "ragas_faithfulness", "ragas_context_precision", "abstention_accuracy", "start_time"
    ] if c in df.columns]

    st.dataframe(df[display_cols], use_container_width=True, height=500)
    st.caption(f"Showing {len(df)} runs")


# ── Page: Run Detail ──────────────────────────────────────────────────────────

elif page == "Run Detail":
    st.title("Run Detail")

    runs = fetch_runs(limit=50)
    if not runs:
        st.info("No runs found.")
        st.stop()

    run_names = [f"{r['run_name']} ({r['run_id'][:8]})" for r in runs]
    selected = st.selectbox("Select run", run_names)
    selected_idx = run_names.index(selected)
    selected_run = runs[selected_idx]

    detail = fetch_run_detail(selected_run["run_id"])
    if not detail:
        st.error("Could not load run details")
        st.stop()

    st.subheader(f"Run: {detail['run_name']}")
    st.caption(f"Run ID: {detail['run_id']} | Status: {detail['status']}")

    # Params and metrics side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Parameters")
        params_df = pd.DataFrame(
            list(detail["params"].items()),
            columns=["Parameter", "Value"]
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Metrics")
        metrics = detail.get("metrics", {})
        metrics_df = pd.DataFrame(
            [(k, f"{v:.4f}") for k, v in sorted(metrics.items())],
            columns=["Metric", "Value"]
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Radar chart
    ragas_keys = {
        "Faithfulness": metrics.get("ragas_faithfulness"),
        "Context Precision": metrics.get("ragas_context_precision"),
        "Context Recall": metrics.get("ragas_context_recall"),
        "Answer Correctness": metrics.get("ragas_answer_correctness"),
        "Abstention": metrics.get("abstention_accuracy"),
    }
    ragas_keys = {k: v for k, v in ragas_keys.items() if v is not None}

    if ragas_keys:
        st.subheader("Radar Chart")
        fig = make_radar_chart(ragas_keys, title=detail["run_name"])
        st.plotly_chart(fig, use_container_width=True)

    # MLflow link
    st.markdown(
        f"🔗 [View full run in MLflow]({MLFLOW_URI}/#/experiments)"
    )


# ── Page: New Evaluation ──────────────────────────────────────────────────────

elif page == "New Evaluation":
    st.title("Trigger New Evaluation")
    st.info(
        "This triggers a full evaluation run: ingest → retrieve → RAG → "
        "RAGAS → DeepEval → abstention → MLflow. Takes 10-20 minutes."
    )

    with st.form("eval_form"):
        col1, col2 = st.columns(2)
        with col1:
            strategy = st.selectbox(
                "Chunking strategy",
                ["recursive", "fixed", "semantic", "hierarchical"]
            )
            top_k = st.slider("top_k (chunks retrieved)", 1, 10, 5)
        with col2:
            run_ragas = st.checkbox("Run RAGAS (5 metrics)", value=True)
            run_deepeval = st.checkbox("Run DeepEval assertions", value=True)
            run_abstention = st.checkbox("Run abstention accuracy", value=True)
            run_redteam = st.checkbox("Run red-team agent (slow)", value=False)

        notes = st.text_input("Notes (optional)", placeholder="e.g. testing semantic chunking")
        submitted = st.form_submit_button("🚀 Start Evaluation")

    if submitted:
        if not health:
            st.error("API not running. Start with: uvicorn api.main:app --port 8000")
        else:
            with st.spinner("Running evaluation pipeline... (this takes 10-20 minutes)"):
                try:
                    payload = {
                        "chunking_strategy": strategy,
                        "top_k": top_k,
                        "run_ragas": run_ragas,
                        "run_deepeval": run_deepeval,
                        "run_abstention": run_abstention,
                        "run_redteam": run_redteam,
                        "notes": notes,
                    }
                    r = httpx.post(
                        f"{API_BASE}/evaluate",
                        json=payload,
                        timeout=1800,  # 30 min max
                    )
                    if r.status_code == 200:
                        result = r.json()
                        st.success(f"✅ Evaluation complete! Run ID: {result['run_id'][:8]}")
                        if result.get("ragas"):
                            ragas = result["ragas"]
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Faithfulness", f"{ragas.get('faithfulness', 0):.3f}" if ragas.get('faithfulness') else "—")
                            col2.metric("Context Recall", f"{ragas.get('context_recall', 0):.3f}" if ragas.get('context_recall') else "—")
                            col3.metric("Answer Correctness", f"{ragas.get('answer_correctness', 0):.3f}" if ragas.get('answer_correctness') else "—")
                        st.info(f"View in MLflow: {MLFLOW_URI}")
                        st.cache_data.clear()
                    else:
                        st.error(f"Evaluation failed: {r.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

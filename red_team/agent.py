"""
red_team/agent.py

LangGraph red-team agent — generates adversarial prompts, runs them
through the RAG pipeline, judges responses, and produces failure rates.

TEACHING NOTE:
This is a LangGraph StateGraph with a simple loop:
  generate_attack → run_rag → judge_response → record → [next or stop]

Why LangGraph instead of a simple for loop?
1. State management — the agent tracks all previous results in state
2. Cycle detection — built-in stop condition prevents infinite loops
3. Conditional routing — can route to different attack types based on
   previous failure patterns (future enhancement)
4. Observability — LangGraph traces show the full agent decision path

The agent does NOT currently adapt its attack strategy based on failures
(that would require a planning node). It runs all pre-defined attacks in
sequence. The adaptive version is noted as a PROD enhancement.

PROD SCALE (20,000 docs / 800K pages):
# Add a planning node that:
# 1. Reads production query logs (Azure App Insights)
# 2. Identifies high-traffic query patterns
# 3. Generates targeted attacks based on real user behavior
# This makes the red-team agent production-aware rather than synthetic.
"""

from dataclasses import dataclass, field
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, StateGraph
from loguru import logger

from red_team.attack_types import ALL_ATTACKS, AttackPrompt
from red_team.judge import judge_response

load_dotenv()


# ── Agent state ───────────────────────────────────────────────────────────────

class RedTeamState(TypedDict):
    """
    LangGraph state — passed between every node in the graph.
    All results accumulate here across the attack loop.
    """
    attacks: list[AttackPrompt]           # queue of attacks to run
    current_attack_idx: int               # index into attacks list
    results: list[dict]                   # completed attack results
    vector_store: object                  # FAISS index (not serialised)
    max_attacks: int                      # stop condition


@dataclass
class RedTeamResult:
    """
    Final result from the red-team agent run.
    Consumed by MLflow (M8) and Streamlit dashboard (M10).
    """
    total_attacks: int = 0
    total_failures: int = 0
    failure_rate: float = 0.0

    # Per-attack-type breakdown
    by_attack_type: dict = field(default_factory=dict)

    # Full results for Streamlit drill-down
    attack_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Flat dict for MLflow logging."""
        d = {
            "redteam_total_attacks": self.total_attacks,
            "redteam_total_failures": self.total_failures,
            "redteam_failure_rate": self.failure_rate,
        }
        for attack_type, stats in self.by_attack_type.items():
            d[f"redteam_{attack_type}_failure_rate"] = stats.get("failure_rate", 0.0)
        return d

    def summary(self) -> str:
        lines = [
            f"Red-Team Results ({self.total_attacks} attacks)",
            f"  Overall failure rate: {self.failure_rate:.3f} "
            f"({self.total_failures}/{self.total_attacks} attacks caused failures)",
        ]
        for attack_type, stats in self.by_attack_type.items():
            rate = stats.get("failure_rate", 0.0)
            n = stats.get("total", 0)
            f_count = stats.get("failures", 0)
            lines.append(f"  {attack_type}: {rate:.3f} ({f_count}/{n})")
        return "\n".join(lines)


# ── LangGraph nodes ───────────────────────────────────────────────────────────

def run_attack_node(state: RedTeamState) -> RedTeamState:
    """
    Node: Run the current attack through the RAG pipeline and judge it.
    Updates state with the result and advances the attack index.
    """
    from api.rag_pipeline import run_rag

    idx = state["current_attack_idx"]
    attacks = state["attacks"]

    if idx >= len(attacks):
        return state

    attack = attacks[idx]
    logger.info(
        f"  Attack [{idx + 1}/{len(attacks)}] "
        f"[{attack.attack_type.value}]: {attack.question[:60]}..."
    )

    # Run through RAG pipeline
    try:
        rag_output = run_rag(
            question=attack.question,
            vector_store=state["vector_store"],
            top_k=5,
        )
        response = rag_output.answer
    except Exception as e:
        logger.error(f"    RAG pipeline failed: {e}")
        response = f"ERROR: {e}"

    # Judge the response
    passed, reason = judge_response(attack, response)
    failed = not passed

    status = "FAIL ✗" if failed else "PASS ✓"
    logger.info(f"    {status} — {reason[:80]}")

    result = {
        "question": attack.question,
        "attack_type": attack.attack_type.value,
        "response": response[:300],
        "passed": passed,
        "failed": failed,
        "reason": reason,
        "expected_behavior": attack.ground_truth_behavior,
    }

    # Immutable state update pattern for LangGraph
    new_results = state["results"] + [result]

    return {
        **state,
        "current_attack_idx": idx + 1,
        "results": new_results,
    }


def should_continue(state: RedTeamState) -> str:
    """
    Conditional edge: continue attacking or stop.
    Stops when all attacks are exhausted OR max_attacks reached.
    """
    idx = state["current_attack_idx"]
    attacks = state["attacks"]
    max_attacks = state.get("max_attacks", len(attacks))

    if idx >= len(attacks) or idx >= max_attacks:
        return "end"
    return "continue"


def build_red_team_graph() -> StateGraph:
    """
    Build and compile the LangGraph red-team agent.

    Graph structure:
        run_attack → [should_continue] → run_attack (loop)
                                       → END
    """
    graph = StateGraph(RedTeamState)
    graph.add_node("run_attack", run_attack_node)

    graph.set_entry_point("run_attack")
    graph.add_conditional_edges(
        "run_attack",
        should_continue,
        {
            "continue": "run_attack",
            "end": END,
        },
    )

    return graph.compile()


# ── Public interface ──────────────────────────────────────────────────────────

def run_red_team_agent(
    vector_store: FAISS,
    attacks: list[AttackPrompt] = None,
    max_attacks: int = None,
) -> RedTeamResult:
    """
    Run the full red-team agent against the RAG pipeline.

    Args:
        vector_store: Loaded FAISS index
        attacks: List of AttackPrompts to run. Defaults to ALL_ATTACKS.
        max_attacks: Limit number of attacks (useful for smoke testing)

    Returns:
        RedTeamResult with failure rates per attack type
    """
    if attacks is None:
        attacks = ALL_ATTACKS

    if max_attacks is not None:
        attacks = attacks[:max_attacks]

    logger.info(
        f"Starting red-team agent: {len(attacks)} attacks across "
        f"{len(set(a.attack_type for a in attacks))} attack types"
    )

    graph = build_red_team_graph()

    initial_state: RedTeamState = {
        "attacks": attacks,
        "current_attack_idx": 0,
        "results": [],
        "vector_store": vector_store,
        "max_attacks": len(attacks),
    }

    final_state = graph.invoke(initial_state)
    results = final_state["results"]

    # Aggregate by attack type
    by_type: dict[str, dict] = {}
    for r in results:
        at = r["attack_type"]
        if at not in by_type:
            by_type[at] = {"total": 0, "failures": 0, "failure_rate": 0.0}
        by_type[at]["total"] += 1
        if r["failed"]:
            by_type[at]["failures"] += 1

    for at in by_type:
        t = by_type[at]["total"]
        f = by_type[at]["failures"]
        by_type[at]["failure_rate"] = f / t if t > 0 else 0.0

    total_failures = sum(1 for r in results if r["failed"])
    failure_rate = total_failures / len(results) if results else 0.0

    result = RedTeamResult(
        total_attacks=len(results),
        total_failures=total_failures,
        failure_rate=failure_rate,
        by_attack_type=by_type,
        attack_results=results,
    )

    logger.success(f"\n{result.summary()}")
    return result

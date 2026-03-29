"""
Tool-selection evaluation for all three manager routing strategies.

Tests:
  1. Keyword/fallback routing (deterministic, no LLM needed)
  2. Full LLM-based routing (requires running Ollama)

Run from the healthcare-agent directory:
    python -m pytest tests/test_tool_selection.py -v
    python -m pytest tests/test_tool_selection.py -v -k keyword    # fast, no LLM
    python -m pytest tests/test_tool_selection.py -v -k llm        # needs Ollama
"""

import csv
import os
import re
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

# Max seconds per single LLM query before it's marked as timed-out
LLM_QUERY_TIMEOUT = int(os.getenv("LLM_QUERY_TIMEOUT", "180"))


class _QueryTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _QueryTimeout()

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_cases import TEST_CASES

RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Category normalisation helpers
# ──────────────────────────────────────────────────────────────────────

# PlannedAgentManager tool → category
PLAN_TOOL_TO_CATEGORY = {
    "chat": "CHAT",
    "rag_search": "RAG",
    "list_slots": "ACTION",
    "list_appointments": "ACTION",
    "book_appointment": "ACTION",
    "cancel_appointment": "ACTION",
    "reschedule_appointment": "ACTION",
    "send_doctor_message": "ACTION",
    "respond": None,  # meta-tool, skip
}

# MultiAgentManager agent → category
AGENT_TO_CATEGORY = {
    "chat_agent": "CHAT",
    "medical_knowledge_agent": "RAG",
    "appointment_agent": "ACTION",
    "messaging_agent": "ACTION",
}

# LangGraph tool name → category (for AgentManager & AgentManagerMultiTools)
LANGGRAPH_TOOL_TO_CATEGORY = {
    "RAG": "RAG",
    "LLM": "CHAT",
    "LLM_ONLY": "CHAT",
    "ACTION": "ACTION",
    "LIST_SLOTS": "ACTION",
    "LIST_APPOINTMENTS": "ACTION",
    "BOOK_APPOINTMENT": "ACTION",
    "CANCEL_APPOINTMENT": "ACTION",
    "RESCHEDULE_APPOINTMENT": "ACTION",
    "SEND_DOCTOR_MESSAGE": "ACTION",
}


def _category_from_plan(plan_dict: dict) -> str:
    """Extract the primary category from a PlannedAgentManager plan."""
    steps = plan_dict.get("steps", [])
    for step in steps:
        tool = step.get("tool", "")
        cat = PLAN_TOOL_TO_CATEGORY.get(tool)
        if cat is not None:
            return cat
    return "CHAT"


def _category_from_agent(delegation: dict) -> str:
    """Extract the category from a MultiAgentManager routing decision."""
    agent = delegation.get("agent", "chat_agent")
    return AGENT_TO_CATEGORY.get(agent, "CHAT")


# ──────────────────────────────────────────────────────────────────────
# Result collection & metrics
# ──────────────────────────────────────────────────────────────────────

class ResultCollector:
    """Accumulates per-test results and computes precision metrics."""

    def __init__(self, manager_name: str):
        self.manager_name = manager_name
        self.rows: List[Dict] = []

    def add(self, message: str, expected: str, predicted: str, time_s: float):
        self.rows.append({
            "message": message,
            "expected": expected,
            "predicted": predicted,
            "correct": expected == predicted,
            "time_s": round(time_s, 4),
        })

    def save_csv(self, filename: str):
        path = RESULTS_DIR / filename
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["message", "expected", "predicted", "correct", "time_s"])
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def compute_metrics(self) -> Dict:
        """Compute overall accuracy and per-category precision."""
        if not self.rows:
            return {}

        categories = {"CHAT", "RAG", "ACTION"}
        tp = defaultdict(int)  # true positives per category
        fp = defaultdict(int)  # false positives per category
        fn = defaultdict(int)  # false negatives per category

        total_correct = 0
        total = len(self.rows)

        for r in self.rows:
            exp, pred = r["expected"], r["predicted"]
            if exp == pred:
                total_correct += 1
                tp[exp] += 1
            else:
                fn[exp] += 1
                fp[pred] += 1

        precision = {}
        recall = {}
        for cat in categories:
            denom_p = tp[cat] + fp[cat]
            denom_r = tp[cat] + fn[cat]
            precision[cat] = round(tp[cat] / denom_p, 4) if denom_p else 0.0
            recall[cat] = round(tp[cat] / denom_r, 4) if denom_r else 0.0

        total_time = sum(r["time_s"] for r in self.rows)
        avg_time = round(total_time / total, 4) if total else 0.0

        return {
            "manager": self.manager_name,
            "total": total,
            "correct": total_correct,
            "accuracy": round(total_correct / total, 4) if total else 0.0,
            "precision": precision,
            "recall": recall,
            "total_time_s": round(total_time, 4),
            "avg_time_s": avg_time,
        }

    def print_report(self, metrics: Dict):
        m = metrics
        print(f"\n{'=' * 60}")
        print(f"  {m['manager']}")
        print(f"{'=' * 60}")
        print(f"  Accuracy:   {m['correct']}/{m['total']}  =  {m['accuracy']:.2%}")
        print(f"  Total time: {m['total_time_s']:.2f}s   Avg: {m['avg_time_s']:.4f}s")
        print(f"  {'Category':<10} {'Precision':>10} {'Recall':>10}")
        print(f"  {'-' * 30}")
        for cat in ("CHAT", "RAG", "ACTION"):
            p = m["precision"].get(cat, 0)
            r = m["recall"].get(cat, 0)
            print(f"  {cat:<10} {p:>10.2%} {r:>10.2%}")
        print()

    def save_metrics(self, metrics: Dict, filename: str):
        path = RESULTS_DIR / filename
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["manager", metrics["manager"]])
            writer.writerow(["total", metrics["total"]])
            writer.writerow(["correct", metrics["correct"]])
            writer.writerow(["accuracy", metrics["accuracy"]])
            writer.writerow(["total_time_s", metrics["total_time_s"]])
            writer.writerow(["avg_time_s", metrics["avg_time_s"]])
            for cat in ("CHAT", "RAG", "ACTION"):
                writer.writerow([f"precision_{cat}", metrics["precision"].get(cat, 0)])
                writer.writerow([f"recall_{cat}", metrics["recall"].get(cat, 0)])
        return path


# ======================================================================
# 1. KEYWORD / FALLBACK routing tests  (no LLM required)
# ======================================================================

class TestKeywordRouting:
    """Test the deterministic keyword-based fallback in each manager."""

    # ── DeterministicAgentManager._fallback_route ────────────────────
    def test_keyword_deterministic(self):
        from orchestration.manager_deterministic import DeterministicAgentManager

        mgr = DeterministicAgentManager.__new__(DeterministicAgentManager)
        # Minimal init – only need _fallback_route which has no deps
        collector = ResultCollector("DeterministicAgent_keyword")

        for message, expected in TEST_CASES:
            t0 = time.perf_counter()
            decision = mgr._fallback_route(message)
            elapsed = time.perf_counter() - t0
            predicted = "CHAT" if decision.tool == "LLM" else decision.tool
            collector.add(message, expected, predicted, elapsed)

        csv_path = collector.save_csv("deterministic_keyword_results.csv")
        metrics = collector.compute_metrics()
        collector.print_report(metrics)
        collector.save_metrics(metrics, "deterministic_keyword_metrics.csv")
        assert metrics["accuracy"] > 0, "Some tests should pass"

    # ── PlannedAgentManager._naive_backup_plan ───────────────────────
    def test_keyword_planned(self):
        from orchestration.manager_commands import PlannedAgentManager

        mgr = PlannedAgentManager.__new__(PlannedAgentManager)
        collector = ResultCollector("PlannedAgent_keyword")

        for message, expected in TEST_CASES:
            t0 = time.perf_counter()
            plan = mgr._naive_backup_plan(message)
            elapsed = time.perf_counter() - t0
            predicted = _category_from_plan(plan)
            collector.add(message, expected, predicted, elapsed)

        csv_path = collector.save_csv("planned_keyword_results.csv")
        metrics = collector.compute_metrics()
        collector.print_report(metrics)
        collector.save_metrics(metrics, "planned_keyword_metrics.csv")
        assert metrics["accuracy"] > 0, "Some tests should pass"

    # ── MultiAgentManager._keyword_fallback ──────────────────────────
    def test_keyword_multiagent(self):
        from orchestration.manager_multi_agent import MultiAgentManager

        mgr = MultiAgentManager.__new__(MultiAgentManager)
        collector = ResultCollector("MultiAgent_keyword")

        for message, expected in TEST_CASES:
            t0 = time.perf_counter()
            delegation = mgr._keyword_fallback(message)
            elapsed = time.perf_counter() - t0
            predicted = _category_from_agent(delegation)
            collector.add(message, expected, predicted, elapsed)

        csv_path = collector.save_csv("multiagent_keyword_results.csv")
        metrics = collector.compute_metrics()
        collector.print_report(metrics)
        collector.save_metrics(metrics, "multiagent_keyword_metrics.csv")
        assert metrics["accuracy"] > 0, "Some tests should pass"


# ======================================================================
# 2. LLM-based routing tests  (requires Ollama running)
# ======================================================================

def _ollama_available() -> bool:
    """Check if Ollama is reachable."""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
class TestLLMRouting:
    """Test LLM-driven routing (slower, requires Ollama)."""

    # ── DeterministicAgentManager._decide_tool (LLM path) ───────────
    def test_llm_deterministic(self):
        from orchestration.manager_deterministic import DeterministicAgentManager

        os.environ["KEYWORD_PREROUTE"] = "0"  # force LLM routing
        mgr = DeterministicAgentManager()
        collector = ResultCollector("DeterministicAgent_LLM")

        for i, (message, expected) in enumerate(TEST_CASES, 1):
            print(f"  [{i}/{len(TEST_CASES)}] {message[:50]}...", end=" ", flush=True)
            mgr.memory.history = []  # reset so previous turns don't leak
            t0 = time.perf_counter()
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(LLM_QUERY_TIMEOUT)
                decision = mgr._decide_tool(message, "")
                signal.alarm(0)
                predicted = "CHAT" if decision.tool == "LLM" else decision.tool
            except _QueryTimeout:
                predicted = "TIMEOUT"
            elapsed = time.perf_counter() - t0
            print(f"-> {predicted} ({elapsed:.1f}s)")
            collector.add(message, expected, predicted, elapsed)

        os.environ.pop("KEYWORD_PREROUTE", None)

        csv_path = collector.save_csv("deterministic_llm_results.csv")
        metrics = collector.compute_metrics()
        collector.print_report(metrics)
        collector.save_metrics(metrics, "deterministic_llm_metrics.csv")
        assert metrics["accuracy"] > 0.5, f"LLM routing accuracy too low: {metrics['accuracy']}"

    # ── PlannedAgentManager._plan (LLM planning) ────────────────────
    def test_llm_planned(self):
        from orchestration.manager_commands import PlannedAgentManager

        mgr = PlannedAgentManager()
        collector = ResultCollector("PlannedAgent_LLM")

        for i, (message, expected) in enumerate(TEST_CASES, 1):
            print(f"  [{i}/{len(TEST_CASES)}] {message[:50]}...", end=" ", flush=True)
            mgr.memory.history = []  # reset so previous turns don't leak
            t0 = time.perf_counter()
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(LLM_QUERY_TIMEOUT)
                plan = mgr._plan(message, "")
                signal.alarm(0)
                predicted = _category_from_plan(plan)
            except _QueryTimeout:
                predicted = "TIMEOUT"
            elapsed = time.perf_counter() - t0
            print(f"-> {predicted} ({elapsed:.1f}s)")
            collector.add(message, expected, predicted, elapsed)

        csv_path = collector.save_csv("planned_llm_results.csv")
        metrics = collector.compute_metrics()
        collector.print_report(metrics)
        collector.save_metrics(metrics, "planned_llm_metrics.csv")
        assert metrics["accuracy"] > 0.5, f"LLM planning accuracy too low: {metrics['accuracy']}"

    # ── MultiAgentManager._route (LLM coordinator) ──────────────────
    def test_llm_multiagent(self):
        from orchestration.manager_multi_agent import MultiAgentManager

        mgr = MultiAgentManager()
        collector = ResultCollector("MultiAgent_LLM")

        for i, (message, expected) in enumerate(TEST_CASES, 1):
            print(f"  [{i}/{len(TEST_CASES)}] {message[:50]}...", end=" ", flush=True)
            mgr.memory.history = []  # reset so previous turns don't leak
            t0 = time.perf_counter()
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(LLM_QUERY_TIMEOUT)
                delegation = mgr._route(message, "")
                signal.alarm(0)
                predicted = _category_from_agent(delegation)
            except _QueryTimeout:
                predicted = "TIMEOUT"
            elapsed = time.perf_counter() - t0
            print(f"-> {predicted} ({elapsed:.1f}s)")
            collector.add(message, expected, predicted, elapsed)

        csv_path = collector.save_csv("multiagent_llm_results.csv")
        metrics = collector.compute_metrics()
        collector.print_report(metrics)
        collector.save_metrics(metrics, "multiagent_llm_metrics.csv")
        assert metrics["accuracy"] > 0.5, f"LLM routing accuracy too low: {metrics['accuracy']}"

    # ── AgentManager / basic tools (LangGraph, 3 tools) ─────────────
    def test_llm_basic_tools(self):
        from orchestration.manager_basic_tools import AgentManager

        mgr = AgentManager()
        collector = ResultCollector("BasicTools_LLM")

        for i, (message, expected) in enumerate(TEST_CASES, 1):
            print(f"  [{i}/{len(TEST_CASES)}] {message[:50]}...", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(LLM_QUERY_TIMEOUT)
                mgr._last_tool_used = None
                mgr._current_history_text = ""
                _response, _intent = mgr._route_and_respond(message, f"test_basic_{i}")
                signal.alarm(0)
                tool_used = (mgr._last_tool_used or "LLM").upper()
                predicted = LANGGRAPH_TOOL_TO_CATEGORY.get(tool_used, "CHAT")
            except _QueryTimeout:
                tool_used = "TIMEOUT"
                _response = None
                predicted = "TIMEOUT"
            except Exception as e:
                signal.alarm(0)
                tool_used = "ERROR"
                _response = None
                predicted = "ERROR"
                print(f"\n    ERROR: {type(e).__name__}: {e}")
            elapsed = time.perf_counter() - t0
            print(f"  tool={tool_used} -> {predicted} ({elapsed:.1f}s)")
            if _response:
                print(f"    Answer: {_response}")
            collector.add(message, expected, predicted, elapsed)

        csv_path = collector.save_csv("basic_tools_llm_results.csv")
        metrics = collector.compute_metrics()
        collector.print_report(metrics)
        collector.save_metrics(metrics, "basic_tools_llm_metrics.csv")
        assert metrics["accuracy"] > 0.5, f"LLM routing accuracy too low: {metrics['accuracy']}"

    # ── AgentManagerMultiTools (LangGraph, 8 tools) ──────────────────
    def test_llm_multiple_tools(self):
        from orchestration.manager_multiple_tools import AgentManagerMultiTools

        mgr = AgentManagerMultiTools()
        collector = ResultCollector("MultipleTools_LLM")

        for i, (message, expected) in enumerate(TEST_CASES, 1):
            print(f"  [{i}/{len(TEST_CASES)}] {message[:50]}...", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(LLM_QUERY_TIMEOUT)
                mgr._last_tool_used = None
                mgr._current_history_text = ""
                _response, _intent = mgr._route_and_respond(message, f"test_multi_{i}")
                signal.alarm(0)
                tool_used = (mgr._last_tool_used or "LLM").upper()
                predicted = LANGGRAPH_TOOL_TO_CATEGORY.get(tool_used, "CHAT")
            except _QueryTimeout:
                tool_used = "TIMEOUT"
                _response = None
                predicted = "TIMEOUT"
            except Exception as e:
                signal.alarm(0)
                tool_used = "ERROR"
                _response = None
                predicted = "ERROR"
                print(f"\n    ERROR: {type(e).__name__}: {e}")
            elapsed = time.perf_counter() - t0
            print(f"  tool={tool_used} -> {predicted} ({elapsed:.1f}s)")
            if _response:
                print(f"    Answer: {_response}")
            collector.add(message, expected, predicted, elapsed)

        csv_path = collector.save_csv("multiple_tools_llm_results.csv")
        metrics = collector.compute_metrics()
        collector.print_report(metrics)
        collector.save_metrics(metrics, "multiple_tools_llm_metrics.csv")
        assert metrics["accuracy"] > 0.5, f"LLM routing accuracy too low: {metrics['accuracy']}"


# ======================================================================
# 3. Summary comparison (runs after all tests)
# ======================================================================

def _load_metrics_csv(path: Path) -> Dict:
    """Load a metrics CSV back into a dict."""
    metrics = {}
    if not path.exists():
        return metrics
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) == 2:
                key, val = row
                try:
                    val = float(val)
                except ValueError:
                    pass
                metrics[key] = val
    return metrics


def test_summary_report():
    """Print a comparison table of all completed evaluations."""
    metric_files = sorted(RESULTS_DIR.glob("*_metrics.csv"))
    if not metric_files:
        pytest.skip("No metric files found – run other tests first")

    all_metrics = []
    for mf in metric_files:
        m = _load_metrics_csv(mf)
        if m:
            all_metrics.append(m)

    if not all_metrics:
        pytest.skip("No metrics loaded")

    print(f"\n{'=' * 80}")
    print("  TOOL SELECTION EVALUATION – SUMMARY")
    print(f"{'=' * 80}")
    header = f"  {'Manager':<30} {'Acc':>7} {'P_CHAT':>8} {'P_RAG':>8} {'P_ACT':>8} {'Avg(s)':>8}"
    print(header)
    print(f"  {'-' * 72}")

    for m in all_metrics:
        name = m.get("manager", "?")
        acc = m.get("accuracy", 0)
        pc = m.get("precision_CHAT", 0)
        pr = m.get("precision_RAG", 0)
        pa = m.get("precision_ACTION", 0)
        avg = m.get("avg_time_s", 0)
        print(f"  {name:<30} {acc:>7.2%} {pc:>8.2%} {pr:>8.2%} {pa:>8.2%} {avg:>8.4f}")

    print(f"{'=' * 80}\n")

#!/usr/bin/env python3
"""
Tool selection evaluation for all agent managers.

Runs test cases from test_cases.py against each agent's routing / tool-selection
step, records the selected tool (mapped to standard CHAT / RAG / ACTION
categories), execution time, and whether the selection was correct.

Results are written to a single CSV file.

Usage:
    # From the healthcare-agent/ directory:
    python -m tests.tool_selection_tests.test_tool_selection                 # all agents
    python -m tests.tool_selection_tests.test_tool_selection deterministic   # one agent
    python -m tests.tool_selection_tests.test_tool_selection planned multi_agent  # subset
"""

import argparse
import asyncio
import csv
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable and CWD is correct
# (some modules use relative paths, e.g. chroma_db/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from tests.tool_selection_tests.test_cases import TEST_CASES
from orchestration.memory import ConversationMemory

# ── Standard category mappings per agent ────────────────────────────────────

DETERMINISTIC_MAP = {
    "LIST_SLOTS": "ACTION",
    "LIST_APPOINTMENTS": "ACTION",
    "BOOK_APPOINTMENT": "ACTION",
    "CANCEL_APPOINTMENT": "ACTION",
    "RESCHEDULE_APPOINTMENT": "ACTION",
    "SEND_DOCTOR_MESSAGE": "ACTION",
    "RAG": "RAG",
    "LLM": "CHAT",
}

PLANNED_MAP = {
    "ACTION": "ACTION",
    "RAG": "RAG",
    "LLM_ONLY": "CHAT",
    "UNKNOWN": "CHAT",
}

MULTI_AGENT_MAP = {
    "appointment_agent": "ACTION",
    "messaging_agent": "ACTION",
    "medical_knowledge_agent": "RAG",
    "chat_agent": "CHAT",
}

BASIC_TOOLS_MAP = {
    "ACTION": "ACTION",
    "RAG": "RAG",
    "LLM": "CHAT",
    "LLM_ONLY": "CHAT",
}

MULTI_TOOLS_MAP = {
    "LIST_SLOTS": "ACTION",
    "LIST_APPOINTMENTS": "ACTION",
    "BOOK_APPOINTMENT": "ACTION",
    "CANCEL_APPOINTMENT": "ACTION",
    "RESCHEDULE_APPOINTMENT": "ACTION",
    "SEND_DOCTOR_MESSAGE": "ACTION",
    "RAG": "RAG",
    "LLM": "CHAT",
    "LLM_ONLY": "CHAT",
}

CSV_FIELDS = ["message", "expected", "predicted", "raw_tool", "correct", "time_s"]
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _mock_tools_exec(tools_exec):
    """Replace ToolExecutor methods with fast no-op stubs so tool execution
    is instant and has no side-effects (no real DB / appointment changes)."""
    tools_exec.run_rag_tool = lambda msg, hist="": ("Mock RAG context.", "rag")
    tools_exec.run_llm_tool = lambda msg: "Mock LLM response."
    tools_exec.run_action_tool = lambda msg: "Mock action result."
    tools_exec.run_list_slots = lambda *a: "Mock slots list."
    tools_exec.run_list_appointments = lambda *a: "Mock appointments list."
    tools_exec.run_book_appointment = lambda msg: "Mock booking result."
    tools_exec.run_book_appointment_by_id = lambda sid: f"Booked {sid}."
    tools_exec.run_cancel_appointment = lambda msg: "Mock cancel result."
    tools_exec.run_cancel_appointment_by_id = lambda aid: f"Cancelled {aid}."
    tools_exec.run_reschedule_appointment = lambda msg: "Mock reschedule result."
    tools_exec.run_reschedule_appointment_by_id = lambda aid, sid: f"Rescheduled {aid} to {sid}."
    tools_exec.run_send_doctor_message = lambda msg: "Mock message sent."


def _print_progress(idx, total, message, raw_tool, elapsed):
    print(f"  [{idx:02d}/{total}] {message[:50]:50s} -> {raw_tool:30s} ({elapsed:.2f}s)")


# ── Agent test functions ────────────────────────────────────────────────────

def test_deterministic():
    """DeterministicAgentManager — pure keyword routing, no LLM needed."""
    print("\n[deterministic] Testing DeterministicAgentManager ...")

    # Mock all heavy dependencies; _route() is pure regex logic.
    with patch("orchestration.manager_deterministic.OllamaClient", MagicMock), \
         patch("orchestration.manager_deterministic.Retriever", MagicMock), \
         patch("orchestration.manager_deterministic.AppointmentManager", MagicMock), \
         patch("orchestration.manager_deterministic.MessageManager", MagicMock):
        from orchestration.manager_deterministic import DeterministicAgentManager
        manager = DeterministicAgentManager()

    results = []
    for message, expected in TEST_CASES:
        t0 = time.perf_counter()
        tool, _reason = manager._route(message)
        elapsed = time.perf_counter() - t0
        predicted = DETERMINISTIC_MAP.get(tool, "CHAT")
        results.append({
            "message": message,
            "expected": expected,
            "predicted": predicted,
            "raw_tool": tool,
            "correct": predicted == expected,
            "time_s": round(elapsed, 6),
        })
        _print_progress(len(results), len(TEST_CASES), message, tool, elapsed)

    return results


def test_planned():
    """PlannedAgentManager — test the LLM planning step only (no execution)."""
    print("\n[planned] Testing PlannedAgentManager (LLM planning step) ...")
    from orchestration.manager_commands import PlannedAgentManager
    manager = PlannedAgentManager()

    results = []
    for message, expected in TEST_CASES:
        manager.memory = ConversationMemory()
        t0 = time.perf_counter()
        try:
            plan = manager._plan(message, "")
            plan = manager._validate_plan(plan)
            intent = manager._extract_intent(plan)
        except Exception as e:
            intent = f"ERROR:{e}"
        elapsed = time.perf_counter() - t0
        predicted = PLANNED_MAP.get(intent, "CHAT")
        results.append({
            "message": message,
            "expected": expected,
            "predicted": predicted,
            "raw_tool": intent,
            "correct": predicted == expected,
            "time_s": round(elapsed, 4),
        })
        _print_progress(len(results), len(TEST_CASES), message, intent, elapsed)

    return results


def test_multi_agent():
    """MultiAgentManager — test the coordinator routing step only."""
    print("\n[multi_agent] Testing MultiAgentManager (coordinator step) ...")
    from orchestration.manager_multi_agent import MultiAgentManager
    manager = MultiAgentManager()

    results = []
    for message, expected in TEST_CASES:
        manager.memory = ConversationMemory()
        t0 = time.perf_counter()
        try:
            coordinator_result = manager._run_coordinator(message, "")
            agent_name = coordinator_result.get("agent", "chat_agent")
        except Exception as e:
            agent_name = f"ERROR:{e}"
        elapsed = time.perf_counter() - t0
        predicted = MULTI_AGENT_MAP.get(agent_name, "CHAT")
        results.append({
            "message": message,
            "expected": expected,
            "predicted": predicted,
            "raw_tool": agent_name,
            "correct": predicted == expected,
            "time_s": round(elapsed, 4),
        })
        _print_progress(len(results), len(TEST_CASES), message, agent_name, elapsed)

    return results


def test_basic_tools():
    """AgentManager (basic_tools) — full LangChain agent, mocked tool execution."""
    print("\n[basic_tools] Testing AgentManager (basic_tools, full agent) ...")
    from orchestration.manager_basic_tools import AgentManager
    manager = AgentManager()
    _mock_tools_exec(manager.tools_exec)

    results = []
    for message, expected in TEST_CASES:
        manager.memory = ConversationMemory()
        manager._last_tool_used = None
        t0 = time.perf_counter()
        try:
            result = asyncio.run(manager.process_message(message, "test_user"))
            raw = result["intent"]
        except Exception as e:
            raw = f"ERROR:{e}"
        elapsed = time.perf_counter() - t0
        predicted = BASIC_TOOLS_MAP.get(raw, "CHAT")
        results.append({
            "message": message,
            "expected": expected,
            "predicted": predicted,
            "raw_tool": raw,
            "correct": predicted == expected,
            "time_s": round(elapsed, 4),
        })
        _print_progress(len(results), len(TEST_CASES), message, raw, elapsed)

    return results


def test_multi_tools():
    """AgentManagerMultiTools — full LangChain agent, real tool execution (like routes_chat)."""
    print("\n[multi_tools] Testing AgentManagerMultiTools (full agent) ...")
    from orchestration.manager_multiple_tools import AgentManagerMultiTools
    manager = AgentManagerMultiTools()

    async def _run_all():
        results = []
        for message, expected in TEST_CASES:
            manager.memory = ConversationMemory()
            manager._last_tool_used = None
            t0 = time.perf_counter()
            try:
                result = await manager.process_message(message, "test_user")
                raw = result["intent"]
            except Exception as e:
                raw = f"ERROR:{e}"
            elapsed = time.perf_counter() - t0
            predicted = MULTI_TOOLS_MAP.get(raw, "CHAT")
            results.append({
                "message": message,
                "expected": expected,
                "predicted": predicted,
                "raw_tool": raw,
                "correct": predicted == expected,
                "time_s": round(elapsed, 4),
            })
            _print_progress(len(results), len(TEST_CASES), message, raw, elapsed)
        return results

    return asyncio.run(_run_all())


# ── Registry ────────────────────────────────────────────────────────────────

AGENT_TESTS = {
    "deterministic": test_deterministic,
    "planned": test_planned,
    "multi_agent": test_multi_agent,
    "basic_tools": test_basic_tools,
    "multi_tools": test_multi_tools,
}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tool selection for healthcare agent managers."
    )
    parser.add_argument(
        "agents",
        nargs="*",
        default=list(AGENT_TESTS.keys()),
        choices=list(AGENT_TESTS.keys()),
        help="Which agents to test (default: all).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}  # agent_name -> list of row dicts
    for agent_name in args.agents:
        test_fn = AGENT_TESTS[agent_name]
        try:
            results = test_fn()
            all_results[agent_name] = results
        except Exception as e:
            print(f"  SKIPPED {agent_name}: {e}")

    if not all_results:
        print("\nNo results collected.")
        return

    # Write per-agent CSV files
    for agent_name, results in all_results.items():
        csv_path = OUTPUT_DIR / f"{agent_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(results)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Results saved to {OUTPUT_DIR}/")
    print(f"{'=' * 70}")
    total_correct = 0
    total_count = 0
    for agent_name, results in sorted(all_results.items()):
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        avg_time = sum(r["time_s"] for r in results) / total
        print(f"  {agent_name:20s}  {correct}/{total} correct ({100 * correct / total:5.1f}%)  avg_time={avg_time:.4f}s")
        total_correct += correct
        total_count += total

    print(f"  {'TOTAL':20s}  {total_correct}/{total_count} correct ({100 * total_correct / total_count:5.1f}%)")


if __name__ == "__main__":
    main()

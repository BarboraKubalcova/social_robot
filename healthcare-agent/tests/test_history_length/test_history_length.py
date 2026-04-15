#!/usr/bin/env python3
"""
Tool selection evaluation under varying conversation history lengths.

Tests whether increasing history context (1–4 prior turns) degrades the
agent's ability to select the correct tool for the current user query.

For each history length the test:
  1. Pre-populates ConversationMemory with fake history turns
  2. Sends the current user query via process_message()
  3. Records which tool was selected and whether it matches the expected tool

Results are written to per-agent, per-history-length CSV files in results/.

Usage:
    # From the healthcare-agent/ directory:
    python -m tests.test_history_length.test_history_length                  # all agents, all lengths
    python -m tests.test_history_length.test_history_length multi_tools      # one agent
    python -m tests.test_history_length.test_history_length basic_tools multi_tools  # subset
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
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from orchestration.memory import ConversationMemory

from tests.test_history_length.test_cases_1 import TEST_CASES as CASES_1
from tests.test_history_length.test_case_2 import TEST_CASES as CASES_2
from tests.test_history_length.test_case_3 import TEST_CASES as CASES_3
from tests.test_history_length.test_case_4 import TEST_CASES as CASES_4

CSV_FIELDS = [
    "history_turns", "description", "message",
    "expected", "predicted", "raw_tool", "correct", "time_s",
]
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# Map history-turn count → test cases
HISTORY_CASES = {
    1: CASES_1,
    2: CASES_2,
    3: CASES_3,
    4: CASES_4,
}

# ── Normalisation maps (same as the other test suites) ──────────────────────

MULTI_TOOLS_MAP = {
    "LIST_SLOTS": "LIST_SLOTS",
    "LIST_APPOINTMENTS": "LIST_APPOINTMENTS",
    "BOOK_APPOINTMENT": "BOOK_APPOINTMENT",
    "CANCEL_APPOINTMENT": "CANCEL_APPOINTMENT",
    "RESCHEDULE_APPOINTMENT": "RESCHEDULE_APPOINTMENT",
    "SEND_DOCTOR_MESSAGE": "SEND_DOCTOR_MESSAGE",
    "RAG": "RAG",
    "LLM": "LLM",
}

BASIC_TOOLS_MAP = {
    "ACTION": "ACTION",
    "RAG": "RAG",
    "LLM": "LLM",
    "LLM_ONLY": "LLM",
}

# For basic_tools we collapse the 8 tool names into 3 categories
EXPECTED_TO_BASIC = {
    "LIST_SLOTS": "ACTION",
    "LIST_APPOINTMENTS": "ACTION",
    "BOOK_APPOINTMENT": "ACTION",
    "CANCEL_APPOINTMENT": "ACTION",
    "RESCHEDULE_APPOINTMENT": "ACTION",
    "SEND_DOCTOR_MESSAGE": "ACTION",
    "RAG": "RAG",
    "LLM": "LLM",
}

PLANNED_MAP = {
    "list_slots": "LIST_SLOTS",
    "list_appointments": "LIST_APPOINTMENTS",
    "book_appointment": "BOOK_APPOINTMENT",
    "cancel_appointment": "CANCEL_APPOINTMENT",
    "reschedule_appointment": "RESCHEDULE_APPOINTMENT",
    "send_doctor_message": "SEND_DOCTOR_MESSAGE",
    "rag_search": "RAG",
    "respond": "LLM",
    "chat": "LLM",
    "UNKNOWN": "LLM",
}

MULTI_AGENT_MAP = {
    "appointment_agent": "ACTION",
    "messaging_agent": "ACTION",
    "medical_knowledge_agent": "RAG",
    "chat_agent": "LLM",
}

MULTI_AGENT_TO_BASIC = EXPECTED_TO_BASIC

# For basic_tools: track which ToolExecutor method was called
BASIC_TOOLS_METHOD_MAP = {
    "run_list_slots": "LIST_SLOTS",
    "run_list_appointments": "LIST_APPOINTMENTS",
    "run_book_appointment": "BOOK_APPOINTMENT",
    "run_book_appointment_by_id": "BOOK_APPOINTMENT",
    "run_cancel_appointment": "CANCEL_APPOINTMENT",
    "run_cancel_appointment_by_id": "CANCEL_APPOINTMENT",
    "run_reschedule_appointment": "RESCHEDULE_APPOINTMENT",
    "run_reschedule_appointment_by_id": "RESCHEDULE_APPOINTMENT",
    "run_send_doctor_message": "SEND_DOCTOR_MESSAGE",
    "run_rag_tool": "RAG",
    "run_llm_tool": "LLM",
    "run_action_tool": "ACTION_GENERIC",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _mock_tools_exec(tools_exec):
    """Replace ToolExecutor methods with fast no-op stubs."""
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


def _make_tracking_mock(tracker: dict):
    """Create mock ToolExecutor methods that record which method was called."""
    def _make(name, retval="Mock result."):
        def fn(*args, **kwargs):
            tracker["called"] = name
            return retval
        return fn

    return {
        "run_rag_tool": lambda msg, hist="": (tracker.__setitem__("called", "run_rag_tool") or "", "rag"),
        "run_llm_tool": _make("run_llm_tool", "Mock LLM response."),
        "run_action_tool": _make("run_action_tool", "Mock action result."),
        "run_list_slots": _make("run_list_slots", "Mock slots list."),
        "run_list_appointments": _make("run_list_appointments", "Mock appointments list."),
        "run_book_appointment": _make("run_book_appointment", "Mock booking result."),
        "run_book_appointment_by_id": _make("run_book_appointment_by_id", "Booked."),
        "run_cancel_appointment": _make("run_cancel_appointment", "Mock cancel result."),
        "run_cancel_appointment_by_id": _make("run_cancel_appointment_by_id", "Cancelled."),
        "run_reschedule_appointment": _make("run_reschedule_appointment", "Mock reschedule result."),
        "run_reschedule_appointment_by_id": _make("run_reschedule_appointment_by_id", "Rescheduled."),
        "run_send_doctor_message": _make("run_send_doctor_message", "Mock message sent."),
    }


def _apply_tracking_mock(tools_exec, tracker):
    """Patch ToolExecutor instance methods with tracking wrappers.
    
    Keeps the real run_action_tool so it can dispatch to tracked sub-methods.
    """
    mocks = _make_tracking_mock(tracker)
    # Don't mock run_action_tool — let the real implementation dispatch
    # to the tracked sub-methods (run_list_slots, run_book_appointment, etc.)
    mocks.pop("run_action_tool", None)
    for attr, fn in mocks.items():
        setattr(tools_exec, attr, fn)


def _prefill_memory(memory: ConversationMemory, history: list[dict]):
    """Inject fake history turns into a ConversationMemory instance."""
    for turn in history:
        memory.add_turn(turn["role"], turn["content"])


def _print_progress(idx, total, turns, desc, raw_tool, predicted, expected, elapsed):
    ok = "OK" if predicted == expected else "FAIL"
    print(
        f"  [{idx:02d}/{total}] turns={turns} {desc[:45]:45s} "
        f"-> {str(raw_tool):25s} [{ok}] ({elapsed:.2f}s)"
    )


# ── Agent test functions ────────────────────────────────────────────────────

def test_basic_tools():
    """AgentManager (basic_tools) — 3-tool agent, track specific sub-tool via method calls."""
    print("\n[basic_tools] Testing AgentManager with history ...")
    from orchestration.manager_basic_tools import AgentManager
    manager = AgentManager()

    # Build a deterministic router as fallback for ACTION_GENERIC
    with patch("orchestration.manager_deterministic.OllamaClient", MagicMock), \
         patch("orchestration.manager_deterministic.Retriever", MagicMock), \
         patch("orchestration.manager_deterministic.AppointmentManager", MagicMock), \
         patch("orchestration.manager_deterministic.MessageManager", MagicMock):
        from orchestration.manager_deterministic import DeterministicAgentManager
        det_router = DeterministicAgentManager()

    tracker = {}
    _apply_tracking_mock(manager.tools_exec, tracker)

    all_results = []
    for n_turns, cases in sorted(HISTORY_CASES.items()):
        for tc in cases:
            manager.memory = ConversationMemory()
            manager._last_tool_used = None
            # Allow the agent to see the full fake history
            manager.max_history_turns = n_turns * 2 + 2  # history + current turn
            tracker["called"] = None

            _prefill_memory(manager.memory, tc["history"])

            t0 = time.perf_counter()
            try:
                asyncio.run(manager.process_message(tc["user_query"], "test_user"))
                method_called = tracker.get("called")
                predicted = BASIC_TOOLS_METHOD_MAP.get(method_called, "LLM")
                # If run_action_tool was the generic entry, use deterministic fallback
                if predicted == "ACTION_GENERIC":
                    det_tool, _ = det_router._route(tc["user_query"])
                    predicted = det_tool
            except Exception as e:
                predicted = f"ERROR:{e}"
            elapsed = time.perf_counter() - t0

            expected = tc["expected_tool"]

            all_results.append({
                "history_turns": n_turns,
                "description": tc["description"],
                "message": tc["user_query"],
                "expected": expected,
                "predicted": predicted,
                "raw_tool": tracker.get("called", "NONE"),
                "correct": predicted == expected,
                "time_s": round(elapsed, 4),
            })
            _print_progress(
                len(all_results), sum(len(c) for c in HISTORY_CASES.values()),
                n_turns, tc["description"], tracker.get("called", "NONE"),
                predicted, expected, elapsed,
            )

    return all_results


def test_multi_tools():
    """AgentManagerMultiTools — 8-tool agent."""
    print("\n[multi_tools] Testing AgentManagerMultiTools with history ...")
    from orchestration.manager_multiple_tools import AgentManagerMultiTools
    manager = AgentManagerMultiTools()
    _mock_tools_exec(manager.tools_exec)

    async def _run_all():
        all_results = []
        for n_turns, cases in sorted(HISTORY_CASES.items()):
            for tc in cases:
                manager.memory = ConversationMemory()
                manager._last_tool_used = None
                # Allow the agent to see the full fake history
                manager.max_history_turns = n_turns * 2 + 2

                _prefill_memory(manager.memory, tc["history"])

                t0 = time.perf_counter()
                try:
                    result = await manager.process_message(tc["user_query"], "test_user")
                    raw = result["intent"]
                except Exception as e:
                    raw = f"ERROR:{e}"
                elapsed = time.perf_counter() - t0

                predicted = MULTI_TOOLS_MAP.get(raw, raw)
                expected = tc["expected_tool"]

                all_results.append({
                    "history_turns": n_turns,
                    "description": tc["description"],
                    "message": tc["user_query"],
                    "expected": expected,
                    "predicted": predicted,
                    "raw_tool": raw,
                    "correct": predicted == expected,
                    "time_s": round(elapsed, 4),
                })
                _print_progress(
                    len(all_results), sum(len(c) for c in HISTORY_CASES.values()),
                    n_turns, tc["description"], raw, predicted, expected, elapsed,
                )
        return all_results

    return asyncio.run(_run_all())


def test_planned():
    """PlannedAgentManager — extract the first action tool from the LLM plan."""
    print("\n[planned] Testing PlannedAgentManager with history ...")
    from orchestration.manager_commands import PlannedAgentManager
    manager = PlannedAgentManager()

    all_results = []
    for n_turns, cases in sorted(HISTORY_CASES.items()):
        for tc in cases:
            manager.memory = ConversationMemory()
            _prefill_memory(manager.memory, tc["history"])

            history_text = "\n".join(
                f"{t['role']}: {t['content']}" for t in tc["history"]
            )

            t0 = time.perf_counter()
            try:
                plan = manager._plan(tc["user_query"], history_text)
                plan = manager._validate_plan(plan)
                # Extract the specific tool name from the first step
                # (don't use _extract_intent which collapses to ACTION)
                raw_tool = "UNKNOWN"
                for step in plan.get("steps", []):
                    tool_name = step.get("tool", "")
                    if tool_name not in ("respond", "chat", ""):
                        raw_tool = tool_name
                        break
                if raw_tool == "UNKNOWN" and plan.get("steps"):
                    raw_tool = plan["steps"][0].get("tool", "UNKNOWN")
            except Exception as e:
                raw_tool = f"ERROR:{e}"
            elapsed = time.perf_counter() - t0

            predicted = PLANNED_MAP.get(raw_tool, raw_tool)
            expected = tc["expected_tool"]

            all_results.append({
                "history_turns": n_turns,
                "description": tc["description"],
                "message": tc["user_query"],
                "expected": expected,
                "predicted": predicted,
                "raw_tool": raw_tool,
                "correct": predicted == expected,
                "time_s": round(elapsed, 4),
            })
            _print_progress(
                len(all_results), sum(len(c) for c in HISTORY_CASES.values()),
                n_turns, tc["description"], raw_tool, predicted, expected, elapsed,
            )

    return all_results


def test_multi_agent():
    """MultiAgentManager — coordinator routing step."""
    print("\n[multi_agent] Testing MultiAgentManager with history ...")
    from orchestration.manager_multi_agent import MultiAgentManager
    manager = MultiAgentManager()

    all_results = []
    for n_turns, cases in sorted(HISTORY_CASES.items()):
        for tc in cases:
            manager.memory = ConversationMemory()
            _prefill_memory(manager.memory, tc["history"])

            history_text = "\n".join(
                f"{t['role']}: {t['content']}" for t in tc["history"]
            )

            t0 = time.perf_counter()
            try:
                coordinator_result = manager._run_coordinator(tc["user_query"], history_text)
                agent_name = coordinator_result.get("agent", "chat_agent")
            except Exception as e:
                agent_name = f"ERROR:{e}"
            elapsed = time.perf_counter() - t0

            predicted = MULTI_AGENT_MAP.get(agent_name, "LLM")
            expected = MULTI_AGENT_TO_BASIC.get(tc["expected_tool"], tc["expected_tool"])

            all_results.append({
                "history_turns": n_turns,
                "description": tc["description"],
                "message": tc["user_query"],
                "expected": expected,
                "predicted": predicted,
                "raw_tool": agent_name,
                "correct": predicted == expected,
                "time_s": round(elapsed, 4),
            })
            _print_progress(
                len(all_results), sum(len(c) for c in HISTORY_CASES.values()),
                n_turns, tc["description"], agent_name, predicted, expected, elapsed,
            )

    return all_results


def test_deterministic():
    """DeterministicAgentManager — pure keyword routing (history-agnostic baseline)."""
    print("\n[deterministic] Testing DeterministicAgentManager with history ...")

    with patch("orchestration.manager_deterministic.OllamaClient", MagicMock), \
         patch("orchestration.manager_deterministic.Retriever", MagicMock), \
         patch("orchestration.manager_deterministic.AppointmentManager", MagicMock), \
         patch("orchestration.manager_deterministic.MessageManager", MagicMock):
        from orchestration.manager_deterministic import DeterministicAgentManager
        manager = DeterministicAgentManager()

    DETERMINISTIC_MAP = {
        "LIST_SLOTS": "LIST_SLOTS",
        "LIST_APPOINTMENTS": "LIST_APPOINTMENTS",
        "BOOK_APPOINTMENT": "BOOK_APPOINTMENT",
        "CANCEL_APPOINTMENT": "CANCEL_APPOINTMENT",
        "RESCHEDULE_APPOINTMENT": "RESCHEDULE_APPOINTMENT",
        "SEND_DOCTOR_MESSAGE": "SEND_DOCTOR_MESSAGE",
        "RAG": "RAG",
        "LLM": "LLM",
    }

    all_results = []
    for n_turns, cases in sorted(HISTORY_CASES.items()):
        for tc in cases:
            t0 = time.perf_counter()
            tool, _reason = manager._route(tc["user_query"])
            elapsed = time.perf_counter() - t0

            predicted = DETERMINISTIC_MAP.get(tool, tool)
            expected = tc["expected_tool"]

            all_results.append({
                "history_turns": n_turns,
                "description": tc["description"],
                "message": tc["user_query"],
                "expected": expected,
                "predicted": predicted,
                "raw_tool": tool,
                "correct": predicted == expected,
                "time_s": round(elapsed, 6),
            })
            _print_progress(
                len(all_results), sum(len(c) for c in HISTORY_CASES.values()),
                n_turns, tc["description"], tool, predicted, expected, elapsed,
            )

    return all_results


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
        description="Evaluate tool selection under varying history lengths."
    )
    parser.add_argument(
        "agents",
        nargs="*",
        default=None,
        help="Which agents to test (default: all). Choices: " + ", ".join(AGENT_TESTS.keys()),
    )
    args = parser.parse_args()
    agents_to_test = args.agents if args.agents else list(AGENT_TESTS.keys())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for agent_name in agents_to_test:
        if agent_name not in AGENT_TESTS:
            print(f"  Unknown agent: {agent_name}. Skipping.")
            continue
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

    # Summary — grouped by history length
    print(f"\n{'=' * 80}")
    print(f"Results saved to {OUTPUT_DIR}/")
    print(f"{'=' * 80}")

    for agent_name, results in sorted(all_results.items()):
        print(f"\n  Agent: {agent_name}")
        print(f"  {'turns':>5s}  {'correct':>7s}  {'total':>5s}  {'accuracy':>8s}  {'avg_time':>8s}")
        print(f"  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*8}  {'-'*8}")

        total_correct = 0
        total_count = 0
        for n_turns in sorted(HISTORY_CASES.keys()):
            turn_results = [r for r in results if r["history_turns"] == n_turns]
            correct = sum(1 for r in turn_results if r["correct"])
            total = len(turn_results)
            avg_time = sum(r["time_s"] for r in turn_results) / total if total else 0
            print(f"  {n_turns:5d}  {correct:7d}  {total:5d}  {100*correct/total:7.1f}%  {avg_time:7.4f}s")
            total_correct += correct
            total_count += total

        print(f"  {'ALL':>5s}  {total_correct:7d}  {total_count:5d}  {100*total_correct/total_count:7.1f}%")


if __name__ == "__main__":
    main()

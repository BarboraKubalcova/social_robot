#!/usr/bin/env python3
"""
Typo resilience test for tool selection evaluation.

Tests that each manager routes to the correct *specific* tool when user
messages contain misspellings, character swaps, missing letters, and
keyboard-adjacent errors.

Tools tested:
  LLM, RAG, LIST_SLOTS, LIST_APPOINTMENTS, BOOK_APPOINTMENT,
  CANCEL_APPOINTMENT, RESCHEDULE_APPOINTMENT, SEND_DOCTOR_MESSAGE

Results are written to per-agent CSV files in results/.

Usage:
    python -m tests.test_typos_in_messages.test_typos              # all
    python -m tests.test_typos_in_messages.test_typos deterministic # one
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

from tests.test_typos_in_messages.test_cases import TEST_CASES
from orchestration.memory import ConversationMemory

CSV_FIELDS = ["message", "expected", "predicted", "raw_tool", "correct", "time_s"]
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# ── Normalisation maps ──────────────────────────────────────────────────────

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

PLANNED_MAP = {
    "list_slots": "LIST_SLOTS",
    "list_appointments": "LIST_APPOINTMENTS",
    "book_appointment": "BOOK_APPOINTMENT",
    "cancel_appointment": "CANCEL_APPOINTMENT",
    "reschedule_appointment": "RESCHEDULE_APPOINTMENT",
    "send_doctor_message": "SEND_DOCTOR_MESSAGE",
    "rag": "RAG",
    "retrieve": "RAG",
    "respond": "LLM",
    "chat": "LLM",
}

MULTI_TOOLS_MAP = {
    "LIST_SLOTS": "LIST_SLOTS",
    "LIST_APPOINTMENTS": "LIST_APPOINTMENTS",
    "BOOK_APPOINTMENT": "BOOK_APPOINTMENT",
    "CANCEL_APPOINTMENT": "CANCEL_APPOINTMENT",
    "RESCHEDULE_APPOINTMENT": "RESCHEDULE_APPOINTMENT",
    "SEND_DOCTOR_MESSAGE": "SEND_DOCTOR_MESSAGE",
    "RAG": "RAG",
    "LLM": "LLM",
    "LLM_ONLY": "LLM",
}

MULTI_AGENT_ACTION_MAP = {
    "list_slots": "LIST_SLOTS",
    "list_appointments": "LIST_APPOINTMENTS",
    "book": "BOOK_APPOINTMENT",
    "cancel": "CANCEL_APPOINTMENT",
    "reschedule": "RESCHEDULE_APPOINTMENT",
    "send_message": "SEND_DOCTOR_MESSAGE",
}

MULTI_AGENT_AGENT_MAP = {
    "chat_agent": "LLM",
    "medical_knowledge_agent": "RAG",
}

BASIC_TOOLS_MAP = {
    "LLM": "LLM",
    "LLM_ONLY": "LLM",
    "RAG": "RAG",
    "ACTION": "ACTION",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _print_progress(idx, total, message, raw_tool, predicted, expected, elapsed):
    ok = "OK" if predicted == expected else "FAIL"
    print(f"  [{idx:02d}/{total}] {message[:50]:50s} -> {str(raw_tool):30s} [{ok}] ({elapsed:.2f}s)")


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


# ── Agent test functions ────────────────────────────────────────────────────

def test_deterministic():
    """DeterministicAgentManager — pure keyword routing."""
    print("\n[deterministic] Testing DeterministicAgentManager ...")

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
        predicted = DETERMINISTIC_MAP.get(tool, tool)
        results.append({
            "message": message,
            "expected": expected,
            "predicted": predicted,
            "raw_tool": tool,
            "correct": predicted == expected,
            "time_s": round(elapsed, 6),
        })
        _print_progress(len(results), len(TEST_CASES), message, tool, predicted, expected, elapsed)

    return results


def test_planned():
    """PlannedAgentManager — extract the first meaningful tool from the LLM plan."""
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
            raw_tool = "UNKNOWN"
            for step in plan.get("steps", []):
                tool_name = step.get("tool", "")
                if tool_name in PLANNED_MAP:
                    raw_tool = tool_name
                    break
            if raw_tool == "UNKNOWN":
                for step in plan.get("steps", []):
                    tool_name = step.get("tool", "")
                    if tool_name in ("respond", "chat"):
                        raw_tool = tool_name
                        break
        except Exception as e:
            raw_tool = f"ERROR:{e}"
        elapsed = time.perf_counter() - t0
        predicted = PLANNED_MAP.get(raw_tool, raw_tool)
        results.append({
            "message": message,
            "expected": expected,
            "predicted": predicted,
            "raw_tool": raw_tool,
            "correct": predicted == expected,
            "time_s": round(elapsed, 4),
        })
        _print_progress(len(results), len(TEST_CASES), message, raw_tool, predicted, expected, elapsed)

    return results


def test_multi_agent():
    """MultiAgentManager — coordinator picks agent, then sub-agent picks action."""
    print("\n[multi_agent] Testing MultiAgentManager (coordinator + sub-agent) ...")
    from orchestration.manager_multi_agent import MultiAgentManager
    manager = MultiAgentManager()

    results = []
    for message, expected in TEST_CASES:
        manager.memory = ConversationMemory()
        t0 = time.perf_counter()
        try:
            coordinator_result = manager._run_coordinator(message, "")
            agent_name = coordinator_result.get("agent", "chat_agent")
            task = coordinator_result.get("task", message)

            if agent_name in MULTI_AGENT_AGENT_MAP:
                raw_tool = agent_name
                predicted = MULTI_AGENT_AGENT_MAP[agent_name]
            elif agent_name == "appointment_agent":
                available_slots = manager.appointments.list_available_slots()
                booked_appointments = manager.appointments.get_patient_appointments()
                available_data = manager._format_available_data(available_slots, booked_appointments)

                from orchestration.manager_multi_agent import _safe_format, _parse_json
                prompt = _safe_format(
                    manager._appointment_prompt,
                    available_data=available_data,
                    history="",
                    task=task,
                )
                appt_raw = manager.llm.generate(prompt)
                parsed = _parse_json(appt_raw)

                if parsed and "action" in parsed:
                    raw_tool = parsed["action"]
                else:
                    recovered = manager._recover_appointment_action_from_text(appt_raw, task, message)
                    if recovered:
                        raw_tool = recovered.get("action", agent_name)
                    else:
                        raw_tool = agent_name

                predicted = MULTI_AGENT_ACTION_MAP.get(raw_tool, raw_tool)

            elif agent_name == "messaging_agent":
                raw_tool = "send_message"
                predicted = MULTI_AGENT_ACTION_MAP.get(raw_tool, raw_tool)
            else:
                raw_tool = agent_name
                predicted = MULTI_AGENT_AGENT_MAP.get(agent_name, agent_name)

        except Exception as e:
            raw_tool = f"ERROR:{e}"
            predicted = raw_tool

        elapsed = time.perf_counter() - t0
        results.append({
            "message": message,
            "expected": expected,
            "predicted": predicted,
            "raw_tool": raw_tool,
            "correct": predicted == expected,
            "time_s": round(elapsed, 4),
        })
        _print_progress(len(results), len(TEST_CASES), message, raw_tool, predicted, expected, elapsed)

    return results


def test_basic_tools():
    """AgentManager (basic_tools) — LLM picks tool category, then deterministic
    routing resolves the specific action tool."""
    print("\n[basic_tools] Testing AgentManager (basic_tools, full agent) ...")
    from orchestration.manager_basic_tools import AgentManager
    manager = AgentManager()

    with patch("orchestration.manager_deterministic.OllamaClient", MagicMock), \
         patch("orchestration.manager_deterministic.Retriever", MagicMock), \
         patch("orchestration.manager_deterministic.AppointmentManager", MagicMock), \
         patch("orchestration.manager_deterministic.MessageManager", MagicMock):
        from orchestration.manager_deterministic import DeterministicAgentManager
        det_router = DeterministicAgentManager()

    _mock_tools_exec(manager.tools_exec)

    results = []
    for message, expected in TEST_CASES:
        manager.memory = ConversationMemory()
        manager._last_tool_used = None
        t0 = time.perf_counter()
        try:
            asyncio.run(manager.process_message(message, "test_user"))
            agent_tool = manager._last_tool_used or "NONE"
            if agent_tool == "ACTION":
                det_tool, _ = det_router._route(message)
                raw_tool = det_tool
                predicted = DETERMINISTIC_MAP.get(raw_tool, raw_tool)
            else:
                raw_tool = agent_tool
                predicted = BASIC_TOOLS_MAP.get(raw_tool, raw_tool)
        except Exception as e:
            raw_tool = f"ERROR:{e}"
            predicted = raw_tool
        elapsed = time.perf_counter() - t0

        results.append({
            "message": message,
            "expected": expected,
            "predicted": predicted,
            "raw_tool": raw_tool,
            "correct": predicted == expected,
            "time_s": round(elapsed, 4),
        })
        _print_progress(len(results), len(TEST_CASES), message, raw_tool, predicted, expected, elapsed)

    return results


def test_multi_tools():
    """AgentManagerMultiTools — _last_tool_used gives the exact tool name."""
    print("\n[multi_tools] Testing AgentManagerMultiTools (full agent) ...")
    from orchestration.manager_multiple_tools import AgentManagerMultiTools
    manager = AgentManagerMultiTools()
    manager.max_history_turns = 0

    _mock_tools_exec(manager.tools_exec)

    async def _run_all():
        results = []
        for message, expected in TEST_CASES:
            manager.memory = ConversationMemory()
            manager._last_tool_used = None
            t0 = time.perf_counter()
            try:
                await manager.process_message(message, "test_user")
                raw_tool = manager._last_tool_used or "NONE"
            except Exception as e:
                raw_tool = f"ERROR:{e}"
            elapsed = time.perf_counter() - t0
            predicted = MULTI_TOOLS_MAP.get(raw_tool, raw_tool)
            results.append({
                "message": message,
                "expected": expected,
                "predicted": predicted,
                "raw_tool": raw_tool,
                "correct": predicted == expected,
                "time_s": round(elapsed, 4),
            })
            _print_progress(len(results), len(TEST_CASES), message, raw_tool, predicted, expected, elapsed)
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
        description="Evaluate typo resilience for healthcare agent tool selection."
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
        test_fn = AGENT_TESTS.get(agent_name)
        if test_fn is None:
            print(f"  Unknown agent: {agent_name}")
            continue
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

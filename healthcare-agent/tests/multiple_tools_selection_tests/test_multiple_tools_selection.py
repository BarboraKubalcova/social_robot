#!/usr/bin/env python3
"""
Specific ACTION tool selection evaluation for all agent managers.

Tests that each manager routes to the correct *specific* action tool
(LIST_SLOTS, LIST_APPOINTMENTS, BOOK_APPOINTMENT, CANCEL_APPOINTMENT,
RESCHEDULE_APPOINTMENT, SEND_DOCTOR_MESSAGE) rather than a generic ACTION
category.

For each manager the test extracts the concrete tool name:
  - deterministic:  _route() returns it directly
  - basic_tools:    tracks which tools_exec method was called
  - planned:        extracts the first action tool from the plan steps
  - multi_tools:    _last_tool_used gives the exact tool name
  - multi_agent:    captures the action inside appointment/messaging agents

Results are written to per-agent CSV files in results/.

Usage:
    python -m tests.multiple_tools_selection_tests.test_multiple_tools_selection              # all
    python -m tests.multiple_tools_selection_tests.test_multiple_tools_selection deterministic # one
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

from tests.multiple_tools_selection_tests.test_cases import TEST_CASES
from orchestration.memory import ConversationMemory

CSV_FIELDS = ["message", "expected", "predicted", "raw_tool", "correct", "time_s"]
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# ── Normalisation maps ──────────────────────────────────────────────────────
# Each map converts manager-specific tool names to the canonical 6-tool names.

DETERMINISTIC_MAP = {
    "LIST_SLOTS": "LIST_SLOTS",
    "LIST_APPOINTMENTS": "LIST_APPOINTMENTS",
    "BOOK_APPOINTMENT": "BOOK_APPOINTMENT",
    "CANCEL_APPOINTMENT": "CANCEL_APPOINTMENT",
    "RESCHEDULE_APPOINTMENT": "RESCHEDULE_APPOINTMENT",
    "SEND_DOCTOR_MESSAGE": "SEND_DOCTOR_MESSAGE",
}

PLANNED_MAP = {
    "list_slots": "LIST_SLOTS",
    "list_appointments": "LIST_APPOINTMENTS",
    "book_appointment": "BOOK_APPOINTMENT",
    "cancel_appointment": "CANCEL_APPOINTMENT",
    "reschedule_appointment": "RESCHEDULE_APPOINTMENT",
    "send_doctor_message": "SEND_DOCTOR_MESSAGE",
}

MULTI_TOOLS_MAP = {
    "LIST_SLOTS": "LIST_SLOTS",
    "LIST_APPOINTMENTS": "LIST_APPOINTMENTS",
    "BOOK_APPOINTMENT": "BOOK_APPOINTMENT",
    "CANCEL_APPOINTMENT": "CANCEL_APPOINTMENT",
    "RESCHEDULE_APPOINTMENT": "RESCHEDULE_APPOINTMENT",
    "SEND_DOCTOR_MESSAGE": "SEND_DOCTOR_MESSAGE",
}

MULTI_AGENT_ACTION_MAP = {
    "list_slots": "LIST_SLOTS",
    "list_appointments": "LIST_APPOINTMENTS",
    "book": "BOOK_APPOINTMENT",
    "cancel": "CANCEL_APPOINTMENT",
    "reschedule": "RESCHEDULE_APPOINTMENT",
    "send_message": "SEND_DOCTOR_MESSAGE",
}

# For basic_tools the agent only sees "ACTION"; we track the specific method
# called on ToolExecutor to determine the sub-tool.
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
    "run_action_tool": "ACTION_GENERIC",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _print_progress(idx, total, message, raw_tool, predicted, expected, elapsed):
    ok = "OK" if predicted == expected else "FAIL"
    print(f"  [{idx:02d}/{total}] {message[:50]:50s} -> {str(raw_tool):30s} [{ok}] ({elapsed:.2f}s)")


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
    """Patch ToolExecutor instance methods with tracking wrappers."""
    mocks = _make_tracking_mock(tracker)
    for attr, fn in mocks.items():
        setattr(tools_exec, attr, fn)


def _mock_tools_exec(tools_exec):
    """Replace ToolExecutor methods with fast no-op stubs so tool execution
    is instant and has no side-effects."""
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
    """PlannedAgentManager — extract the first action tool from the LLM plan."""
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
            # Extract the first non-respond, non-chat tool from steps
            raw_tool = "UNKNOWN"
            for step in plan.get("steps", []):
                tool_name = step.get("tool", "")
                if tool_name not in ("respond", "chat"):
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
    """MultiAgentManager — capture the specific action from sub-agents."""
    print("\n[multi_agent] Testing MultiAgentManager (coordinator + sub-agent) ...")
    from orchestration.manager_multi_agent import MultiAgentManager
    manager = MultiAgentManager()

    results = []
    for message, expected in TEST_CASES:
        manager.memory = ConversationMemory()
        t0 = time.perf_counter()
        try:
            # Step 1: coordinator picks agent
            coordinator_result = manager._run_coordinator(message, "")
            agent_name = coordinator_result.get("agent", "chat_agent")
            task = coordinator_result.get("task", message)

            raw_tool = agent_name  # fallback

            if agent_name == "appointment_agent":
                # Get what the appointment sub-agent would decide
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
                    # Try text recovery
                    recovered = manager._recover_appointment_action_from_text(appt_raw, task, message)
                    if recovered:
                        raw_tool = recovered.get("action", agent_name)

            elif agent_name == "messaging_agent":
                raw_tool = "send_message"

        except Exception as e:
            raw_tool = f"ERROR:{e}"

        elapsed = time.perf_counter() - t0
        predicted = MULTI_AGENT_ACTION_MAP.get(raw_tool, raw_tool)
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
    """AgentManager (basic_tools) — LLM picks ACTION, then deterministic
    routing on the original message determines the specific action tool.

    basic_tools exposes a single ACTION tool to the LLM agent. When the agent
    selects it, we use the deterministic _route() on the original user message
    to resolve the specific action tool (LIST_SLOTS, BOOK_APPOINTMENT, etc.).
    """
    print("\n[basic_tools] Testing AgentManager (basic_tools, full agent) ...")
    from orchestration.manager_basic_tools import AgentManager
    manager = AgentManager()

    # Build a deterministic router for sub-action classification
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
            # If the agent picked ACTION, use deterministic routing to get the specific tool
            if agent_tool == "ACTION":
                det_tool, _ = det_router._route(message)
                raw_tool = det_tool
            else:
                raw_tool = agent_tool
        except Exception as e:
            raw_tool = f"ERROR:{e}"
        elapsed = time.perf_counter() - t0

        predicted = DETERMINISTIC_MAP.get(raw_tool, raw_tool)
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
    manager.max_history_turns = 0  # minimize token count per call

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
        description="Evaluate specific ACTION tool selection for healthcare agent managers."
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

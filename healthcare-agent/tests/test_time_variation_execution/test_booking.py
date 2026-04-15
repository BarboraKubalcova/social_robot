#!/usr/bin/env python3
"""
Actual execution test for appointment booking.

Unlike the tool-selection tests, this suite verifies that the appointment
was **really booked** in the AppointmentManager table after the agent
processes the message.

Two groups of 10 test cases each:
  1. DIRECT SLOT ID  — user mentions slot_1, slot_5, …
  2. DAY + TIME      — user mentions "Monday 9:30", "Wednesday 13:00", …

Tests all 5 agent managers:
  - deterministic   (DeterministicAgentManager)
  - basic_tools     (AgentManager)
  - planned         (PlannedAgentManager)
  - multi_tools     (AgentManagerMultiTools)
  - multi_agent     (MultiAgentManager)

Usage:
    python -m tests.test_actual_execution.test_booking                          # all agents, both groups
    python -m tests.test_actual_execution.test_booking --agents multi_tools     # one agent
    python -m tests.test_actual_execution.test_booking --groups direct          # one group
    python -m tests.test_actual_execution.test_booking --agents planned multi_tools --groups day_time
"""

import argparse
import asyncio
import csv
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from tests.test_actual_execution.test_cases import DIRECT_SLOT_CASES, DAY_TIME_CASES
from orchestration.memory import ConversationMemory
from execution.actions.appointments import AppointmentManager

CSV_FIELDS = [
    "agent", "group", "message", "expected_slot", "booked_slot",
    "correct", "response", "time_s",
]
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

CASE_GROUPS = {
    "direct": ("Direct slot ID", DIRECT_SLOT_CASES),
    "day_time": ("Day + time", DAY_TIME_CASES),
}


def _reset_appointment_manager() -> AppointmentManager:
    """Destroy the singleton and create a fresh AppointmentManager."""
    AppointmentManager._instance = None
    if hasattr(AppointmentManager, "_initialized"):
        del AppointmentManager._initialized
    mgr = AppointmentManager()
    return mgr


def _find_booked_slot(appointments: AppointmentManager) -> str | None:
    """Return the slot_id of the first (and hopefully only) booked slot,
    or None if nothing was booked."""
    for slot in appointments.get_all_slots():
        if slot["status"] == "occupied":
            return slot["id"]
    return None


def _print_progress(idx, total, message, expected, booked, correct, elapsed):
    tag = "OK" if correct else "FAIL"
    print(
        f"  [{idx:02d}/{total}] {message[:55]:55s} "
        f"expected={expected:8s} booked={str(booked):8s} [{tag}] ({elapsed:.2f}s)"
    )


# ── Factory functions for each manager ──────────────────────────────────────

def _make_deterministic():
    from orchestration.manager_deterministic import DeterministicAgentManager
    manager = DeterministicAgentManager()
    return manager


def _make_basic_tools():
    from orchestration.manager_basic_tools import AgentManager
    manager = AgentManager()
    manager.memory = ConversationMemory()
    manager.max_history_turns = 0
    return manager


def _make_planned():
    from orchestration.manager_commands import PlannedAgentManager
    manager = PlannedAgentManager()
    manager.memory = ConversationMemory()
    manager.max_history_turns = 0
    return manager


def _make_multi_tools():
    from orchestration.manager_multiple_tools import AgentManagerMultiTools
    manager = AgentManagerMultiTools()
    manager.memory = ConversationMemory()
    manager.max_history_turns = 0
    return manager


def _make_multi_agent():
    from orchestration.manager_multi_agent import MultiAgentManager
    manager = MultiAgentManager()
    manager.memory = ConversationMemory()
    manager.max_history_turns = 0
    return manager


AGENT_FACTORIES = {
    "deterministic": _make_deterministic,
    "basic_tools": _make_basic_tools,
    "planned": _make_planned,
    "multi_tools": _make_multi_tools,
    "multi_agent": _make_multi_agent,
}


# ── Test runner ─────────────────────────────────────────────────────────────

async def run_test_group(
    agent_name: str,
    group_name: str,
    cases: list[tuple[str, str]],
) -> list[dict]:
    """Run one group of test cases for one agent and return result dicts."""
    factory = AGENT_FACTORIES[agent_name]
    results = []
    total = len(cases)

    for idx, (message, expected_slot) in enumerate(cases, 1):
        # Fresh appointment table for each test case
        fresh_appts = _reset_appointment_manager()

        # Re-create the manager so it picks up the fresh singleton
        manager = factory()

        # Verify the manager's appointment table is the fresh one
        appts_ref = getattr(manager, "appointments", None)
        assert appts_ref is fresh_appts, (
            f"{agent_name}: manager.appointments is not the fresh singleton"
        )

        t0 = time.perf_counter()
        try:
            result = await manager.process_message(message, "test_user")
            response_text = result.get("response", "")
        except Exception as e:
            response_text = f"ERROR: {e}"
        elapsed = time.perf_counter() - t0

        booked_slot = _find_booked_slot(appts_ref)
        correct = booked_slot == expected_slot

        results.append({
            "agent": agent_name,
            "group": group_name,
            "message": message,
            "expected_slot": expected_slot,
            "booked_slot": booked_slot or "NONE",
            "correct": correct,
            "response": response_text[:200],
            "time_s": round(elapsed, 4),
        })

        _print_progress(idx, total, message, expected_slot, booked_slot, correct, elapsed)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test actual appointment booking execution."
    )
    parser.add_argument(
        "--agents",
        nargs="*",
        default=None,
        help="Which agents to test (default: all). Choices: "
             + ", ".join(AGENT_FACTORIES.keys()),
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Which case groups to test: direct, day_time (default: both)",
    )
    args = parser.parse_args()

    agents_to_test = args.agents if args.agents else list(AGENT_FACTORIES.keys())
    groups_to_test = args.groups if args.groups else list(CASE_GROUPS.keys())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}  # agent_name -> list[dict]

    for agent_name in agents_to_test:
        if agent_name not in AGENT_FACTORIES:
            print(f"  Unknown agent: {agent_name}")
            continue

        agent_results = []
        for group_key in groups_to_test:
            if group_key not in CASE_GROUPS:
                print(f"  Unknown group: {group_key}")
                continue

            label, cases = CASE_GROUPS[group_key]
            print(f"\n{'='*70}")
            print(f"  Agent: {agent_name}  |  Group: {label} ({len(cases)} cases)")
            print(f"{'='*70}")

            try:
                results = asyncio.run(run_test_group(agent_name, group_key, cases))
                agent_results.extend(results)
            except Exception as e:
                print(f"  SKIPPED {agent_name}/{group_key}: {e}")

        if agent_results:
            all_results[agent_name] = agent_results

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
    print(f"\n{'='*70}")
    print(f"  Results saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")

    grand_correct = 0
    grand_total = 0

    for agent_name, results in sorted(all_results.items()):
        print(f"\n  Agent: {agent_name}")
        for group_key in groups_to_test:
            if group_key not in CASE_GROUPS:
                continue
            group_results = [r for r in results if r["group"] == group_key]
            if not group_results:
                continue
            correct = sum(1 for r in group_results if r["correct"])
            total = len(group_results)
            avg_time = sum(r["time_s"] for r in group_results) / total
            print(
                f"    {CASE_GROUPS[group_key][0]:20s}  "
                f"{correct}/{total} correct ({100 * correct / total:5.1f}%)  "
                f"avg_time={avg_time:.4f}s"
            )

        agent_correct = sum(1 for r in results if r["correct"])
        agent_total = len(results)
        avg_time = sum(r["time_s"] for r in results) / agent_total
        print(
            f"    {'TOTAL':20s}  "
            f"{agent_correct}/{agent_total} correct ({100 * agent_correct / agent_total:5.1f}%)  "
            f"avg_time={avg_time:.4f}s"
        )
        grand_correct += agent_correct
        grand_total += agent_total

    print(
        f"\n  {'GRAND TOTAL':20s}  "
        f"{grand_correct}/{grand_total} correct ({100 * grand_correct / grand_total:5.1f}%)"
    )

    # Print failures
    all_flat = [r for results in all_results.values() for r in results]
    failures = [r for r in all_flat if not r["correct"]]
    if failures:
        print(f"\n  Failed cases:")
        for r in failures:
            print(
                f"    [{r['agent']:15s} {r['group']:8s}] "
                f"expected={r['expected_slot']:8s} booked={r['booked_slot']:8s}  "
                f"\"{r['message'][:55]}\""
            )
            print(f"{'':40s} response: {r['response'][:120]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Print a summary of actual booking execution test results.

Shows per-agent accuracy broken down by group (direct slot ID vs day+time),
failed cases, and the agent's response for failed cases.

Usage:
    python -m tests.test_actual_execution.print_results
"""

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

GROUPS = {
    "direct": "Direct slot ID",
    "day_time": "Day + time",
}


def load_results(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["correct"] = row["correct"] == "True"
            row["time_s"] = float(row["time_s"])
            rows.append(row)
        return rows


def print_agent_summary(agent_name: str, results: list[dict]):
    print(f"\n  Agent: {agent_name}")

    # ── Per-group accuracy ──
    print(f"\n    {'Group':<20s}  {'Correct':>7s}  {'Total':>5s}  {'Accuracy':>8s}  {'Avg Time':>9s}")
    print(f"    {'-'*20}  {'-'*7}  {'-'*5}  {'-'*8}  {'-'*9}")
    for group_key, group_label in GROUPS.items():
        group_rows = [r for r in results if r["group"] == group_key]
        if not group_rows:
            continue
        correct = sum(1 for r in group_rows if r["correct"])
        total = len(group_rows)
        pct = 100 * correct / total if total else 0
        avg_t = sum(r["time_s"] for r in group_rows) / total if total else 0
        print(f"    {group_label:<20s}  {correct:7d}  {total:5d}  {pct:7.1f}%  {avg_t:8.4f}s")

    # ── Overall ──
    total_correct = sum(1 for r in results if r["correct"])
    total_count = len(results)
    avg_time = sum(r["time_s"] for r in results) / total_count if total_count else 0
    pct_all = 100 * total_correct / total_count if total_count else 0
    print(f"\n    {'OVERALL':<20s}  {total_correct:7d}  {total_count:5d}  {pct_all:7.1f}%  avg_time={avg_time:.4f}s")

    # ── Failed cases ──
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n    Failed cases:")
        for r in failures:
            print(
                f"      [{r['group']:8s}] expected={r['expected_slot']:8s} "
                f"booked={r['booked_slot']:8s}  \"{r['message'][:60]}\""
            )
            print(f"                response: {r['response'][:120]}")


def main():
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {RESULTS_DIR}")
        return

    print(f"{'=' * 70}")
    print(f"  Booking Execution Test Results ({RESULTS_DIR})")
    print(f"{'=' * 70}")

    for csv_path in csv_files:
        agent_name = csv_path.stem
        results = load_results(csv_path)
        print_agent_summary(agent_name, results)

    print()


if __name__ == "__main__":
    main()

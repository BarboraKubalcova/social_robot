#!/usr/bin/env python3
"""
Print a summary of all history-length test results from the results/ directory.

Usage:
    python -m tests.test_history_length.print_results
"""

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_results(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["history_turns"] = int(row["history_turns"])
            row["correct"] = row["correct"] == "True"
            row["time_s"] = float(row["time_s"])
            rows.append(row)
        return rows


def print_agent_summary(agent_name: str, results: list[dict]):
    turns = sorted(set(r["history_turns"] for r in results))

    print(f"\n  Agent: {agent_name}")
    print(f"  {'turns':>5s}  {'correct':>7s}  {'total':>5s}  {'accuracy':>8s}  {'avg_time':>9s}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*8}  {'-'*9}")

    total_correct = 0
    total_count = 0
    for t in turns:
        turn_rows = [r for r in results if r["history_turns"] == t]
        correct = sum(1 for r in turn_rows if r["correct"])
        total = len(turn_rows)
        avg_time = sum(r["time_s"] for r in turn_rows) / total if total else 0
        pct = 100 * correct / total if total else 0
        print(f"  {t:5d}  {correct:7d}  {total:5d}  {pct:7.1f}%  {avg_time:8.4f}s")
        total_correct += correct
        total_count += total

    pct_all = 100 * total_correct / total_count if total_count else 0
    print(f"  {'ALL':>5s}  {total_correct:7d}  {total_count:5d}  {pct_all:7.1f}%")


def main():
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {RESULTS_DIR}")
        return

    print(f"{'=' * 60}")
    print(f"  History-length test results ({RESULTS_DIR})")
    print(f"{'=' * 60}")

    for csv_path in csv_files:
        agent_name = csv_path.stem
        results = load_results(csv_path)
        print_agent_summary(agent_name, results)

    print()


if __name__ == "__main__":
    main()

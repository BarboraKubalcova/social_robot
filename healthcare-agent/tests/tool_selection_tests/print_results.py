#!/usr/bin/env python3
"""
Print a summary of tool selection test results from the results/ directory.

Shows per-agent accuracy broken down by category (CHAT, RAG, ACTION),
plus failed cases.

Usage:
    python -m tests.tool_selection_tests.print_results
"""

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CATEGORIES = ["CHAT", "RAG", "ACTION"]


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

    # ── Per-category accuracy ──
    print(f"\n    {'Category':<15s}  {'Correct':>7s}  {'Total':>5s}  {'Accuracy':>8s}  {'Avg Time':>9s}")
    print(f"    {'-'*15}  {'-'*7}  {'-'*5}  {'-'*8}  {'-'*9}")
    for cat in CATEGORIES:
        cat_rows = [r for r in results if r["expected"] == cat]
        correct = sum(1 for r in cat_rows if r["correct"])
        total = len(cat_rows)
        pct = 100 * correct / total if total else 0
        avg_t = sum(r["time_s"] for r in cat_rows) / total if total else 0
        print(f"    {cat:<15s}  {correct:7d}  {total:5d}  {pct:7.1f}%  {avg_t:8.4f}s")

    # ── Overall ──
    total_correct = sum(1 for r in results if r["correct"])
    total_count = len(results)
    avg_time = sum(r["time_s"] for r in results) / total_count if total_count else 0
    pct_all = 100 * total_correct / total_count if total_count else 0
    print(f"\n    {'OVERALL':<15s}  {total_correct:7d}  {total_count:5d}  {pct_all:7.1f}%  avg_time={avg_time:.4f}s")

    # ── Failed cases ──
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n    Failed cases:")
        for r in failures:
            print(f"      expected={r['expected']:8s} predicted={r['predicted']:8s}  \"{r['message'][:70]}\"")


def main():
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {RESULTS_DIR}")
        return

    print(f"{'=' * 70}")
    print(f"  Tool Selection Test Results ({RESULTS_DIR})")
    print(f"{'=' * 70}")

    for csv_path in csv_files:
        agent_name = csv_path.stem
        results = load_results(csv_path)
        print_agent_summary(agent_name, results)

    print()


if __name__ == "__main__":
    main()

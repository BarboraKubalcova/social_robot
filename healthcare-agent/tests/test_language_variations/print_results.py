#!/usr/bin/env python3
"""
Print a summary of language variation test results from the results/ directory.

Shows per-agent accuracy broken down by tool and by language style.

Usage:
    python -m tests.test_language_variations.print_results
"""

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Language styles in order (matches test_cases.py: 5 per tool, rotating)
STYLES = ["formal", "slang", "typos", "non-native", "verbose"]

# Tools in test_cases.py order
TOOLS = [
    "LLM", "RAG", "LIST_SLOTS", "LIST_APPOINTMENTS",
    "BOOK_APPOINTMENT", "CANCEL_APPOINTMENT",
    "RESCHEDULE_APPOINTMENT", "SEND_DOCTOR_MESSAGE",
]


def load_results(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["correct"] = row["correct"] == "True"
            row["time_s"] = float(row["time_s"])
            rows.append(row)
        return rows


def assign_style(index: int) -> str:
    """Each tool has 5 cases in order: formal, slang, typos, non-native, verbose."""
    return STYLES[index % len(STYLES)]


def assign_tool(index: int) -> str:
    """5 cases per tool, in order."""
    return TOOLS[index // len(STYLES)]


def print_agent_summary(agent_name: str, results: list[dict]):
    print(f"\n  Agent: {agent_name}")

    # Annotate rows with style and tool group
    for i, r in enumerate(results):
        r["style"] = assign_style(i)
        r["tool_group"] = assign_tool(i)

    # ── Per-tool accuracy ──
    print(f"\n    {'Tool':<25s}  {'Correct':>7s}  {'Total':>5s}  {'Accuracy':>8s}  {'Avg Time':>9s}")
    print(f"    {'-'*25}  {'-'*7}  {'-'*5}  {'-'*8}  {'-'*9}")
    for tool in TOOLS:
        tool_rows = [r for r in results if r["tool_group"] == tool]
        correct = sum(1 for r in tool_rows if r["correct"])
        total = len(tool_rows)
        pct = 100 * correct / total if total else 0
        avg_t = sum(r["time_s"] for r in tool_rows) / total if total else 0
        print(f"    {tool:<25s}  {correct:7d}  {total:5d}  {pct:7.1f}%  {avg_t:8.4f}s")

    # ── Per-style accuracy ──
    print(f"\n    {'Style':<25s}  {'Correct':>7s}  {'Total':>5s}  {'Accuracy':>8s}  {'Avg Time':>9s}")
    print(f"    {'-'*25}  {'-'*7}  {'-'*5}  {'-'*8}  {'-'*9}")
    for style in STYLES:
        style_rows = [r for r in results if r["style"] == style]
        correct = sum(1 for r in style_rows if r["correct"])
        total = len(style_rows)
        pct = 100 * correct / total if total else 0
        avg_t = sum(r["time_s"] for r in style_rows) / total if total else 0
        print(f"    {style:<25s}  {correct:7d}  {total:5d}  {pct:7.1f}%  {avg_t:8.4f}s")

    # ── Overall ──
    total_correct = sum(1 for r in results if r["correct"])
    total_count = len(results)
    avg_time = sum(r["time_s"] for r in results) / total_count if total_count else 0
    pct_all = 100 * total_correct / total_count if total_count else 0
    print(f"\n    {'OVERALL':<25s}  {total_correct:7d}  {total_count:5d}  {pct_all:7.1f}%  avg_time={avg_time:.4f}s")

    # ── Failed cases ──
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n    Failed cases:")
        for r in failures:
            print(f"      [{r['style']:10s}] expected={r['expected']:25s} predicted={r['predicted']:25s}  \"{r['message'][:60]}\"")


def main():
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {RESULTS_DIR}")
        return

    print(f"{'=' * 70}")
    print(f"  Language Variation Test Results ({RESULTS_DIR})")
    print(f"{'=' * 70}")

    for csv_path in csv_files:
        agent_name = csv_path.stem
        results = load_results(csv_path)
        print_agent_summary(agent_name, results)

    print()


if __name__ == "__main__":
    main()

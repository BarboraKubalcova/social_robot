# Tests – Tool Selection Evaluation

This test suite evaluates the routing accuracy of all manager strategies in the healthcare agent. It measures how well each manager routes user messages to the correct category: **CHAT**, **RAG**, or **ACTION**.

## Prerequisites

1. **Install dev dependencies** (from the `healthcare-agent/` directory):

   ```bash
   pip install -e ".[dev]"
   ```

2. **Ollama** (only required for LLM-based tests): make sure the Ollama server is running on `localhost:11434`.

## Test Structure

| File | Purpose |
|---|---|
| `test_cases.py` | Defines all `(message, expected_category)` pairs used by the tests |
| `test_tool_selection.py` | Contains two test classes and a summary report |

### Test Classes

- **`TestKeywordRouting`** – Fast, deterministic keyword/fallback routing. No LLM needed.
- **`TestLLMRouting`** – Full LLM-based routing through each manager. Requires a running Ollama instance. Automatically skipped if Ollama is not available.

### Tested Managers

| Manager | Keyword test | LLM test |
|---|---|---|
| `DeterministicAgentManager` | `_fallback_route` | `_decide_tool` |
| `PlannedAgentManager` | `_naive_backup_plan` | `_plan` |
| `MultiAgentManager` | `_keyword_fallback` | `_route` |
| `AgentManager` (basic tools) | – | `_route_and_respond` |
| `AgentManagerMultiTools` | – | `_route_and_respond` |

## Running the Tests

All commands should be run from the `healthcare-agent/` directory.

### Run all tests

```bash
python -m pytest tests/test_tool_selection.py -v
```

### Run only keyword tests (fast, no LLM)

```bash
python -m pytest tests/test_tool_selection.py -v -k keyword
```

### Run only LLM tests (requires Ollama)

```bash
python -m pytest tests/test_tool_selection.py -v -k llm
```

### Run a single test

```bash
python -m pytest tests/test_tool_selection.py -v -k test_keyword_deterministic
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `LLM_QUERY_TIMEOUT` | `180` | Max seconds per single LLM query before it is marked as timed out |
| `KEYWORD_PREROUTE` | – | Set to `0` to force LLM routing in `DeterministicAgentManager` |

## Results

After a test run, CSV files are written to `tests/results/`:

- `*_results.csv` – Per-query results (message, expected, predicted, correct, time)
- `*_metrics.csv` – Aggregated metrics (accuracy, precision, recall per category)

The `test_summary_report` test prints a comparison table across all completed evaluations.

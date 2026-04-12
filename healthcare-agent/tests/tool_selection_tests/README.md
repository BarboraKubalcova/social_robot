# Tool Selection Tests

Evaluates how accurately each agent manager routes user messages to the correct tool category.

## Categories

Every agent uses its own tool naming internally. For evaluation, all tools are mapped to three standard categories:

| Category | Description |
|----------|-------------|
| **CHAT** | Greetings, casual conversation, emotional support |
| **RAG** | Medical knowledge, clinic procedures, policies, preparation instructions |
| **ACTION** | Appointments (book/cancel/reschedule/list), slot listing, doctor messaging |

## Agents Tested

| Agent | Module | What's tested | Requires LLM |
|-------|--------|---------------|:---:|
| `deterministic` | `manager_deterministic.py` | `_route()` — pure keyword/regex routing | No |
| `planned` | `manager_commands.py` | `_plan()` + `_validate_plan()` → `_extract_intent()` | Yes |
| `multi_agent` | `manager_multi_agent.py` | `_run_coordinator()` — agent delegation | Yes |
| `basic_tools` | `manager_basic_tools.py` | Full LangChain agent (tool execution mocked) | Yes |
| `multi_tools` | `manager_multiple_tools.py` | Full LangChain agent (tool execution mocked) | Yes |

Only the **tool selection** step is evaluated — actual tool execution (RAG retrieval, appointment booking, etc.) is mocked so tests run fast and without side effects.

## Test Cases

Defined in `test_cases.py` — 30 messages (10 per category). Add new cases as `(message, expected_category)` tuples.

## Usage

Run from the `healthcare-agent/` directory:

```bash
# All agents
python3 -m tests.tool_selection_tests.test_tool_selection

# Single agent
python3 -m tests.tool_selection_tests.test_tool_selection deterministic

# Subset
python3 -m tests.tool_selection_tests.test_tool_selection planned multi_agent
```

## Output

Per-agent CSV files are saved to `results/`:

```
tests/tool_selection_tests/results/
├── deterministic.csv
├── planned.csv
├── multi_agent.csv
├── basic_tools.csv
└── multi_tools.csv
```

CSV columns:

| Column | Description |
|--------|-------------|
| `message` | Input user message |
| `expected` | Ground-truth category (CHAT / RAG / ACTION) |
| `predicted` | Category predicted by the agent |
| `raw_tool` | Agent's native tool name before mapping |
| `correct` | `True` if predicted == expected |
| `time_s` | Tool selection time in seconds |

A summary with accuracy and average time per agent is printed to stdout after each run.

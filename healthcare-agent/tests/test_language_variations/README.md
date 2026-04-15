# Language Variation Tests

Tests tool selection accuracy when the same intent is expressed in very different language styles: **formal**, **slang/casual**, **typos/misspellings**, **non-native phrasing**, and **verbose/roundabout**.

## Tools Tested

Unlike the category-level tests, these evaluate the **actual specific tool** selected after routing:

| Tool | Description |
|------|-------------|
| `LLM` | Chat / greetings / casual conversation |
| `RAG` | Medical knowledge, procedures, preparation |
| `LIST_SLOTS` | Show available appointment slots |
| `LIST_APPOINTMENTS` | Show booked appointments |
| `BOOK_APPOINTMENT` | Book a specific slot |
| `CANCEL_APPOINTMENT` | Cancel an existing appointment |
| `RESCHEDULE_APPOINTMENT` | Move an appointment to a new slot |
| `SEND_DOCTOR_MESSAGE` | Send a message to the doctor |

## Test Cases

Defined in `test_cases.py` — **40 messages** (5 per tool × 8 tools). Each tool has one example per language style:

| Style | Example |
|-------|---------|
| Formal | *"I wish to enquire about the currently available appointment slots."* |
| Slang | *"yo got any slots open?"* |
| Typos | *"Shwo me avalable sltos"* |
| Non-native | *"I am wanting to see what times are free for making the appointment please"* |
| Verbose | *"Could you possibly check and let me know which time slots are still available?"* |

## Usage

Run from the `healthcare-agent/` directory:

```bash
# All agents
python3 -m tests.test_language_variations.test_language_variations

# Single agent
python3 -m tests.test_language_variations.test_language_variations deterministic

# Subset
python3 -m tests.test_language_variations.test_language_variations planned multi_tools

# Print results summary
python3 -m tests.test_language_variations.print_results
```

## Output

Per-agent CSV files are saved to `results/`:

```
tests/test_language_variations/results/
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
| `expected` | Ground-truth tool name |
| `predicted` | Tool predicted by the agent (after normalisation) |
| `raw_tool` | Agent's native tool name before mapping |
| `correct` | `True` if predicted == expected |
| `time_s` | Tool selection time in seconds |

## Results Summary

`print_results.py` shows accuracy broken down by:

- **Tool** — which tools are hardest to identify under language variation
- **Style** — which language styles cause the most misclassifications
- **Failed cases** — lists every incorrect prediction with expected vs predicted

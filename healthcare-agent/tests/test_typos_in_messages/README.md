# Typo Resilience Tests

Tests tool selection accuracy when user messages contain **misspellings, character swaps, missing letters, and keyboard-adjacent errors**. Evaluates whether agents can still route to the correct specific tool despite garbled input.

## Tools Tested

These evaluate the **actual specific tool** selected after routing, not just the high-level category:

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

Defined in `test_cases.py` — **40 messages** (5 per tool × 8 tools). Every message contains realistic typos:

| Tool | Example |
|------|---------|
| LLM | *"Helo thre, how r you?"* |
| RAG | *"How shoudl I preprae for an MIR?"* |
| LIST_SLOTS | *"Shwo me avialble sltos"* |
| LIST_APPOINTMENTS | *"Waht apointments do I hve?"* |
| BOOK_APPOINTMENT | *"Boko me an apointmnet for slot_5"* |
| CANCEL_APPOINTMENT | *"Cancle my apointment appt_2"* |
| RESCHEDULE_APPOINTMENT | *"Reshedule appt_1 to slot_9"* |
| SEND_DOCTOR_MESSAGE | *"Sned a mesage to my doctr sayign I feel unwel"* |

## Usage

Run from the `healthcare-agent/` directory:

```bash
# All agents
python3 -m tests.test_typos_in_messages.test_typos

# Single agent
python3 -m tests.test_typos_in_messages.test_typos deterministic

# Subset
python3 -m tests.test_typos_in_messages.test_typos planned multi_tools

# Print results summary
python3 -m tests.test_typos_in_messages.print_results
```

## Output

Per-agent CSV files are saved to `results/`:

```
tests/test_typos_in_messages/results/
├── deterministic.csv
├── planned.csv
├── multi_agent.csv
├── basic_tools.csv
└── multi_tools.csv
```

CSV columns:

| Column | Description |
|--------|-------------|
| `message` | Input user message (with typos) |
| `expected` | Ground-truth tool name |
| `predicted` | Tool predicted by the agent (after normalisation) |
| `raw_tool` | Agent's native tool name before mapping |
| `correct` | `True` if predicted == expected |
| `time_s` | Tool selection time in seconds |

## Results Summary

`print_results.py` shows accuracy broken down by:

- **Tool** — which tools are hardest to identify when misspelled
- **Failed cases** — lists every incorrect prediction with expected vs predicted

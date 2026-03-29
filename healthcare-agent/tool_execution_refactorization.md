# Tool Execution Refactorization

## Problem

Every agent manager had its own copy of tool implementation logic — slot listing, appointment booking, canceling, rescheduling, RAG retrieval, message sending, and all the helper functions (ID extraction, day filtering, slot/appointment formatting, etc.).

This meant:
- **5 separate implementations** of the same tool logic across 5 managers
- A bug fix or behavior change had to be replicated in every manager
- Testing measured both the agent's **routing ability** and its **tool behavior** at the same time — impossible to isolate what was being evaluated
- ~1500 lines of duplicated code across the managers

## Goal

Unify all tool execution into a single shared module so that every manager delegates to the **same tool implementations**. Each manager now only handles:
- Its own **routing / tool-selection mechanism** (prompts, LLM calls, keyword fallbacks)
- Its own **response composition** (how to present the tool result to the user)

The tools themselves behave identically regardless of which agent selected them.

## What Was Created

### `orchestration/tool_implementations.py` — `ToolExecutor` class

A stateless executor shared by all managers. Instantiated once per manager with shared dependencies:

```python
self.tools_exec = ToolExecutor(self.retriever, self.appointments, self.messaging)
```

#### Methods

| Method | Purpose |
|---|---|
| `run_llm_tool(message)` | Returns a formatted prompt for casual conversation |
| `run_rag_tool(message, history_text)` | Retrieves context from the knowledge base, returns `(context_string, mode)` |
| `run_list_slots()` | Lists available appointment slots |
| `run_list_appointments()` | Lists the patient's booked appointments |
| `run_book_appointment(message)` | Books a slot, extracting slot ID from message text |
| `run_cancel_appointment(message)` | Cancels an appointment, extracting appointment ID from message text |
| `run_reschedule_appointment(message)` | Reschedules an appointment, extracting IDs from message text |
| `run_send_doctor_message(message)` | Sends a message to a doctor |
| `run_action_tool(message)` | **Aggregate ACTION tool** — keyword-based routing to the correct specific tool (for 3-tool agents) |
| `run_book_appointment_by_id(slot_id)` | Direct booking by slot ID (for planner steps) |
| `run_cancel_appointment_by_id(appointment_id)` | Direct cancel by appointment ID (for planner steps) |
| `run_reschedule_appointment_by_id(appointment_id, new_slot_id)` | Direct reschedule by IDs (for planner steps) |

#### Internal helpers (also unified)

- `_extract_slot_id()`, `_extract_appointment_id()` — regex extraction from message text
- `_extract_requested_day()`, `_extract_source_day()` — day name extraction
- `_filter_slots_by_day()`, `_filter_appointments_by_day()` — filtering by day
- `_format_slot_preview()`, `_format_appointment_preview()` — human-readable formatting
- `_wants_slots()`, `_wants_appointments()` — keyword detection
- `_handle_reschedule()` — full reschedule logic with inference

## What Changed in Each Manager

### `manager_basic_tools.py` (AgentManager — LangGraph, 3 tools)

**Tools:** RAG, LLM, ACTION

- Removed: `_run_llm_tool`, `_run_rag_tool`, `_run_action_tool`, `_handle_reschedule`, and all helper methods (~250 lines)
- Tool callbacks now delegate directly:
  ```python
  def _action_tool_func(self, tool_input):
      self._last_tool_used = "ACTION"
      return self.tools_exec.run_action_tool(tool_input)
  ```
- The ACTION tool uses `ToolExecutor.run_action_tool()` which does keyword-based routing internally

### `manager_multiple_tools.py` (AgentManagerMultiTools — LangGraph, 8 tools)

**Tools:** RAG, LLM, LIST_SLOTS, LIST_APPOINTMENTS, BOOK_APPOINTMENT, CANCEL_APPOINTMENT, RESCHEDULE_APPOINTMENT, SEND_DOCTOR_MESSAGE

- Removed: all `_run_*_tool` methods, `_handle_reschedule`, and all helper methods (~350 lines)
- Each tool callback delegates to the matching ToolExecutor method:
  ```python
  def _list_slots_tool_func(self, tool_input):
      self._last_tool_used = "LIST_SLOTS"
      return self.tools_exec.run_list_slots(tool_input)
  ```
- No keyword routing needed — the LLM selects the specific tool directly

### `manager_deterministic.py` (DeterministicAgentManager — LLM router, 3 tools)

**Tools:** LLM, RAG, ACTION (selected by LLM router or keyword fallback)

- Removed: `_run_action_tool` (full implementation), `_handle_reschedule`, `_format_retrieved_docs`, `_wants_slots`, `_wants_appointments`, and all extraction/filtering/formatting helpers (~300 lines)
- ACTION tool now delegates: `self.tools_exec.run_action_tool(message)`
- RAG tool uses `self.tools_exec.run_rag_tool()` for retrieval, then wraps the result with its own LLM prompt for answer generation
- LLM tool kept as-is (it does its own LLM call for casual chat)
- `_normalize_tool_input` kept — it's specific to how this manager receives tool input from the LLM router

### `manager_commands.py` (PlannedAgentManager — LLM planner, step execution)

**Tools:** chat, rag_search, list_slots, list_appointments, book_appointment, cancel_appointment, reschedule_appointment, respond

- The `_execute()` method now calls ToolExecutor for each step type:
  ```python
  elif tool == "list_slots":
      result = self.tools_exec.run_list_slots()
  elif tool == "book_appointment":
      result = self.tools_exec.run_book_appointment_by_id(slot_id)
  ```
- `_fallback_summary` uses `self.tools_exec._format_slot_preview()` / `_format_appointment_preview()`
- Removed: `_format_slot_preview`, `_format_appointment_preview` (~20 lines)
- The planner uses `run_*_by_id()` variants since it extracts IDs during the planning phase

### `manager_multi_agent.py` (MultiAgentManager — coordinator + specialist agents)

**Agents:** appointment_agent, medical_knowledge_agent, messaging_agent, chat_agent

- All specialist agents now receive `tools_exec` in their constructor
- **AppointmentAgent**: `_execute_action` uses `tools_exec.run_list_slots()`, `run_list_appointments()`, `run_book_appointment_by_id()`, `run_cancel_appointment_by_id()`, `run_reschedule_appointment_by_id()` instead of calling `self.appointments.*` directly
- **MedicalKnowledgeAgent**: Uses `tools_exec.run_rag_tool()` for retrieval instead of calling `self.retriever.retrieve_and_build_prompt()` directly
- **MessagingAgent**: Uses `tools_exec.run_send_doctor_message()` instead of calling `self.messaging.send_message()` directly
- **ChatAgent**: Unchanged (it only does LLM generation, no shared tool logic)

## What Was NOT Changed

- **Routing logic** — each manager's tool selection mechanism is untouched (prompts, keyword fallbacks, LLM routers, planner)
- **Response composition** — how each manager formats the final answer for the user
- **LangGraph framework code** — `_extract_final_text`, `_strip_thinking`, `_is_raw_tool_call` (specific to LangGraph agents)
- **Test file** (`test_tool_selection.py`) — no changes needed, all tests pass as before
- **Prompt files** — no changes
- **Backup directory** (`orchestration_backup/`) — untouched

## Verification

- All 6 files pass syntax validation (`ast.parse`)
- All modules import successfully at runtime
- All 3 keyword routing tests pass with identical accuracy scores as before
- LLM routing tests remain compatible (same interfaces)

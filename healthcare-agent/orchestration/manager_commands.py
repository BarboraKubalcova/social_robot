import json
import logging
import os
import re
import time
from typing import Any, Dict, List

from langchain_ollama import OllamaLLM

from orchestration.memory import ConversationMemory
from orchestration.tool_implementations import ToolExecutor
from execution.rag.retrieve import Retriever
from execution.actions.appointments import AppointmentManager
from execution.actions.messaging import MessageManager


logger = logging.getLogger("PlannedAgentManager")


# 8. "send_doctor_message"
# Args:
# {
#   "body": <string>
# }

class PlannedAgentManager:
    def __init__(self):
        self.memory = ConversationMemory()
        self.logger = logging.getLogger("PlannedAgentManager")
        self.max_history_turns = int(os.getenv("MAX_HISTORY_TURNS", "3"))

        self.model = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "qwen3:4b"))
        self.retriever = Retriever()
        self.appointments = AppointmentManager()
        self.messaging = MessageManager()

        self.tools_exec = ToolExecutor(self.retriever, self.appointments, self.messaging)

    async def process_message(self, user_message: str, user_id: str) -> Dict[str, Any]:
        self.memory.add_turn("user", user_message)
        history_text = self._build_history_text()
        time_start = time.time()
        print(f"The user message is: {user_message} ")
        plan_dict = self._plan(user_message, history_text)
        print(f"Time for planning: {time.time() - time_start:.2f} seconds")
        print(f"Generated plan: {plan_dict}")
        plan_dict = self._validate_plan(plan_dict)
        print(f"Time for validation: {time.time() - time_start:.2f} seconds")
        print(f"Validated plan: {plan_dict}")

        response = self._execute(plan_dict, user_message, history_text)
        print(f"Time for execution: {time.time() - time_start:.2f} seconds")
        intent = self._extract_intent(plan_dict)

        self.memory.add_turn("assistant", response)

        return {
            "response": response,
            "intent": intent,
            "history": self.memory.get_recent_history(),
        }

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan(self, user_input: str, history_text: str = "") -> dict:
        prompt = (
            self._planner_system_prompt()
            + "\n\nConversation history:\n"
            + history_text
            + "\n\nUser request:\n"
            + user_input
            + "\n\nReturn ONLY valid JSON."
        )

        raw = (self.model.invoke(prompt) or "").strip()

        try:
            return json.loads(raw)
        except Exception:
            pass

        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        return self._naive_backup_plan(user_input)

    def _planner_system_prompt(self) -> str:
        return """
You are a precise planner for a healthcare assistant system.

Your job:
Receive a natural-language user request and produce a valid JSON execution plan.
You do NOT answer the user directly.
You do NOT execute tools.
You only return a structured plan.

AVAILABLE TOOLS
You MUST use only these tool names:

1. "chat"
Args:
{
  "message": <string>
}
Use this only for casual conversation, greetings, emotional support,
or simple non-grounded replies.

2. "rag_search"
Args:
{
  "question": <string>
}
Use this for clinic procedures, MRI, CT, roentgen/X-ray, preparation instructions,
insurance/policy questions, treatment info, and grounded healthcare knowledge.

3. "list_slots"
Args: {}

4. "list_appointments"
Args: {}

5. "book_appointment"
Args:
{
  "slot_id": <string>
}
Use this to book a specific slot. Requires a slot_id (e.g. "slot_3").
If the user wants to book an appointment but does NOT specify a slot or slot_id,
use "list_slots" first so the user can see available options and choose one.
If the user says just a day and time, calculate the slot_id using the mapping below.

6. "cancel_appointment"
Args:
{
  "appointment_id": <string>
}

7. "reschedule_appointment"
Args:
{
  "appointment_id": <string>,
  "new_slot_id": <string>
}
If the user wants to reschedule but does not specify a slot, use "list_slots"
first to show available options.

8. "send_doctor_message"
Args:
{
  "body": <string>
}
Use this to send a message to the patient's doctor. The body should be in professional and respectful tone.

9. "respond"
Args:
{
  "style": "brief" | "normal"
}
Use this after backend tools when the system should summarize
the collected tool results into a final user-facing response.

SLOT ID MAPPING
Each day has 9 slots starting at 7:30, in 30-minute increments (last slot at 16:00).
- Monday: slot_1 to slot_9
- Tuesday: slot_10 to slot_18
- Wednesday: slot_19 to slot_27
For example, Monday 7:30 = slot_1, Monday 8:00 = slot_2, Tuesday 7:30 = slot_10, etc.
If the user mentions a day and time, compute the correct slot_id from this mapping.

RULES
- Output STRICT JSON only.
- Return exactly this schema:
  {
    "steps": [
      {"tool": "<tool_name>", "args": {...}},
      ...
    ]
  }
- The top-level JSON must contain ONLY the key "steps".
- Do not include explanations, markdown, comments, or extra fields.
- Tool names must exactly match the allowed tools above.
- If the request needs backend info or action, usually end with "respond".
- If the request is casual conversation only, use only one "chat" step.
- Prefer short plans.
- If the user wants to book or reschedule without specifying a slot,
  use "list_slots" first and then "respond" to let the user choose.

EXAMPLES

User: "Hi"
Output:
{
  "steps": [
    {"tool": "chat", "args": {"message": "Hi"}}
  ]
}

User: "What appointments do I have?"
Output:
{
  "steps": [
    {"tool": "list_appointments", "args": {}},
    {"tool": "respond", "args": {"style": "normal"}}
  ]
}

User: "Show me available slots"
Output:
{
  "steps": [
    {"tool": "list_slots", "args": {}},
    {"tool": "respond", "args": {"style": "brief"}}
  ]
}

User: "Show me available appointments"
Output:
{
  "steps": [
    {"tool": "list_slots", "args": {}},
    {"tool": "respond", "args": {"style": "brief"}}
  ]
}


User: "Book slot_12"
Output:
{
  "steps": [
    {"tool": "book_appointment", "args": {"slot_id": "slot_12"}},
    {"tool": "respond", "args": {"style": "brief"}}
  ]
}

User: "I want to book an appointment"
Output:
{
  "steps": [
    {"tool": "list_slots", "args": {}},
    {"tool": "respond", "args": {"style": "normal"}}
  ]
}

User: "I'd like to schedule a visit"
Output:
{
  "steps": [
    {"tool": "list_slots", "args": {}},
    {"tool": "respond", "args": {"style": "normal"}}
  ]
}

User: "Book me for Tuesday at 8:00"
Output:
{
  "steps": [
    {"tool": "book_appointment", "args": {"slot_id": "slot_11"}},
    {"tool": "respond", "args": {"style": "brief"}}
  ]
}

User: "Reschedule appt_2 to slot_9"
Output:
{
  "steps": [
    {"tool": "reschedule_appointment", "args": {"appointment_id": "appt_2", "new_slot_id": "slot_9"}},
    {"tool": "respond", "args": {"style": "brief"}}
  ]
}

User: "How should I prepare for an MRI?"
Output:
{
  "steps": [
    {"tool": "rag_search", "args": {"question": "How should I prepare for an MRI?"}},
    {"tool": "respond", "args": {"style": "normal"}}
  ]
}


Return only valid JSON.
"""

    def _naive_backup_plan(self, user_input: str) -> dict:
        text = user_input.lower()

        slot_match = re.search(r"\bslot[_\s-]?(\d+)\b", text)
        appt_match = re.search(r"\bappt[_\s-]?(\d+)\b", text)

        slot_id = f"slot_{slot_match.group(1)}" if slot_match else None
        appt_id = f"appt_{appt_match.group(1)}" if appt_match else None

        if any(k in text for k in ["available slots", "free slots", "show slots", "list slots"]):
            return {
                "steps": [
                    {"tool": "list_slots", "args": {}},
                    {"tool": "respond", "args": {"style": "brief"}}
                ]
            }

        if any(k in text for k in ["my appointments", "show appointments", "list appointments"]):
            return {
                "steps": [
                    {"tool": "list_appointments", "args": {}},
                    {"tool": "respond", "args": {"style": "normal"}}
                ]
            }

        if "reschedule" in text and appt_id and slot_id:
            return {
                "steps": [
                    {
                        "tool": "reschedule_appointment",
                        "args": {
                            "appointment_id": appt_id,
                            "new_slot_id": slot_id
                        }
                    },
                    {"tool": "respond", "args": {"style": "brief"}}
                ]
            }

        if "cancel" in text and appt_id:
            return {
                "steps": [
                    {"tool": "cancel_appointment", "args": {"appointment_id": appt_id}},
                    {"tool": "respond", "args": {"style": "brief"}}
                ]
            }

        if ("book" in text or "schedule" in text) and slot_id:
            return {
                "steps": [
                    {"tool": "book_appointment", "args": {"slot_id": slot_id}},
                    {"tool": "respond", "args": {"style": "brief"}}
                ]
            }

        if "message doctor" in text or "send message" in text or "email doctor" in text:
            return {
                "steps": [
                    {"tool": "send_doctor_message", "args": {"body": user_input}},
                    {"tool": "respond", "args": {"style": "brief"}}
                ]
            }

        rag_keywords = [
            "mri", "ct", "x-ray", "xray", "roentgen", "prepare", "preparation",
            "clinic", "insurance", "policy", "procedure", "treatment"
        ]
        if any(k in text for k in rag_keywords):
            return {
                "steps": [
                    {"tool": "rag_search", "args": {"question": user_input}},
                    {"tool": "respond", "args": {"style": "normal"}}
                ]
            }

        return {
            "steps": [
                {"tool": "chat", "args": {"message": user_input}}
            ]
        }

    def _validate_plan(self, plan_dict: dict) -> dict:
        allowed_tools = {
            "chat",
            "rag_search",
            "list_slots",
            "list_appointments",
            "book_appointment",
            "cancel_appointment",
            "reschedule_appointment",
            "send_doctor_message",
            "respond",
        }

        if not isinstance(plan_dict, dict):
            return self._naive_backup_plan("")

        steps = plan_dict.get("steps")
        if not isinstance(steps, list) or not steps:
            return self._naive_backup_plan("")

        cleaned = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            tool = step.get("tool")
            args = step.get("args", {})
            if tool not in allowed_tools:
                continue
            if not isinstance(args, dict):
                args = {}
            if not self._validate_args(tool, args):
                self.logger.warning("Invalid args for tool %s: %s", tool, args)
                continue
            cleaned.append({"tool": tool, "args": args})

        if not cleaned:
            return self._naive_backup_plan("")

        return {"steps": cleaned}

    @staticmethod
    def _validate_args(tool: str, args: dict) -> bool:
        """Validate that argument values match expected schemas."""
        slot_pattern = re.compile(r"^slot_\d+$")
        appt_pattern = re.compile(r"^appt_\d+$")

        if tool == "chat":
            return bool(isinstance(args.get("message"), str) and args["message"].strip())

        if tool == "rag_search":
            return bool(isinstance(args.get("question"), str) and args["question"].strip())

        if tool in ("list_slots", "list_appointments"):
            return True

        if tool == "book_appointment":
            slot_id = args.get("slot_id")
            return bool(isinstance(slot_id, str) and slot_pattern.match(slot_id))

        if tool == "cancel_appointment":
            appt_id = args.get("appointment_id")
            return bool(isinstance(appt_id, str) and appt_pattern.match(appt_id))

        if tool == "reschedule_appointment":
            appt_id = args.get("appointment_id")
            new_slot = args.get("new_slot_id")
            return bool(
                isinstance(appt_id, str) and appt_pattern.match(appt_id)
                and isinstance(new_slot, str) and slot_pattern.match(new_slot)
            )

        if tool == "send_doctor_message":
            return bool(isinstance(args.get("body"), str) and args["body"].strip())

        if tool == "respond":
            style = args.get("style", "normal")
            return style in ("brief", "normal")

        return True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute(self, plan_dict: dict, user_input: str, history_text: str = "") -> str:
        steps = plan_dict.get("steps", [])
        if not isinstance(steps, list) or not steps:
            return "Plan has no steps."

        execution_log: List[Dict[str, Any]] = []

        for idx, step in enumerate(steps, start=1):
            tool = step.get("tool")
            args = step.get("args", {}) or {}
            print(f"Executing step {idx}: tool={tool}, args={args}")

            try:
                if tool == "chat":
                    msg = str(args.get("message", user_input))
                    prompt = (
                        "You are a helpful healthcare assistant.\n"
                        "Be concise, warm, and clear.\n"
                        "Do not invent medical facts, appointments, or policy details.\n\n"
                        f"Conversation history:\n{history_text}\n\n"
                        f"User message:\n{msg}\n"
                    )
                    reply = (self.model.invoke(prompt) or "").strip()
                    return reply or "I’m here to help."

                elif tool == "rag_search":
                    question = str(args.get("question", user_input)).strip()
                    context_str, mode = self.tools_exec.run_rag_tool(question, history_text)

                    execution_log.append({
                        "tool": "rag_search",
                        "ok": bool(context_str.strip()),
                        "message": (
                            "Knowledge base search completed."
                            if context_str.strip()
                            else "No grounded context found."
                        ),
                        "data": {
                            "question": question,
                            "context": context_str,
                            "mode": mode,
                        }
                    })

                elif tool == "list_slots":
                    result = self.tools_exec.run_list_slots()
                    execution_log.append({
                        "tool": "list_slots",
                        "ok": True,
                        "message": result,
                        "data": {"slots": self.appointments.list_available_slots() or []}
                    })

                elif tool == "list_appointments":
                    result = self.tools_exec.run_list_appointments()
                    execution_log.append({
                        "tool": "list_appointments",
                        "ok": True,
                        "message": result,
                        "data": {"appointments": self.appointments.get_patient_appointments() or []}
                    })

                elif tool == "book_appointment":
                    slot_id = str(args.get("slot_id", "")).strip()
                    result = self.tools_exec.run_book_appointment_by_id(slot_id)
                    execution_log.append({
                        "tool": "book_appointment",
                        "ok": "successfully" in result.lower(),
                        "message": result,
                        "data": {"slot_id": slot_id}
                    })

                elif tool == "cancel_appointment":
                    appointment_id = str(args.get("appointment_id", "")).strip()
                    result = self.tools_exec.run_cancel_appointment_by_id(appointment_id)
                    execution_log.append({
                        "tool": "cancel_appointment",
                        "ok": "successfully" in result.lower() or "canceled" in result.lower(),
                        "message": result,
                        "data": {"appointment_id": appointment_id}
                    })

                elif tool == "reschedule_appointment":
                    appointment_id = str(args.get("appointment_id", "")).strip()
                    new_slot_id = str(args.get("new_slot_id", "")).strip()
                    result = self.tools_exec.run_reschedule_appointment_by_id(appointment_id, new_slot_id)
                    execution_log.append({
                        "tool": "reschedule_appointment",
                        "ok": "successfully" in result.lower(),
                        "message": result,
                        "data": {
                            "appointment_id": appointment_id,
                            "new_slot_id": new_slot_id,
                        }
                    })

                elif tool == "send_doctor_message":
                    body = str(args.get("body", "")).strip()
                    doctors = self.messaging.list_doctors()
                    recipient = doctors[0]["id"] if doctors else "doc_1"
                    ok = self.messaging.send_message(
                        recipient_id=recipient,
                        subject="Patient request",
                        body=body,
                    )
                    execution_log.append({
                        "tool": "send_doctor_message",
                        "ok": bool(ok),
                        "message": (
                            "Doctor message sent successfully."
                            if ok else
                            "Failed to send doctor message."
                        ),
                        "data": {
                            "recipient_id": recipient,
                            "body": body,
                        }
                    })

                elif tool == "respond":
                    print(f"Summarizing execution log for final response... {execution_log}")
                    return self._summarize_execution_log(
                        execution_log=execution_log,
                        user_input=user_input,
                        history_text=history_text,
                        style=str(args.get("style", "normal")),
                    )

                else:
                    return f"Unknown tool: {tool}"

            except Exception as e:
                self.logger.exception("Step %s (%s) failed", idx, tool)
                return f"Step {idx} ({tool}) failed: {e}"

        return self._summarize_execution_log(
            execution_log=execution_log,
            user_input=user_input,
            history_text=history_text,
            style="normal",
        )

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _summarize_execution_log(
        self,
        execution_log: List[Dict[str, Any]],
        user_input: str,
        history_text: str = "",
        style: str = "normal",
    ) -> str:
        if not execution_log:
            return "I could not find any result to respond with."

        prompt = (
            "You are a healthcare assistant writing the final reply to the patient.\n"
            "The patient cannot see the internal tool output.\n"
            "Internal tool output is private system data, not something to analyze.\n\n"
            "Your job:\n"
            "- Write a direct answer for the patient.\n"
            "- Do NOT mention JSON, execution logs, structure, fields, metadata, formatting, or observations.\n"
            "- Do NOT explain the data structure.\n"
            "- Do NOT critique, validate, or inspect the internal data.\n"
            "- Do NOT mention inconsistencies unless the patient explicitly asked for validation.\n"
            "- If the tool output lists appointments, simply list the appointments in natural language.\n"
            "- If the tool output lists slots, simply list the available slots in natural language.\n"
            "- If the tool output reports an action result, state whether it succeeded or failed.\n"
            "- Keep the response short, clear, and patient-friendly.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Patient request:\n{user_input}\n\n"
            f"Private internal tool output:\n{json.dumps(execution_log, ensure_ascii=False, indent=2)}\n\n"
            "Write only the final reply for the patient."
        )

        try:
            response = (self.model.invoke(prompt) or "").strip()
            if response:
                return response
        except Exception as e:
            self.logger.exception("Summarization failed: %s", e)

        return self._fallback_summary(execution_log)

    def _fallback_summary(self, execution_log: List[Dict[str, Any]]) -> str:
        last = execution_log[-1]
        tool = last.get("tool")
        data = last.get("data", {})

        if tool == "list_slots":
            slots = data.get("slots", [])
            if not slots:
                return "There are no available slots."
            preview = ", ".join(self.tools_exec._format_slot_preview(s) for s in slots[:8])
            more = f" ... and {len(slots) - 8} more." if len(slots) > 8 else ""
            return f"Available slots: {preview}{more}"

        if tool == "list_appointments":
            appts = data.get("appointments", [])
            if not appts:
                return "You have no appointments scheduled."
            preview = ", ".join(self.tools_exec._format_appointment_preview(a) for a in appts[:8])
            more = f" ... and {len(appts) - 8} more." if len(appts) > 8 else ""
            return f"Your appointments: {preview}{more}"

        if tool == "rag_search":
            context = data.get("context", "")
            if context:
                return context[:1200]
            return last.get("message", "I could not find grounded information.")

        return last.get("message", "The action was completed.")

    # ------------------------------------------------------------------
    # Intent + helpers
    # ------------------------------------------------------------------

    def _extract_intent(self, plan_dict: Dict[str, Any]) -> str:
        steps = plan_dict.get("steps", [])
        if not steps or not isinstance(steps, list):
            return "UNKNOWN"

        first_tool = steps[0].get("tool", "UNKNOWN")

        if first_tool == "rag_search":
            return "RAG"

        if first_tool in {
            "list_slots",
            "list_appointments",
            "book_appointment",
            "cancel_appointment",
            "reschedule_appointment",
            "send_doctor_message"
        }:
            return "ACTION"

        return "LLM_ONLY"

    def _build_history_text(self) -> str:
        lines = []
        for turn in self.memory.get_recent_history(limit=self.max_history_turns):
            content = turn["content"]
            if turn["role"] == "assistant":
                content = self._clean_response(content)
            if content:
                lines.append(f"{turn['role']}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _clean_response(text: str) -> str:
        """Extract only the human-readable response, stripping leaked JSON."""
        stripped = text.strip()
        if stripped.startswith("{"):
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict) and "response" in obj:
                    resp = obj["response"]
                    if isinstance(resp, str) and resp.strip():
                        return resp.strip()
            except (json.JSONDecodeError, ValueError):
                pass
        return stripped
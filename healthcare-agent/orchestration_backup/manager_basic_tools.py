import logging
import os
import re
import time
from typing import Any, Dict, Optional

from langchain_core.tools import Tool
from langchain.agents import create_agent

from orchestration.memory import ConversationMemory
from execution.ollama_client import OllamaClient
from execution.rag.retrieve import Retriever
from execution.actions.appointments import AppointmentManager
from execution.actions.messaging import MessageManager

from contextlib import contextmanager


# Takes ages because:
# LangGraph ReAct loop then does:

# Step 1 — LLM call: Sends the system prompt + tools descriptions + user message to qwen3. The LLM outputs a tool call (e.g., {"name": "RAG", "args": {"tool_input": "What is an MRI?"}}).
# Step 2 — Tool execution: LangGraph invokes the matching callback (_rag_tool_func, _llm_tool_func, or _action_tool_func). These are plain Python — they run the retriever, format a string, or do appointment logic.
# Step 3 — LLM call: The tool result is appended to the message history and sent back to the LLM. The LLM reads the tool output and generates the final natural-language response.
# LangGraph checks if the LLM wants to call another tool. If not, the loop ends.
# _extract_final_text() pulls the last assistant message from the result.

logger = logging.getLogger("AgentManager")


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s took %.2fs", label, time.perf_counter() - t0)


system_prompt = """
You are a healthcare clinic assistant. You help patients with appointment management, clinic-related questions, and general conversation.

For every user message, call exactly ONE tool, then use its output to write a short, helpful reply.

Tool selection:
- ACTION — Use when the patient wants to book, cancel, or reschedule an appointment, list available time slots, list their appointments, or send a message/email to a doctor.
  Examples: "Book me an appointment", "Cancel my visit", "What slots are free?", "Message my doctor"
- RAG — Use when the patient asks about medical procedures, clinic policies, preparation instructions, or anything that should be answered from the clinic knowledge base.
  Examples: "What is an MRI?", "How do I prepare for an ultrasound?", "Can I bring someone with me?", "What are visiting hours?"
- LLM — Use for greetings, casual chat, emotional support, thank-yous, and general questions that are not clinic-specific and do not require an action.
  Examples: "Hello!", "Thank you", "I'm feeling nervous", "What's the weather like?"

Rules:
- Call exactly one tool per message. Never call multiple tools.
- Do not repeat the raw tool output back to the patient.
- Use the tool result to compose a clear, concise, patient-friendly answer.
- If unsure between RAG and LLM, prefer RAG for anything health or clinic related.
"""


class AgentManager:
    """
    Agent-based healthcare assistant using exactly one tool per request.
    """

    def __init__(self) -> None:
        self.memory = ConversationMemory()
        self.max_history_turns = int(os.getenv("MAX_HISTORY_TURNS", "3"))

        self.llm = OllamaClient()
        self.retriever = Retriever()
        self.appointments = AppointmentManager()
        self.messaging = MessageManager()

        self._last_tool_used: Optional[str] = None
        self._current_history_text: str = ""

        self.tools = [
            Tool(
                name="RAG",
                description=(
                    "Use for questions that should be grounded in the knowledge base, "
                    "such as medical procedures, clinic rules/policies, and preparation "
                    "guidelines (e.g., 'What is an MRI?', 'How do I prepare for an ultrasound?', "
                    "'Can I bring someone with me?', 'Can I wear jewelry?').  "
    
                ),
                func=self._rag_tool_func,
            ),
            Tool(
                name="LLM",
                description=(
                    "Use for casual chat, greetings, emotional support, and general questions "
                    "not requiring retrieval or operational actions."
                ),
                func=self._llm_tool_func,
            ),
            Tool(
                name="ACTION",
                description=(
                    "Use for operations: booking, canceling, rescheduling appointments, "
                    "listing slots, listing appointments, or sending a doctor message. "
                    "If the user gives an exact time instead of a slot id, interpret it from "
                    "the available slot order when possible."
                ),
                func=self._action_tool_func,
            ),
        ]

        self.tools_agent = create_agent(
            model=self.llm.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=False,
        )

        logger.info("AgentManager initialized")

    async def process_message(self, user_message: str, user_id: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        logger.info("User=%s msg=%s", user_id, user_message)

        self.memory.add_turn("user", user_message)
        self._current_history_text = self._build_history_text()
        self._last_tool_used = None

        # response, intent = self._route_and_respond(user_message)
        response, intent = self._route_and_respond(user_message, user_id)

        self.memory.add_turn("assistant", response)

        logger.info(
            "Done user=%s intent=%s total=%.2fs",
            user_id,
            intent,
            time.perf_counter() - t0,
        )

        return {
            "response": response,
            "intent": intent,
            "history": self.memory.get_recent_history(),
        }

    # def _route_and_respond(self, user_message: str) -> tuple[str, str]:
    #     result = self.tools_agent.invoke(
    #         {
    #             "messages": [
    #                 {"role": "user", "content": user_message}
    #             ]
    #         }
    #     )
    #     response = self._extract_final_text(result)
    #     intent = self._resolve_intent()
    #     return response, intent
    def _route_and_respond(self, user_message: str, user_id: str) -> tuple[str, str]:
        # Build a trimmed message list from ConversationMemory (already capped
        # at max_history_turns) so the LLM context doesn't grow unbounded.
        recent = self.memory.get_recent_history(limit=self.max_history_turns)
        messages = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in recent
        ]
        # Ensure the current user message is included (it was already added
        # to memory before this call, but guard against edge cases).
        if not messages or messages[-1]["content"] != user_message:
            messages.append({"role": "user", "content": user_message})

        result = self.tools_agent.invoke({"messages": messages})
        response = self._extract_final_text(result)
        intent = self._resolve_intent()
        return response, intent

    # -------- Tool callbacks --------

    def _rag_tool_func(self, tool_input: str) -> str:
        self._last_tool_used = "RAG"
        text, _ = self._run_rag_tool(tool_input)
        return text

    def _llm_tool_func(self, tool_input: str) -> str:
        self._last_tool_used = "LLM"
        return self._run_llm_tool(tool_input)

    def _action_tool_func(self, tool_input: str) -> str:
        self._last_tool_used = "ACTION"
        return self._run_action_tool(tool_input)

    # -------- Core implementations --------

    def _build_history_text(self) -> str:
        turns = self.memory.get_recent_history(limit=self.max_history_turns)
        return "\n".join(f"{t['role']}: {t['content']}" for t in turns)

    def _run_llm_tool(self, message: str) -> str:
        # f"Conversation history:\n{self._current_history_text}\n"
        return (
            f"Task type: casual conversation or general support.\n"
            f"User message:\n{message}\n"
            f"Instruction: Reply briefly, warmly, and helpfully."
        )

    def _run_rag_tool(self, message: str) -> tuple[str, str]:
        with timed("retriever.retrieve_and_build_prompt"):
            prompt_template, prompt_kwargs, mode = self.retriever.retrieve_and_build_prompt(
                message,
                self._current_history_text,
            )

        context = prompt_kwargs.get("context", "")
        question = prompt_kwargs.get("question", message)
        return prompt_template, mode
        # return (
        #     f"task=grounded_healthcare_answer\n"
        #     f"question={question}\n"
        #     f"context={context}\n"
        #     f"instruction=Answer using only the context. Keep it concise."
        # ), mode


    def _run_action_tool(self, message: str) -> str:
        msg = message.lower()

        slot_id = self._extract_slot_id(msg)
        appt_id = self._extract_appointment_id(msg)

        if self._wants_slots(msg):
            slots = self.appointments.list_available_slots()
            if not slots:
                return "There are no available slots in the current timetable."
            preview = ", ".join(
                self._format_slot_preview(s) for s in slots[:8]
            )
            return f"Here are available slots: {preview}"

        if re.search(r"\b(reschedule|rebook|move|change)\b", msg):
            return self._handle_reschedule(message, msg, appt_id, slot_id)

        if re.search(r"\bcancel\b", msg):
            if not appt_id:
                appts = self.appointments.get_patient_appointments()
                if not appts:
                    return "There is no active appointment to cancel."
                if len(appts) > 1:
                    preview = ", ".join(
                        self._format_appointment_preview(a) for a in appts[:5]
                    )
                    return (
                        "You have multiple appointments. "
                        f"Please specify which one to cancel: {preview}"
                    )
                appt_id = appts[0]["id"]

            ok = self.appointments.cancel_appointment(appt_id)
            return (
                f"Appointment {appt_id} has been canceled."
                if ok
                else f"I could not cancel appointment {appt_id}."
            )

        if re.search(r"\bbook\b", msg) or slot_id:
            if not slot_id:
                available = self.appointments.list_available_slots()
                if not available:
                    return "There are no available slots to book right now."

                requested_day = self._extract_requested_day(msg)
                if requested_day:
                    matched = self._filter_slots_by_day(available, requested_day)
                    if matched:
                        slot_id = matched[0]["id"]
                    else:
                        return f"I could not find an available slot for {requested_day}."
                else:
                    slot_id = available[0]["id"]

            ok = self.appointments.book_appointment(slot_id)
            return (
                f"I have booked {slot_id} for you."
                if ok
                else f"I could not book {slot_id}. It may already be reserved or invalid."
            )

        if "send message" in msg or "email" in msg:
            doctors = self.messaging.list_doctors()
            recipient = doctors[0]["id"] if doctors else "doc_1"
            ok = self.messaging.send_message(
                recipient_id=recipient,
                subject="Patient request",
                body=message,
            )
            return (
                "I have sent your message to the doctor."
                if ok
                else "I could not send the message."
            )

        return "I can help with booking, canceling, rescheduling appointments, listing slots, or sending a doctor message."
    # -------- Helpers --------

    def _handle_reschedule(
        self,
        original_message: str,
        msg_lower: str,
        appt_id: Optional[str],
        slot_id: Optional[str],
    ) -> str:
        # Old behavior still works if user provides both ids explicitly.
        if appt_id and slot_id:
            ok = self.appointments.reschedule_appointment(appt_id, slot_id)
            return (
                f"Appointment {appt_id} was rescheduled to {slot_id}."
                if ok
                else f"I could not reschedule {appt_id} to {slot_id}."
            )

        appts = self.appointments.get_patient_appointments()
        if not appts:
            return "There is no active appointment to reschedule."

        # If appointment id not given, try to infer it.
        target_appt = None
        if appt_id:
            target_appt = next((a for a in appts if a.get("id") == appt_id), None)
            if not target_appt:
                return f"I could not find appointment {appt_id}."
        else:
            mentioned_old_day = self._extract_source_day(msg_lower)
            matched_appts = (
                self._filter_appointments_by_day(appts, mentioned_old_day)
                if mentioned_old_day
                else appts
            )

            if len(matched_appts) == 1:
                target_appt = matched_appts[0]
            elif len(matched_appts) > 1:
                preview = ", ".join(
                    self._format_appointment_preview(a) for a in matched_appts[:5]
                )
                return (
                    "I found multiple matching appointments to reschedule. "
                    f"Please specify which one: {preview}"
                )
            else:
                return "I could not find the appointment you want to reschedule."

        # If slot id not given, try to infer new slot from requested day.
        target_slot = None
        if slot_id:
            slots = self.appointments.list_available_slots()
            target_slot = next((s for s in slots if s.get("id") == slot_id), None)
            if not target_slot:
                return f"I could not find slot {slot_id}."
        else:
            requested_new_day = self._extract_requested_day(msg_lower)
            if not requested_new_day:
                return (
                    "Please provide the new day or slot id for rescheduling "
                    "(for example: 'move it to Thursday' or 'to slot_12')."
                )

            available = self.appointments.list_available_slots()
            matching_slots = self._filter_slots_by_day(available, requested_new_day)

            if not matching_slots:
                return f"I could not find any available slots for {requested_new_day}."

            target_slot = matching_slots[0]

        # Safe order: find current -> find new -> cancel -> book
        old_appt_id = target_appt["id"]
        new_slot_id = target_slot["id"]

        cancel_ok = self.appointments.cancel_appointment(old_appt_id)
        if not cancel_ok:
            return f"I found appointment {old_appt_id}, but I could not cancel it."

        book_ok = self.appointments.book_appointment(new_slot_id)
        if not book_ok:
            return (
                f"I canceled {old_appt_id}, but I could not book {new_slot_id}. "
                "Manual review is needed."
            )

        return (
            f"Your appointment {old_appt_id} was rescheduled to {self._format_slot_preview(target_slot)}."
        )


    def _extract_requested_day(self, message_lower: str) -> Optional[str]:
        days = (
            "monday", "tuesday", "wednesday",
            "thursday", "friday", "saturday", "sunday"
        )

        m = re.search(
            r"\b(?:to|for|on)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            message_lower,
        )
        if m:
            return m.group(1)

        for day in days:
            if re.search(rf"\b{day}\b", message_lower):
                return day

        return None


    def _extract_source_day(self, message_lower: str) -> Optional[str]:
        m = re.search(
            r"\bfrom\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            message_lower,
        )
        return m.group(1) if m else None


    def _filter_slots_by_day(self, slots: list[dict], day_name: str) -> list[dict]:
        matched = []
        for slot in slots:
            haystack = " ".join(
                str(slot.get(k, "")).lower()
                for k in ("day", "date", "datetime", "label", "time")
            )
            if day_name.lower() in haystack:
                matched.append(slot)
        return matched


    def _filter_appointments_by_day(self, appointments: list[dict], day_name: Optional[str]) -> list[dict]:
        if not day_name:
            return appointments

        matched = []
        for appt in appointments:
            haystack = " ".join(
                str(appt.get(k, "")).lower()
                for k in ("day", "date", "datetime", "label", "time")
            )
            if day_name.lower() in haystack:
                matched.append(appt)
        return matched


    def _format_slot_preview(self, slot: dict) -> str:
        slot_id = slot.get("id", "unknown_slot")
        day = slot.get("day") or slot.get("date") or ""
        time_val = slot.get("time") or ""
        extra = " ".join(part for part in [str(day).strip(), str(time_val).strip()] if part).strip()
        return f"{slot_id} ({extra})" if extra else slot_id


    def _format_appointment_preview(self, appt: dict) -> str:
        appt_id = appt.get("id", "unknown_appt")
        day = appt.get("day") or appt.get("date") or ""
        time_val = appt.get("time") or ""
        extra = " ".join(part for part in [str(day).strip(), str(time_val).strip()] if part).strip()
        return f"{appt_id} ({extra})" if extra else appt_id


    def _extract_final_text(self, result: Any) -> str:
        if not isinstance(result, dict):
            text = str(result).strip()
            return text if text else "I'm sorry, I couldn't process your request."

        messages = result.get("messages") or []
        for msg in reversed(messages):
            role = (
                msg.get("role")
                if isinstance(msg, dict)
                else getattr(msg, "role", None) or getattr(msg, "type", None)
            )
            content = (
                msg.get("content")
                if isinstance(msg, dict)
                else getattr(msg, "content", None)
            )

            tool_calls = (
                msg.get("tool_calls")
                if isinstance(msg, dict)
                else getattr(msg, "tool_calls", None)
            )
            if tool_calls:
                continue

            text = self._extract_text_content(content)
            if role in ("assistant", "ai", "tool") and text:
                if self._is_raw_tool_call(text):
                    continue
                return text

        return "I'm sorry, I couldn't process your request."

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks and \\boxed{} that leak from qwen3."""
        # Full <think>...</think> block
        text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
        # Missing opening tag: everything before </think>
        text = re.sub(r"^[\s\S]*?</think>\s*", "", text)
        # Strip LaTeX \boxed{...} wrapper (possibly with $$ delimiters)
        m = re.search(r"\\boxed\{(.+?)\}", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        text = re.sub(r"\$\$\s*", "", text)
        return text.strip()

    @staticmethod
    def _is_raw_tool_call(text: str) -> bool:
        """Detect text that is a raw JSON tool call leaked by the model."""
        stripped = text.strip()
        if not stripped.startswith("{"):
            return False
        try:
            import json
            obj = json.loads(stripped)
            return isinstance(obj, dict) and (
                "name" in obj or "tool" in obj
            ) and ("arguments" in obj or "args" in obj)
        except (json.JSONDecodeError, ValueError):
            return False

    def _extract_text_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("text"):
                    parts.append(block["text"])
            return self._strip_thinking("\n".join(parts).strip())
        return self._strip_thinking(str(content).strip())

    def _resolve_intent(self) -> str:
        return (self._last_tool_used or "LLM_ONLY").upper()

    def _wants_slots(self, msg_lower: str) -> bool:
        return any(
            phrase in msg_lower
            for phrase in (
                "available slots",
                "available appointments",
                "which appointments are available",
                "when can i come",
                "list slots",
                "show slots",
                "list available appointments",
                "show available appointments",
                "list available slots",
                "show available slots",
            )
        )

    def _extract_slot_id(self, message_lower: str) -> Optional[str]:
        m = re.search(r"\bslot[_\s-]?(\d+)\b", message_lower)
        return f"slot_{m.group(1)}" if m else None

    def _extract_appointment_id(self, message_lower: str) -> Optional[str]:
        m = re.search(r"\bappt[_\s-]?(\d+)\b", message_lower)
        return f"appt_{m.group(1)}" if m else None
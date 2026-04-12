import logging
import os
import re
import time
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import Tool, StructuredTool
from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError

from orchestration.memory import ConversationMemory
from orchestration.tool_implementations import ToolExecutor
from execution.ollama_client import OllamaClient
from execution.rag.retrieve import Retriever
from execution.actions.appointments import AppointmentManager
from execution.actions.messaging import MessageManager

from contextlib import contextmanager


logger = logging.getLogger("AgentManagerMultiTools")


# @contextmanager
# def timed(label: str):
#     t0 = time.perf_counter()
#     try:
#         yield
#     finally:
#         logger.info("%s took %.2fs", label, time.perf_counter() - t0)


system_prompt = """
You are a friendly healthcare clinic assistant helping patients.

For each user message, call exactly ONE tool, then use its output to write a short, helpful reply.

IMPORTANT RULES:
- Reply ONLY with the final answer for the patient. Do NOT explain your reasoning.
- Do NOT describe which tool you chose or why.
- Do NOT use LaTeX, math notation, or \\boxed{}.
- Keep responses concise, warm, and patient-friendly.
- Do not repeat the raw tool output back to the patient (especially if its in json format).
- You MUST call a tool for EVERY message. NEVER answer directly without a tool call. 
- Answer in maximum 7 sentences when using the RAG tool, and 3 sentences for general LLM responses.

For the appointment booking and rescheduling tools, the user can say just a day and time.
Each day has 9 slots at: 07:30, 08:30, 09:30, 10:30, 11:30, 13:00, 14:00, 15:00, 16:00.
- Monday: slot_1=07:30, slot_2=08:30, slot_3=09:30, slot_4=10:30, slot_5=11:30, slot_6=13:00, slot_7=14:00, slot_8=15:00, slot_9=16:00
- Tuesday: slot_10=07:30, slot_11=08:30, slot_12=09:30, slot_13=10:30, slot_14=11:30, slot_15=13:00, slot_16=14:00, slot_17=15:00, slot_18=16:00
- Wednesday: slot_19=07:30, slot_20=08:30, slot_21=09:30, slot_22=10:30, slot_23=11:30, slot_24=13:00, slot_25=14:00, slot_26=15:00, slot_27=16:00
If the user tries to book or reschedule without specifying a slot, use the LIST_SLOTS tool to show 
them available options and prompt them to choose one. You can inform user abou calculation of the slot id
and book it using the BOOK_APPOINTMENT tool with the slot_id argument.


Tool guide:
- LIST_SLOTS: show available appointment slots
- LIST_APPOINTMENTS: show booked appointments
- BOOK_APPOINTMENT: book a specific slot (requires slot_id, e.g. 'slot_3')
- CANCEL_APPOINTMENT: cancel a specific appointment (requires appointment_id, e.g. 'appt_2')
- RESCHEDULE_APPOINTMENT: move an appointment to a new slot (requires appointment_id and new_slot_id)
- SEND_DOCTOR_MESSAGE: send a message to a doctor (requires the message text)
- RAG: clinic knowledge, medical procedures, policies, preparation instructions
- LLM: greetings, empathy, casual chat, thank-yous, general questions not requiring retrieval or actions
"""


# -------- Pydantic input schemas for structured tools --------

class BookAppointmentInput(BaseModel):
    slot_id: str = Field(description="The slot ID to book, e.g. 'slot_3'")


# class BookAppointmentInput(BaseModel):
#     slot_id: str = Field(description="The slot ID to book, e.g. 'slot_3'")


class CancelAppointmentInput(BaseModel):
    appointment_id: str = Field(description="The appointment ID to cancel, e.g. 'appt_2'")


class RescheduleAppointmentInput(BaseModel):
    appointment_id: str = Field(description="The appointment ID to reschedule, e.g. 'appt_2'")
    new_slot_id: str = Field(description="The new slot ID to move the appointment to, e.g. 'slot_5'")


class SendDoctorMessageInput(BaseModel):
    message: str = Field(description="The message body to send to the doctor")


class EmptyInput(BaseModel):
    pass


class AgentManagerMultiTools:
    """
    Agent-based healthcare assistant using exactly one tool per request.
    """

    def __init__(self) -> None:
        self.memory = ConversationMemory()
        self.max_history_turns = int(os.getenv("MAX_HISTORY_TURNS", "1"))

        self.llm = OllamaClient()
        self.retriever = Retriever()
        self.appointments = AppointmentManager()
        self.messaging = MessageManager()

        self.tools_exec = ToolExecutor(self.retriever, self.appointments, self.messaging)

        self._last_tool_used: Optional[str] = None
        self._current_history_text: str = ""

        self.tools = [
            Tool(
                name="RAG",
                description=(
                    "Use for questions that should be grounded in the knowledge base, "
                    "such as medical procedures, clinic rules/policies, preparation "
                    "guidelines (e.g., 'What is an MRI?', 'How do I prepare for an ultrasound?', "
                    "'Can I bring someone with me?'), questions about MRI, CT and ultrasound. " \
                    "If patient wants to know about preparation for preparation, for example fasting policy," \
                    " or if they ask about what to expect during a procedure, or any other question that requires"
                    " specific knowledge about the clinic's operations, policies, or medical information, use RAG. " \
                    "Answer in max 7 sentences."
                ),
                func=self._rag_tool_func,
            ),
            Tool(
                name="LLM",
                description=(
                    "Use for casual chat, greetings, emotional support, and general questions "
                    "not requiring retrieval or operational actions. Answer in max 3 sentences."
                ),
                func=self._llm_tool_func,
            ),
            StructuredTool(
                name="LIST_SLOTS",
                description=(
                    "Use when the user wants to see available slots or asks when they can come, "
                    "what appointments are available, or to show/list slots. Show all slots, "
                    "not just a few (e.g. don't say 'I have a slot on Monday at 10am' if there "
                    "are actually 10 slots available - instead list them all or say 'I have "
                    "several slots available, including Monday at 10am, Tuesday at 2pm, etc.'). " 
                    "Also use this tool if the user tries to book an appointment without specifying "
                    "a slot, so you can prompt them with available options. Also use this tool" 
                    "if a user asks to list available appointment times or something similar, because"
                    "he does not want to see his appointments, but wants to book some."
                ),
                func=self._list_slots_tool_func, 
                args_schema=EmptyInput,
            ),
            StructuredTool(
                name="LIST_APPOINTMENTS",
                description=(
                    "Use when the user wants to see their current booked appointments." 
                    "Use this tool if the user asks to list their appointments, or if they "
                    "ask to cancel or reschedule an appointment without specifying which one "
                    "- you can show them their current appointments with this tool so they "
                    "can choose. " 
                ),
                func=self._list_appointments_tool_func, 
                args_schema=EmptyInput,
            ),
            StructuredTool(
                name="BOOK_APPOINTMENT",
                description=(
                    "Book an appointment for a specific slot. "
                    "Requires the slot_id (e.g. 'slot_3'). "
                    "Use LIST_SLOTS first if the user has not specified a slot."
                    "SLOT ID MAPPING in case the user provides a day and time: "
                    "Each day has 9 slots at: 07:30, 08:30, 09:30, 10:30, 11:30, 13:00, 14:00, 15:00, 16:00. "
                    "Monday: slot_1=07:30, slot_2=08:30, slot_3=09:30, slot_4=10:30, slot_5=11:30, slot_6=13:00, slot_7=14:00, slot_8=15:00, slot_9=16:00. "
                    "Tuesday: slot_10=07:30, slot_11=08:30, slot_12=09:30, slot_13=10:30, slot_14=11:30, slot_15=13:00, slot_16=14:00, slot_17=15:00, slot_18=16:00. "
                    "Wednesday: slot_19=07:30, slot_20=08:30, slot_21=09:30, slot_22=10:30, slot_23=11:30, slot_24=13:00, slot_25=14:00, slot_26=15:00, slot_27=16:00. "
                    "If the user mentions a day and time, find the matching slot_id from this mapping."
                ),
                func=self._book_appointment_tool_func,
                args_schema=BookAppointmentInput,
            ),
            StructuredTool(
                name="CANCEL_APPOINTMENT",
                description=(
                    "Cancel an existing appointment. "
                    "Requires the appointment_id (e.g. 'appt_2'). "
                    "Use LIST_APPOINTMENTS first if the user has not specified an appointment."
                ),
                func=self._cancel_appointment_tool_func,
                args_schema=CancelAppointmentInput,
            ),
            StructuredTool(
                name="RESCHEDULE_APPOINTMENT",
                description=(
                    "Move an existing appointment to a different slot. "
                    "Requires the appointment_id and the new_slot_id." 
                    "If the user just says 'I want to reschedule my appointment' "
                    "without giving details, use LIST_APPOINTMENTS to show them their "
                    "current appointments and LIST_SLOTS to show them available slots, "
                    "so they can specify which appointment to move and where."
                ),
                func=self._reschedule_appointment_tool_func,
                args_schema=RescheduleAppointmentInput,
            ),
            StructuredTool(
                name="SEND_DOCTOR_MESSAGE",
                description=(
                    "Send a message or email to the patient's doctor. "
                    "Requires the message text. Choose this tool if a user want to contact their doctor, "
                    "send a message to their doctor, or ask something that should be relayed to their doctor. " 
                    "Write the message in professional and respectful tone for the doctor."
                ),
                func=self._send_doctor_message_tool_func,
                args_schema=SendDoctorMessageInput,
            ),
        ]

        self.tools_agent = create_agent(
            model=self.llm.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=False,
        )

        logger.info("AgentManagerMultiTools initialized")

    async def process_message(self, user_message: str, user_id: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        logger.info("User=%s msg=%s", user_id, user_message)

        self.memory.add_turn("user", user_message)
        self._current_history_text = self._build_history_text()
        self._last_tool_used = None

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

    def _route_and_respond(self, user_message: str, user_id: str) -> tuple[str, str]:
        # Build a trimmed message list from ConversationMemory (already capped
        # at max_history_turns) so the LLM context doesn't grow unbounded.
        recent = self.memory.get_recent_history(limit=self.max_history_turns)
        messages = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in recent
        ]
        if not messages or messages[-1]["content"] != user_message:
            messages.append({"role": "user", "content": user_message})

        try:
            result = self.tools_agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 6},
            )
        except GraphRecursionError:
            logger.warning("Agent hit recursion limit for: %s", user_message)
            return "I'm sorry, I had trouble processing that. Could you rephrase?", "ERROR"

        # Log all tool calls from the agent result
        if isinstance(result, dict):
            for msg in result.get("messages", []):
                tool_calls = (
                    msg.get("tool_calls")
                    if isinstance(msg, dict)
                    else getattr(msg, "tool_calls", None)
                )
                if tool_calls:
                    logger.info("Tool calls: %s", tool_calls)
                    print(f"[AGENT DEBUG] Tool calls: {tool_calls}")

        response = self._extract_final_text(result)
        intent = self._resolve_intent()
        return response, intent

    # -------- Tool callbacks --------

    def _rag_tool_func(self, tool_input: str) -> str:
        self._last_tool_used = "RAG"
        text, _ = self.tools_exec.run_rag_tool(tool_input, self._current_history_text)
        return text

    def _llm_tool_func(self, tool_input: str) -> str:
        self._last_tool_used = "LLM"
        return self.tools_exec.run_llm_tool(tool_input)

    def _list_slots_tool_func(self) -> str:
        self._last_tool_used = "LIST_SLOTS"
        return self.tools_exec.run_list_slots()

    def _list_appointments_tool_func(self) -> str:
        self._last_tool_used = "LIST_APPOINTMENTS"
        return self.tools_exec.run_list_appointments()

    def _book_appointment_tool_func(self, slot_id: str) -> str:
        self._last_tool_used = "BOOK_APPOINTMENT"
        # If the model passed a proper slot ID, use the direct method;
        # otherwise fall back to text-based parsing (handles "Monday 7:30" etc.)
        if re.match(r"^slot[_\s-]?\d+$", slot_id.strip(), re.IGNORECASE):
            return self.tools_exec.run_book_appointment_by_id(slot_id.strip())
        return self.tools_exec.run_book_appointment(slot_id)

    def _cancel_appointment_tool_func(self, appointment_id: str) -> str:
        self._last_tool_used = "CANCEL_APPOINTMENT"
        return self.tools_exec.run_cancel_appointment_by_id(appointment_id)

    def _reschedule_appointment_tool_func(self, appointment_id: str, new_slot_id: str) -> str:
        self._last_tool_used = "RESCHEDULE_APPOINTMENT"
        return self.tools_exec.run_reschedule_appointment_by_id(appointment_id, new_slot_id)

    def _send_doctor_message_tool_func(self, message: str) -> str:
        self._last_tool_used = "SEND_DOCTOR_MESSAGE"
        return self.tools_exec.run_send_doctor_message(message)

    # -------- Core implementations --------

    def _build_history_text(self) -> str:
        turns = self.memory.get_recent_history(limit=self.max_history_turns)
        return "\n".join(f"{t['role']}: {t['content']}" for t in turns)

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
            if role in ("assistant", "ai") and text:
                if self._is_raw_tool_call(text):
                    # Try to salvage a "response" field from the leaked JSON
                    salvaged = self._try_extract_response_from_json(text)
                    if salvaged:
                        return salvaged
                    continue
                return text
            # For tool messages, skip bare tool names / very short outputs
            # that are just echoed tool names rather than real results
            if role == "tool" and text and not self._is_bare_tool_name(text):
                if self._is_raw_tool_call(text):
                    salvaged = self._try_extract_response_from_json(text)
                    if salvaged:
                        return salvaged
                    continue
                return text

        return "I'm sorry, I couldn't process your request."

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

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks and \\boxed{} that leak from qwen3."""
        text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
        text = re.sub(r"^[\s\S]*?</think>\s*", "", text)
        # Strip LaTeX \boxed{...} wrapper (possibly with $$ delimiters)
        m = re.search(r"\\boxed\{(.+?)\}", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        text = re.sub(r"\$\$\s*", "", text)
        return text.strip()

    _TOOL_NAMES = {
        "RAG", "LLM", "LIST_SLOTS", "LIST_APPOINTMENTS",
        "BOOK_APPOINTMENT", "CANCEL_APPOINTMENT",
        "RESCHEDULE_APPOINTMENT", "SEND_DOCTOR_MESSAGE",
    }

    @classmethod
    def _is_bare_tool_name(cls, text: str) -> bool:
        """Detect text that is just a tool name echoed back."""
        return text.strip().upper() in cls._TOOL_NAMES

    @staticmethod
    def _is_raw_tool_call(text: str) -> bool:
        """Detect text that is a raw JSON tool call leaked by the model."""
        stripped = text.strip()
        if not stripped.startswith("{"):
            return False
        try:
            import json
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                return False
            # Direct tool call: {"name": ..., "arguments": ...}
            if ("name" in obj or "tool" in obj) and ("arguments" in obj or "args" in obj):
                return True
            # Wrapped tool call: {"tool_call": {...}, ...}
            if "tool_call" in obj:
                return True
            return False
        except (json.JSONDecodeError, ValueError):
            return False

    @staticmethod
    def _try_extract_response_from_json(text: str) -> Optional[str]:
        """If the model leaked JSON with a 'response' field, extract it."""
        stripped = text.strip()
        if not stripped.startswith("{"):
            return None
        try:
            import json
            obj = json.loads(stripped)
            if isinstance(obj, dict) and "response" in obj:
                resp = obj["response"]
                if isinstance(resp, str) and resp.strip():
                    return resp.strip()
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _resolve_intent(self) -> str:
        return (self._last_tool_used or "LLM_ONLY").upper()

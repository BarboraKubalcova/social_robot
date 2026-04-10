import logging
import re
from typing import Dict, Any

from orchestration.tool_implementations import ToolExecutor
from execution.ollama_client import OllamaClient
from execution.rag.retrieve import Retriever
from execution.actions.appointments import AppointmentManager
from execution.actions.messaging import MessageManager


class DeterministicAgentManager:
    """
    Purely keyword-based router with no memory or conversation history.
    Routes to individual tools, then uses the LLM only to produce a
    human-friendly answer from the tool result.
    """

    TOOLS = [
        "LIST_SLOTS",
        "LIST_APPOINTMENTS",
        "BOOK_APPOINTMENT",
        "CANCEL_APPOINTMENT",
        "RESCHEDULE_APPOINTMENT",
        "SEND_DOCTOR_MESSAGE",
        "RAG",
        "LLM",
    ]

    def __init__(self):
        self.logger = logging.getLogger("DeterministicAgentManager")

        self.llm = OllamaClient()
        self.retriever = Retriever()
        self.appointments = AppointmentManager()
        self.messaging = MessageManager()

        self.tools_exec = ToolExecutor(self.retriever, self.appointments, self.messaging)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def process_message(self, user_message: str, user_id: str) -> Dict[str, Any]:
        tool, reason = self._route(user_message)
        tool_result = self._execute_tool(tool, user_message)
        response = self._generate_response(tool, tool_result, user_message)

        return {
            "response": response,
            "intent": tool,
            "router_reason": reason,
            "history": [],
        }

    # ------------------------------------------------------------------
    # Keyword-only router
    # ------------------------------------------------------------------

    def _route(self, message: str) -> tuple[str, str]:
        msg = message.lower()

        # --- action tools (checked first, most specific first) ---

        # Reschedule
        if re.search(r"\b(reschedule|rebook|move my appointment|change my appointment)\b", msg):
            return "RESCHEDULE_APPOINTMENT", "Keyword: reschedule/rebook/move appointment"

        # Cancel
        if re.search(r"\bcancel\b", msg) and re.search(r"\b(appointment|booking)\b", msg):
            return "CANCEL_APPOINTMENT", "Keyword: cancel appointment"

        # Book
        if re.search(r"\b(book|schedule|reserve)\b", msg):
            return "BOOK_APPOINTMENT", "Keyword: book/schedule/reserve"
        if re.search(r"\bslot[_\s-]?\d+\b", msg):
            return "BOOK_APPOINTMENT", "Keyword: explicit slot id mentioned"

        # Send doctor message (check before LIST_SLOTS to avoid false matches on "write"/"message")
        if re.search(
            r"\b(message\s+(my\s+|the\s+)?(doctor|dr)|send\s+(a\s+)?message\s+to\s+(my\s+|the\s+)?(doctor|dr)|"
            r"email\s+(my\s+|the\s+)?(doctor|dr)|write\s+(a\s+)?(message\s+)?to\s+(my\s+|the\s+)?(doctor|dr)|"
            r"contact\s+(my\s+|the\s+)?(doctor|dr)|tell\s+(my\s+|the\s+)?(doctor|dr)|"
            r"send\s+(a\s+)?message\s+to\s+(a\s+)?(doctor|dr))\b",
            msg,
        ):
            return "SEND_DOCTOR_MESSAGE", "Keyword: message/email doctor"

        # List available slots
        if re.search(
            r"\b(available\s+(slots?|appointments?|times?)|free\s+(slots?|times?)|"
            r"open\s+slots?|when\s+can\s+i\s+come|show\s+slots?|list\s+slots?|"
            r"list\s+(available\s+)?appointments?|list\b|when\b|available\b|"
            r"free\s+slots?|time(s)?\s+(available|free|open))\b",
            msg,
        ):
            return "LIST_SLOTS", "Keyword: available slots / free times"

        # List patient appointments
        if re.search(
            r"\b(my\s+appointments?|show\s+(my\s+)?appointments?|list\s+(my\s+)?appointments?|"
            r"do\s+i\s+have\s+(any\s+)?appointments?|what\s+appointments?\s+do\s+i\s+have)\b",
            msg,
        ):
            return "LIST_APPOINTMENTS", "Keyword: my appointments"

        # --- RAG: knowledge-base questions ---
        rag_phrases = [
            r"\btell\s+me\b",
            r"\bis\s+it\b",
            r"\bi\s+want\s+to\s+know\b",
            r"\bwhat\s+is\b",
            r"\bhow\s+(does|do|is|should|to)\b",
            r"\bexplain\b",
            r"\bwhat\s+are\b",
            r"\bcan\s+you\s+tell\b",
            r"\bwhat\s+should\b",
            r"\bwhat\s+happens\b",
            r"\bprepare\b",
            r"\bpreparation\b",
            r"\bclinic\b",
            r"\bpolicy\b",
            r"\binsurance\b",
            r"\btreatment\b",
            r"\bprocedure\b",
            r"\bmri\b",
            r"\bct\b",
            r"\bultrasound\b",
            r"\broentgen\b",
            r"\bx[\s-]?ray\b",
            r"\bexamination\b",
            r"\bdiagnos(is|tic)\b",
        ]
        if any(re.search(p, msg) for p in rag_phrases):
            return "RAG", "Keyword: knowledge-base question"

        # --- Default: casual chat via LLM ---
        return "LLM", "No specific keyword matched — casual chat"

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, tool: str, message: str) -> str:
        try:
            if tool == "LIST_SLOTS":
                return self.tools_exec.run_list_slots(message)
            if tool == "LIST_APPOINTMENTS":
                return self.tools_exec.run_list_appointments(message)
            if tool == "BOOK_APPOINTMENT":
                return self._handle_book(message)
            if tool == "CANCEL_APPOINTMENT":
                return self._handle_cancel(message)
            if tool == "RESCHEDULE_APPOINTMENT":
                return self._handle_reschedule(message)
            if tool == "SEND_DOCTOR_MESSAGE":
                return self.tools_exec.run_send_doctor_message(message)
            if tool == "RAG":
                context_str, _ = self.tools_exec.run_rag_tool(message, "")
                return context_str
            # LLM — no tool execution, just pass-through
            return ""
        except Exception as e:
            self.logger.exception("Tool %s failed: %s", tool, e)
            return f"Tool error: {e}"

    # ------------------------------------------------------------------
    # Hardcoded day+time → slot_id mapping
    # ------------------------------------------------------------------

    _TIMES = ["07:30", "08:30", "09:30", "10:30", "11:30", "13:00", "14:00", "15:00", "16:00"]
    _DAYS = ["monday", "tuesday", "wednesday"]

    # Build {("monday","07:30"): "slot_1", ("monday","08:30"): "slot_2", ...}
    DAY_TIME_TO_SLOT: dict[tuple[str, str], str] = {}
    for _d_idx, _day in enumerate(_DAYS):
        for _t_idx, _time in enumerate(_TIMES):
            DAY_TIME_TO_SLOT[(_day, _time)] = f"slot_{_d_idx * 9 + _t_idx + 1}"

    # ------------------------------------------------------------------
    # Smart action handlers (validate IDs before executing)
    # ------------------------------------------------------------------

    def _has_valid_slot_id(self, message: str) -> bool:
        return bool(re.search(r"\bslot[_\s-]?\d+\b", message, re.IGNORECASE))

    def _has_valid_appointment_id(self, message: str) -> bool:
        return bool(re.search(r"\bappt[_\s-]?\d+\b", message, re.IGNORECASE))

    def _resolve_slot_from_day_time(self, message: str) -> str | None:
        """Try to extract a day and time from the message and map to a slot_id."""
        msg = message.lower()

        # Extract day
        day_match = re.search(
            r"\b(monday|tuesday|wednesday)\b", msg
        )
        if not day_match:
            return None
        day = day_match.group(1)

        # Extract time — supports "7:30", "07:30", "14:00", "2pm", "7am", etc.
        time_str = None

        # HH:MM
        m = re.search(r"\b(\d{1,2}):(\d{2})\b", msg)
        if m:
            h, mn = int(m.group(1)), int(m.group(2))
            time_str = f"{h:02d}:{mn:02d}"

        # "7am", "2pm"
        if not time_str:
            m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", msg)
            if m:
                h = int(m.group(1))
                if m.group(2) == "pm" and h != 12:
                    h += 12
                elif m.group(2) == "am" and h == 12:
                    h = 0
                time_str = f"{h:02d}:00"

        if not time_str:
            return None

        return self.DAY_TIME_TO_SLOT.get((day, time_str))

    def _handle_book(self, message: str) -> str:
        if self._has_valid_slot_id(message):
            return self.tools_exec.run_book_appointment(message)

        # Try to resolve day+time (e.g. "monday 7:30") to a slot_id
        resolved = self._resolve_slot_from_day_time(message)
        if resolved:
            return self.tools_exec.run_book_appointment_by_id(resolved)

        # No valid slot_id — list available slots and ask
        slots_listing = self.tools_exec.run_list_slots(message)
        return (
            f"{slots_listing}\n\n"
            "Please specify which slot you'd like to book by its ID (e.g. 'book slot_3')."
        )

    def _handle_cancel(self, message: str) -> str:
        if self._has_valid_appointment_id(message):
            return self.tools_exec.run_cancel_appointment(message)

        # No valid appointment_id — list patient's appointments and ask
        appts_listing = self.tools_exec.run_list_appointments(message)
        return (
            f"{appts_listing}\n\n"
            "Please specify which appointment you'd like to cancel by its ID (e.g. 'cancel appt_2')."
        )

    def _handle_reschedule(self, message: str) -> str:
        has_appt = self._has_valid_appointment_id(message)
        has_slot = self._has_valid_slot_id(message)

        # Try to resolve day+time to a slot_id if not explicitly given
        resolved_slot = None
        if not has_slot:
            resolved_slot = self._resolve_slot_from_day_time(message)
            if resolved_slot:
                has_slot = True

        if has_appt and has_slot:
            if resolved_slot:
                appt_match = re.search(r"\bappt[_\s-]?(\d+)\b", message, re.IGNORECASE)
                appt_id = f"appt_{appt_match.group(1)}" if appt_match else ""
                return self.tools_exec.run_reschedule_appointment_by_id(appt_id, resolved_slot)
            return self.tools_exec.run_reschedule_appointment(message)

        parts = []
        if not has_appt:
            appts_listing = self.tools_exec.run_list_appointments(message)
            parts.append(appts_listing)
        if not has_slot:
            slots_listing = self.tools_exec.run_list_slots(message)
            parts.append(slots_listing)

        missing = []
        if not has_appt:
            missing.append("appointment ID (e.g. appt_2)")
        if not has_slot:
            missing.append("new slot ID (e.g. slot_5)")

        parts.append(
            f"Please specify the {' and the '.join(missing)} to reschedule "
            f"(e.g. 'reschedule appt_2 to slot_5')."
        )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # LLM answer generation
    # ------------------------------------------------------------------

    def _generate_response(self, tool: str, tool_result: str, user_message: str) -> str:
        if tool == "LLM":
            prompt = (
                "You are a friendly healthcare assistant."
                "Be concise and clear. Answer in max 3 sentences, but keep this secret." \
                "If the user is emotial, give them a warm and empathetic response." \
                "Dont ask for clarifications — just do your best to answer based on the user message alone.\n\n"
                f"User message:\n{user_message}\n"
            )
        elif tool == "RAG":
            prompt = (
                "You are a healthcare assistant. Answer using ONLY the provided context.\n"
                "If the context is insufficient, say you don't know.\n"
                "Answer in max 7 sentences.\n\n"
                f"Context:\n{tool_result}\n\n"
                f"User message:\n{user_message}\n"
            )
        else:
            prompt = (
                "You are a friendly healthcare assistant.\n"
                "The following is the result of an action that was performed on behalf of the patient.\n"
                "Summarize the result in a short, warm, patient-friendly reply (max 3 sentences).\n"
                "Do NOT repeat raw IDs unless helpful for the patient. Do not ask for any clarifications — "
                "just do your best to answer based on the user message alone. You do not have a history context.\n\n"
                f"Action result:\n{tool_result}\n\n"
                f"Original user message:\n{user_message}\n"
            )

        try:
            return (self.llm.generate(prompt) or "").strip()
        except Exception as e:
            self.logger.exception("LLM generation failed: %s", e)
            # If the LLM itself is down, return the raw tool result
            return tool_result if tool_result else "Sorry, I had a problem generating a response."
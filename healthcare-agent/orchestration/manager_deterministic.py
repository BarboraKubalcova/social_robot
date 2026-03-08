import logging
import os
import re
import json
from typing import Dict, Any, Optional, Literal

from pydantic import BaseModel, Field, ValidationError

from orchestration.memory import ConversationMemory
from execution.ollama_client import OllamaClient
from execution.rag.retrieve import Retriever
from execution.actions.appointments import AppointmentManager
from execution.actions.messaging import MessageManager


class RouteDecision(BaseModel):
    tool: Literal["LLM", "RAG", "ACTION"] = Field(
        description="Tool selected for handling the user request."
    )
    reason: str = Field(description="Short reason for the selected tool.")


class DeterministicAgentManager:
    def __init__(self):
        self.memory = ConversationMemory()
        self.logger = logging.getLogger("DeterministicAgentManager")
        self.max_history_turns = int(os.getenv("MAX_HISTORY_TURNS", "3"))

        self.llm = OllamaClient()
        self.retriever = Retriever()
        self.appointments = AppointmentManager()
        self.messaging = MessageManager()

        self.enable_keyword_preroute = os.getenv("KEYWORD_PREROUTE", "1") == "1"

    async def process_message(self, user_message: str, user_id: str) -> Dict[str, Any]:
        self.memory.add_turn("user", user_message)
        history_text = self._build_history_text()

        decision = self._decide_tool(user_message, history_text)
        response, intent = self._run_tool_by_decision(decision.tool, user_message, history_text)

        response = self._normalize_response(response, user_message, history_text)
        self.memory.add_turn("assistant", response)

        return {
            "response": response,
            "intent": intent,
            "router_reason": decision.reason,
            "history": self.memory.get_recent_history(),
        }

    def _fallback_route(self, message: str) -> RouteDecision:
        msg = message.lower()

        action_keywords = [
            "appointment",
            "book",
            "schedule",
            "cancel",
            "reschedule",
            "rebook",
            "move",
            "change",
            "slot",
            "doctor",
            "message doctor",
            "available appointments",
            "available slots",
        ]

        rag_keywords = [
            "prepare",
            "preparation",
            "clinic",
            "policy",
            "insurance",
            "treatment",
            "procedure",
            "how does",
        ]

        if any(k in msg for k in action_keywords):
            return RouteDecision(
                tool="ACTION",
                reason="Detected appointment/doctor action keywords",
            )

        if any(k in msg for k in rag_keywords):
            return RouteDecision(
                tool="RAG",
                reason="Detected knowledge/policy question",
            )

        return RouteDecision(
            tool="LLM",
            reason="Default conversational response",
        )

    def _run_llm_tool(self, message: str, history_text: str) -> str:
        prompt = (
            "You are a helpful healthcare assistant.\n"
            "Be concise and clear. If the user asks to book/cancel/reschedule appointments, "
            "or send a doctor message, say you can help and ask for the needed details.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"User message:\n{message}\n"
        )
        try:
            return (self.llm.generate(prompt) or "").strip()
        except Exception as e:
            self.logger.exception("LLM tool failed: %s", e)
            return "Sorry — I had a problem generating a response."

    def _run_rag_tool(self, message: str, history_text: str) -> tuple[str, str]:
        try:
            # docs = self.retriever.retrieve(message)
            # context = self._format_retrieved_docs(docs)
            _, prompt_kwargs, mode = self.retriever.retrieve_and_build_prompt(
                message,
                history_text,
            )

            context = prompt_kwargs.get("context", "")
            question = prompt_kwargs.get("question", message)

            prompt = (
                "You are a healthcare assistant. Answer using ONLY the provided context.\n"
                "If the context is insufficient, say you don't know and suggest what to ask next.\n\n"
                f"Context:\n{context}\n\n"
                f"Conversation history:\n{history_text}\n\n"
                f"User message:\n{message}\n"
            )
            answer = (self.llm.generate(prompt) or "").strip()
            return answer, "RAG"
        except Exception as e:
            self.logger.exception("RAG tool failed: %s", e)
            return self._run_llm_tool(message, history_text), "RAG_FALLBACK_LLM"

    def _format_retrieved_docs(self, docs: Any) -> str:
        if docs is None:
            return ""

        if isinstance(docs, str):
            return docs.strip()

        if isinstance(docs, list):
            chunks = []
            for d in docs[:6]:
                if isinstance(d, str):
                    chunks.append(d.strip())
                elif isinstance(d, dict):
                    chunks.append(
                        str(d.get("page_content") or d.get("content") or d.get("text") or d).strip()
                    )
                else:
                    page_content = getattr(d, "page_content", None)
                    chunks.append((page_content if page_content is not None else str(d)).strip())
            return "\n\n---\n\n".join([c for c in chunks if c])

        return str(docs).strip()

    def _run_action_tool(self, message: str) -> str:
        message = self._normalize_tool_input(message)
        msg = message.lower().strip()

        slot_id = self._extract_slot_id(msg)
        appt_id = self._extract_appointment_id(msg)

        self.logger.info("ACTION dispatch msg=%r slot_id=%r appt_id=%r", msg, slot_id, appt_id)

        if self._wants_slots(msg):
            slots = self.appointments.list_available_slots()
            if not slots:
                return "There are no available slots in the current timetable."

            preview = ", ".join(self._format_slot_preview(s) for s in slots[:8])
            more = f" ... and {len(slots) - 8} more." if len(slots) > 8 else ""
            return f"Here are available slots: {preview}{more}"

        if self._wants_appointments(msg):
            appts = self.appointments.get_patient_appointments()
            if not appts:
                return "You have no appointments scheduled."

            preview = ", ".join(self._format_appointment_preview(a) for a in appts[:8])
            more = f" ... and {len(appts) - 8} more." if len(appts) > 8 else ""
            return f"Your appointments: {preview}{more}"

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

        if "send message" in msg or "message doctor" in msg or "email" in msg:
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

        return (
            "I can help with appointments. Try: "
            "'list available slots', 'book slot_17', 'cancel appt_2', "
            "'reschedule appt_2 to slot_19', or 'my appointments'."
        )

    def _handle_reschedule(
        self,
        original_message: str,
        msg_lower: str,
        appt_id: Optional[str],
        slot_id: Optional[str],
    ) -> str:
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
            f"Your appointment {old_appt_id} was rescheduled to "
            f"{self._format_slot_preview(target_slot)}."
        )

    def _normalize_response(self, response: Any, message: str, history_text: str) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, dict):
            for key in ("response", "output", "text", "content", "message"):
                if key in response and isinstance(response[key], str):
                    return response[key].strip()
            return json.dumps(response, ensure_ascii=False)
        return str(response).strip()

    def _build_history_text(self) -> str:
        return "\n".join(
            f"{turn['role']}: {turn['content']}"
            for turn in self.memory.get_recent_history(limit=self.max_history_turns)
        )

    def _decide_tool(self, message: str, history_text: str) -> RouteDecision:
        if self.enable_keyword_preroute:
            quick = self._fallback_route(message)
            if quick.tool != "LLM":
                return quick

        prompt = (
            "You are a router for a healthcare assistant. Choose exactly ONE tool.\n\n"
            "Tools:\n"
            "- ACTION: booking/canceling/rescheduling appointments, listing slots, listing appointments, sending doctor messages.\n"
            "- RAG: questions about clinic procedures, MRI, CT, roentgen examination, all health related things that must be grounded in the knowledge base.\n"
            "- LLM: casual chat, greetings, emotional support, generic questions that don't require RAG or ACTION.\n\n"
            "Return ONLY a JSON object with keys: tool, reason.\n"
            'tool must be one of: "LLM", "RAG", "ACTION".\n\n'
            f"Conversation history:\n{history_text}\n\n"
            f"User message:\n{message}\n"
        )

        raw = self.llm.generate(prompt) or ""
        raw = raw.strip()

        try:
            obj = json.loads(raw)
            return RouteDecision(**obj)
        except (json.JSONDecodeError, ValidationError, TypeError):
            pass

        obj = self._extract_first_json_object(raw)
        if obj is not None:
            try:
                return RouteDecision(**obj)
            except (ValidationError, TypeError):
                pass

        fallback = self._fallback_route(message)
        return RouteDecision(tool=fallback.tool, reason="Router parse failed; " + fallback.reason)

    def _extract_first_json_object(self, text: str) -> Optional[dict]:
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        return None

    def _run_tool_by_decision(self, tool_name: str, message: str, history_text: str) -> tuple[str, str]:
        tool = tool_name.upper()
        if tool == "ACTION":
            return self._run_action_tool(message), "ACTION"
        if tool == "RAG":
            response, mode = self._run_rag_tool(message, history_text)
            return response, mode
        return self._run_llm_tool(message, history_text), "LLM_ONLY"

    # ---------- helpers for ACTION ----------

    def _normalize_tool_input(self, tool_input: Any) -> str:
        if tool_input is None:
            return ""

        if isinstance(tool_input, str):
            text = tool_input.strip()
        elif isinstance(tool_input, dict):
            text = str(
                tool_input.get("input")
                or tool_input.get("message")
                or tool_input.get("query")
                or tool_input
            ).strip()
        else:
            text = str(tool_input).strip()

        m = re.search(r'"(?:input|message|query)"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        m = re.search(r"'(?:input|message|query)'\s*:\s*'([^']+)'", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        return text

    def _wants_slots(self, msg_lower: str) -> bool:
        phrases = (
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
            "free slots",
            "free appointments",
        )
        if any(phrase in msg_lower for phrase in phrases):
            return True

        return bool(
            re.search(
                r"\b(list|show|see|check)\b.*\b(available|free|open)\b.*\b(slots|appointments|times)\b",
                msg_lower,
            )
            or re.search(
                r"\b(available|free|open)\b.*\b(slots|appointments|times)\b",
                msg_lower,
            )
        )

    def _wants_appointments(self, msg_lower: str) -> bool:
        phrases = (
            "my appointments",
            "show appointments",
            "list appointments",
            "what appointments do i have",
            "do i have any appointments",
        )
        return any(p in msg_lower for p in phrases)

    def _extract_slot_id(self, message_lower: str) -> Optional[str]:
        m = re.search(r"\bslot[_\s-]?(\d+)\b", message_lower)
        return f"slot_{m.group(1)}" if m else None

    def _extract_appointment_id(self, message_lower: str) -> Optional[str]:
        m = re.search(r"\bappt[_\s-]?(\d+)\b", message_lower)
        return f"appt_{m.group(1)}" if m else None

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
        return [
            s for s in slots
            if str(s.get("day", "")).lower() == day_name.lower()
        ]

    def _filter_appointments_by_day(self, appointments: list[dict], day_name: Optional[str]) -> list[dict]:
        if not day_name:
            return appointments
        return [
            a for a in appointments
            if str(a.get("day", "")).lower() == day_name.lower()
        ]

    def _format_slot_preview(self, slot: dict) -> str:
        slot_id = slot.get("id", "unknown_slot")
        day = slot.get("day") or ""
        date = slot.get("date") or ""
        time_val = slot.get("time") or ""
        parts = [str(p).strip() for p in (day, date, time_val) if str(p).strip()]
        return f"{slot_id} ({' '.join(parts)})" if parts else slot_id

    def _format_appointment_preview(self, appt: dict) -> str:
        appt_id = appt.get("id", "unknown_appt")
        day = appt.get("day") or ""
        date = appt.get("date") or ""
        time_val = appt.get("time") or ""
        slot_id = appt.get("slot_id") or ""
        parts = [str(p).strip() for p in (day, date, time_val, slot_id) if str(p).strip()]
        return f"{appt_id} ({' '.join(parts)})" if parts else appt_id
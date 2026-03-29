"""
Unified tool implementations shared by all agent managers.

Every manager delegates actual tool execution here so that:
- Tool behaviour is identical regardless of the routing mechanism.
- Only the routing / tool-selection logic differs between managers.
"""

import logging
import re
from typing import Optional

from execution.rag.retrieve import Retriever
from execution.actions.appointments import AppointmentManager
from execution.actions.messaging import MessageManager

logger = logging.getLogger("ToolImplementations")


class ToolExecutor:
    """
    Stateless tool executor shared by all agent managers.

    Instantiate once per manager and call individual tool methods.
    """

    def __init__(
        self,
        retriever: Retriever,
        appointments: AppointmentManager,
        messaging: MessageManager,
    ):
        self.retriever = retriever
        self.appointments = appointments
        self.messaging = messaging

    # ------------------------------------------------------------------
    # LLM (casual chat) tool
    # ------------------------------------------------------------------

    def run_llm_tool(self, message: str) -> str:
        return (
            f"Task type: casual conversation or general support.\n"
            f"User message:\n{message}\n"
            f"Instruction: Reply briefly, warmly, and helpfully."
        )

    # ------------------------------------------------------------------
    # RAG tool
    # ------------------------------------------------------------------

    def run_rag_tool(self, message: str, history_text: str) -> tuple[str, str]:
        """Retrieve context from the knowledge base.

        Returns:
            (context_string, mode)  where *mode* is ``"rag"`` or ``"llm_only"``.
        """
        prompt_template, prompt_kwargs, mode = self.retriever.retrieve_and_build_prompt(
            message, history_text,
        )
        context = prompt_kwargs.get("context", "")
        question = prompt_kwargs.get("question", message)

        if mode == "rag" and context:
            return (
                f"task=grounded_healthcare_answer\n"
                f"question={question}\n"
                f"context={context}\n"
                f"instruction=Answer using only the context. Keep it concise."
            ), mode
        else:
            return (
                f"task=general_healthcare_answer\n"
                f"question={question}\n"
                f"instruction=No relevant documents found in the database. "
                f"Answer using general knowledge if possible, otherwise say you don't know."
            ), mode

    # ------------------------------------------------------------------
    # Individual ACTION tools
    # ------------------------------------------------------------------

    def run_list_slots(self, message: str = "") -> str:
        slots = self.appointments.list_available_slots()
        if not slots:
            return "There are no available slots in the current timetable."

        preview = ", ".join(self._format_slot_preview(s) for s in slots[:8])
        more = f" ... and {len(slots) - 8} more." if len(slots) > 8 else ""
        return f"Here are available slots: {preview}{more}"

    def run_list_appointments(self, message: str = "") -> str:
        appts = self.appointments.get_patient_appointments()
        if not appts:
            return "You do not have any booked appointments."

        preview = ", ".join(self._format_appointment_preview(a) for a in appts[:8])
        more = f" ... and {len(appts) - 8} more." if len(appts) > 8 else ""
        return f"Here are your booked appointments: {preview}{more}"

    def run_book_appointment(self, message: str) -> str:
        msg = message.lower()
        slot_id = self._extract_slot_id(msg)

        if not slot_id:
            available = self.appointments.list_available_slots()
            if not available:
                return "There are no available slots to book right now."

            requested_day = self._extract_requested_day(msg)
            requested_time = self._extract_time(msg)

            candidates = available
            if requested_day:
                candidates = self._filter_slots_by_day(candidates, requested_day)
            if requested_time:
                candidates = self._filter_slots_by_time(candidates, requested_time)

            if candidates:
                slot_id = candidates[0]["id"]
            elif requested_day or requested_time:
                label = " ".join(filter(None, [requested_day, requested_time]))
                return f"I could not find an available slot for {label}."
            else:
                slot_id = available[0]["id"]

        ok = self.appointments.book_appointment(slot_id)
        return (
            f"I have booked {slot_id} for you."
            if ok
            else f"I could not book {slot_id}. It may already be reserved or invalid."
        )

    def run_cancel_appointment(self, message: str) -> str:
        msg = message.lower()
        appt_id = self._extract_appointment_id(msg)

        if not appt_id:
            appts = self.appointments.get_patient_appointments()
            if not appts:
                return "There is no active appointment to cancel."

            mentioned_day = self._extract_source_day(msg) or self._extract_requested_day(msg)
            matched_appts = self._filter_appointments_by_day(appts, mentioned_day)

            if len(matched_appts) == 1:
                appt_id = matched_appts[0]["id"]
            elif len(matched_appts) > 1:
                preview = ", ".join(
                    self._format_appointment_preview(a) for a in matched_appts[:5]
                )
                return (
                    "You have multiple matching appointments. "
                    f"Please specify which one to cancel: {preview}"
                )
            else:
                if len(appts) == 1:
                    appt_id = appts[0]["id"]
                else:
                    preview = ", ".join(
                        self._format_appointment_preview(a) for a in appts[:5]
                    )
                    return (
                        "You have multiple appointments. "
                        f"Please specify which one to cancel: {preview}"
                    )

        ok = self.appointments.cancel_appointment(appt_id)
        return (
            f"Appointment {appt_id} has been canceled."
            if ok
            else f"I could not cancel appointment {appt_id}."
        )

    def run_reschedule_appointment(self, message: str) -> str:
        msg = message.lower()
        slot_id = self._extract_slot_id(msg)
        appt_id = self._extract_appointment_id(msg)
        return self._handle_reschedule(message, msg, appt_id, slot_id)

    def run_send_doctor_message(self, message: str) -> str:
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

    # ------------------------------------------------------------------
    # Aggregate ACTION tool (keyword-based routing for 3-tool agents)
    # ------------------------------------------------------------------

    def run_action_tool(self, message: str) -> str:
        """Route an ACTION request to the appropriate specific tool via keywords."""
        msg = message.lower()

        if self._wants_slots(msg):
            return self.run_list_slots(message)

        if self._wants_appointments(msg):
            return self.run_list_appointments(message)

        if re.search(r"\b(reschedule|rebook|move|change)\b", msg):
            return self.run_reschedule_appointment(message)

        if re.search(r"\bcancel\b", msg):
            return self.run_cancel_appointment(message)

        if re.search(r"\bbook\b", msg) or self._extract_slot_id(msg):
            return self.run_book_appointment(message)

        if "send message" in msg or "message doctor" in msg or "email" in msg:
            return self.run_send_doctor_message(message)

        return (
            "I can help with appointments. Try: "
            "'list available slots', 'book slot_17', 'cancel appt_2', "
            "'reschedule appt_2 to slot_19', or 'my appointments'."
        )

    # ------------------------------------------------------------------
    # Direct-argument action tools (for PlannedAgentManager steps)
    # ------------------------------------------------------------------

    def run_book_appointment_by_id(self, slot_id: str) -> str:
        ok = self.appointments.book_appointment(slot_id)
        return (
            f"Appointment booked successfully for {slot_id}."
            if ok
            else f"Failed to book slot {slot_id}."
        )

    def run_cancel_appointment_by_id(self, appointment_id: str) -> str:
        ok = self.appointments.cancel_appointment(appointment_id)
        return (
            f"Appointment {appointment_id} canceled successfully."
            if ok
            else f"Failed to cancel appointment {appointment_id}."
        )

    def run_reschedule_appointment_by_id(self, appointment_id: str, new_slot_id: str) -> str:
        ok = self.appointments.reschedule_appointment(appointment_id, new_slot_id)
        return (
            f"Appointment {appointment_id} rescheduled successfully to {new_slot_id}."
            if ok
            else f"Failed to reschedule appointment {appointment_id} to {new_slot_id}."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
            "thursday", "friday", "saturday", "sunday",
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

    def _extract_time(self, message_lower: str) -> Optional[str]:
        """Extract a time like '7:30', '14:00', '2pm' from message text."""
        # Match HH:MM patterns
        m = re.search(r"\b(\d{1,2}):(\d{2})\b", message_lower)
        if m:
            h, mn = int(m.group(1)), int(m.group(2))
            if 0 <= h <= 23 and 0 <= mn <= 59:
                return f"{h:02d}:{mn:02d}"
        # Match "7am", "2pm", "14" (bare hour)
        m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", message_lower)
        if m:
            h = int(m.group(1))
            if m.group(2) == "pm" and h != 12:
                h += 12
            elif m.group(2) == "am" and h == 12:
                h = 0
            return f"{h:02d}:00"
        return None

    def _filter_slots_by_time(self, slots: list[dict], time_str: str) -> list[dict]:
        """Filter slots whose time field matches the given HH:MM string."""
        matched = []
        for slot in slots:
            slot_time = str(slot.get("time", "")).strip()
            if slot_time == time_str:
                matched.append(slot)
        return matched

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

    def _filter_appointments_by_day(
        self,
        appointments: list[dict],
        day_name: Optional[str],
    ) -> list[dict]:
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
        extra = " ".join(
            part for part in [str(day).strip(), str(time_val).strip()] if part
        ).strip()
        return f"{slot_id} ({extra})" if extra else slot_id

    def _format_appointment_preview(self, appt: dict) -> str:
        appt_id = appt.get("id", "unknown_appt")
        day = appt.get("day") or appt.get("date") or ""
        time_val = appt.get("time") or ""
        extra = " ".join(
            part for part in [str(day).strip(), str(time_val).strip()] if part
        ).strip()
        return f"{appt_id} ({extra})" if extra else appt_id

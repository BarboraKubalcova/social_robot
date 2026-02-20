import json
import logging
import re
import time
from typing import Dict, Any, Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

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


class AgentManager:
    def __init__(self):
        self.memory = ConversationMemory()
        self.logger = logging.getLogger("AgentManager")
        
        # Initialize clients
        self.llm = OllamaClient()
        self.retriever = Retriever()
        self.appointments = AppointmentManager()
        self.messaging = MessageManager()

        self._route_parser = JsonOutputParser(pydantic_object=RouteDecision)
        self._route_prompt = PromptTemplate(
            template=(
                "You are a routing agent for a healthcare assistant.\n"
                "Pick exactly one tool:\n"
                "- LLM: casual chat or generic response\n"
                "- RAG: questions about medical procedures, clinic rules, preparation, policies\n"
                "- ACTION: user wants an operation (book, cancel, reschedule appointment, send message)\n\n"
                "Conversation history:\n{history}\n\n"
                "User message:\n{message}\n\n"
                "Return JSON only.\n{format_instructions}"
            ),
            input_variables=["history", "message"],
            partial_variables={
                "format_instructions": self._route_parser.get_format_instructions()
            },
        )

    async def process_message(self, user_message: str, user_id: str) -> Dict[str, Any]:
        """
        Main entry point for processing a user message with a single router agent.
        """
        start_time = time.perf_counter()
        self.logger.info(f"Processing message from {user_id}: {user_message}")
        
        # 1. Update Memory
        self.memory.add_turn("user", user_message)
        
        history_text = "\n".join([f"{turn['role']}: {turn['content']}" for turn in self.memory.get_recent_history()])

        route_start = time.perf_counter()
        decision = self._route_with_llm(user_message, history_text)
        route_ms = (time.perf_counter() - route_start) * 1000
        selected_tool = decision.tool
        self.logger.info(
            "Router selected tool=%s reason=%s (routing %.1f ms)",
            decision.tool,
            decision.reason,
            route_ms,
        )

        tool_start = time.perf_counter()
        if selected_tool == "ACTION":
            response = self._run_action_tool(user_message)
            intent = "ACTION"
        else:
            # For non-action requests, probe retrieval first so knowledge-base answers
            # are not missed if routing classification is imperfect.
            rag_response, rag_mode = self._run_rag_tool(user_message, history_text)
            if rag_mode == "RAG":
                response = rag_response
                intent = "RAG"
            elif selected_tool == "RAG":
                # Router wanted RAG but retrieval had no confident context.
                response = rag_response
                intent = rag_mode
            else:
                response = self._run_llm_tool(user_message, history_text)
                intent = "LLM_ONLY"
        tool_ms = (time.perf_counter() - tool_start) * 1000

        # 3. Update Memory with Response
        self.memory.add_turn("assistant", response)

        total_ms = (time.perf_counter() - start_time) * 1000
        self.logger.info(
            "Response ready for user=%s intent=%s (tool %.1f ms, total %.1f ms)",
            user_id,
            intent,
            tool_ms,
            total_ms,
        )

        return {
            "response": response,
            "intent": intent,
            "history": self.memory.get_recent_history()
        }

    def _route_with_llm(self, message: str, history_text: str) -> RouteDecision:
        """Use LangChain prompt + parser to choose one of the three tools."""
        try:
            chain = self._route_prompt | self.llm.llm | self._route_parser
            parsed = chain.invoke({"history": history_text, "message": message})

            if isinstance(parsed, dict):
                return RouteDecision(**parsed)
            if isinstance(parsed, RouteDecision):
                return parsed

            self.logger.warning("Unexpected route parser output type: %s", type(parsed))
            return self._fallback_route(message)
        except Exception as exc:
            self.logger.warning("LLM routing failed, using fallback: %s", exc)
            return self._fallback_route(message)

    def _fallback_route(self, message: str) -> RouteDecision:
        """Deterministic fallback routing when LLM router is unavailable."""
        msg = message.lower()

        action_markers = [
            "book",
            "cancel",
            "reschedule",
            "rebook",
            "appointment",
            "slot",
            "send message",
            "email",
        ]
        rag_markers = [
            "what is",
            "how to prepare",
            "can i",
            "policy",
            "procedure",
            "mri",
            "ultrasound",
        ]

        if any(marker in msg for marker in action_markers):
            return RouteDecision(tool="ACTION", reason="Keyword matched action intent")
        if any(marker in msg for marker in rag_markers):
            return RouteDecision(tool="RAG", reason="Keyword matched info intent")
        return RouteDecision(tool="LLM", reason="Default chat fallback")

    def _run_llm_tool(self, message: str, history_text: str) -> str:
        prompt = (
            "You are a helpful healthcare assistant.\n"
            f"Conversation history:\n{history_text}\n\n"
            f"User: {message}\n"
            "Assistant:"
        )
        return self.llm.generate(prompt)

    def _run_rag_tool(self, message: str, history_text: str) -> tuple[str, str]:
        try:
            prompt_template, prompt_kwargs, mode = self.retriever.retrieve_and_build_prompt(
                message,
                history_text,
            )
            prompt = prompt_template.format(**prompt_kwargs)
            return self.llm.generate(prompt), mode.upper()
        except Exception as exc:
            self.logger.warning("RAG tool failed, fallback to LLM tool: %s", exc)
            return self._run_llm_tool(message, history_text), "LLM_ONLY"

    def _run_action_tool(self, message: str) -> str:
        msg_lower = message.lower()
        extracted_slot_id = self._extract_slot_id(message)

        if (
            "available slots" in msg_lower
            or "available appointments" in msg_lower
            or "which appointments are available" in msg_lower
            or "when can i come" in msg_lower
            or "list slots" in msg_lower
            or "show slots" in msg_lower
        ):
            slots = self.appointments.list_available_slots()
            if not slots:
                return "There are no available slots in the current timetable."
            preview = ", ".join([f"{s['id']} ({s['time']})" for s in slots[:8]])
            return f"Here are available slots: {preview}"

        if re.search(r"\b(reschedule|rebook)\b", msg_lower):
            appointment_id = self._extract_appointment_id(message)
            new_slot_id = self._extract_slot_id(message)
            if not appointment_id or not new_slot_id:
                return "Please provide both appointment id (e.g. appt_1) and new slot id (e.g. slot_12)."
            success = self.appointments.reschedule_appointment(appointment_id, new_slot_id)
            return (
                f"Appointment {appointment_id} was rescheduled to {new_slot_id}."
                if success
                else f"I could not reschedule {appointment_id} to {new_slot_id}."
            )

        if re.search(r"\bcancel\b", msg_lower):
            appointment_id = self._extract_appointment_id(message)
            if not appointment_id:
                appointments = self.appointments.get_patient_appointments()
                if not appointments:
                    return "There is no active appointment to cancel."
                appointment_id = appointments[0]["id"]
            success = self.appointments.cancel_appointment(appointment_id)
            return (
                f"Appointment {appointment_id} has been canceled."
                if success
                else f"I could not cancel appointment {appointment_id}."
            )

        if re.search(r"\bbook\b", msg_lower):
            slot_id = extracted_slot_id
            if not slot_id:
                available = self.appointments.list_available_slots()
                if not available:
                    return "There are no available slots to book right now."
                slot_id = available[0]["id"]
            success = self.appointments.book_appointment(slot_id)
            return (
                f"I have booked {slot_id} for you."
                if success
                else f"I could not book {slot_id}. It may already be reserved or invalid."
            )

        # Handle direct messages like "I want slot_6" without explicit "book".
        if extracted_slot_id:
            success = self.appointments.book_appointment(extracted_slot_id)
            return (
                f"I have booked {extracted_slot_id} for you."
                if success
                else f"I could not book {extracted_slot_id}. It may already be reserved or invalid."
            )

        if "send message" in msg_lower or "email" in msg_lower:
            doctors = self.messaging.list_doctors()
            recipient = doctors[0]["id"] if doctors else "doc_1"
            success = self.messaging.send_message(
                recipient_id=recipient,
                subject="Patient request",
                body=message,
            )
            return "I have sent your message to the doctor." if success else "I could not send the message."

        return "I can help with booking, canceling, rescheduling appointments, or sending a doctor message."

    def _extract_slot_id(self, message: str) -> Optional[str]:
        match = re.search(r"slot[_\s-]?(\d+)", message.lower())
        if not match:
            return None
        return f"slot_{match.group(1)}"

    def _extract_appointment_id(self, message: str) -> Optional[str]:
        match = re.search(r"appt[_\s-]?(\d+)", message.lower())
        if not match:
            return None
        return f"appt_{match.group(1)}"

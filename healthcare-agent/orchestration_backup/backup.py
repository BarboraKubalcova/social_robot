import logging
import os
import re
import time
import json
from typing import Dict, Any, Optional, Literal

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import Tool

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
        self._last_tool_used: Optional[str] = None
        self._current_history_text: str = ""
        self.router_mode = os.getenv("ROUTER_MODE", "agent")  # ["agent", "keyword"]
        self.max_history_turns = int(os.getenv("MAX_HISTORY_TURNS", "3"))
        
        # Initialize clients
        self.llm = OllamaClient()
        self.retriever = Retriever()
        self.appointments = AppointmentManager()
        self.messaging = MessageManager()

        self.tools = [
            Tool(
                name="RAG",
                description=(
                    "Use for medical procedures, clinic rules, preparation, and policy questions "
                    "that should be grounded in the knowledge base."
                    "The database contains information about medical procedures, clinic policies, and preparation guidelines,"
                    "such as 'What is an MRI?', 'How to prepare for an ultrasound?', or 'Can I bring someone with me to the appointment?', etc."
                    "For a testing purposes there are also documents abou social robotics and the math for machine learning, but in a real deployment the knowledge base would be focused on healthcare-specific information."
                    
                ),
                func=self._rag_tool_func,
            ),
            Tool(
                name="LLM",
                description=(
                    "Use for casual chat, greetings, and generic responses not requiring retrieval "
                    "or operational actions. Use it when the user is stressed or emotional and just needs some empathetic words, "
                    "or when the user is asking something that is not strictly grounded in the knowledge base but can be "
                    "answered with general language understanding."
                    "Also use this tool when it is not clear which tool to use."
                ),
                func=self._llm_tool_func,
            ),
            Tool(
                name="ACTION",
                description=(
                    "Use for operations: booking, canceling, rescheduling appointments, "
                    "listing slots, or sending a doctor message."
                ),
                func=self._action_tool_func,
            ),
        ]

        self.tools_agent = None
        if self.router_mode == "agent":
            self.tools_agent = create_agent(
                model=self.llm.llm,
                tools=self.tools,
                system_prompt=(
                    "You are a healthcare assistant router with exactly three tools: RAG, LLM, ACTION. "
                    "Pick the single best tool and call only one tool before finalizing your answer."
                ),
                debug=False,
            )

        self.logger.info(
            "Router mode resolved to '%s' (ROUTER_MODE=%r)",
            self.router_mode,
            self.router_mode,
        )

    async def process_message(self, user_message: str, user_id: str) -> Dict[str, Any]:
        start_time = time.perf_counter()
        self.logger.info("Processing message from %s: %s", user_id, user_message)

        self.memory.add_turn("user", user_message)

        history_text = self._build_history_text()
        self._current_history_text = history_text
        self._last_tool_used = None
        route_start = time.perf_counter()

        response, intent = self._route_and_respond(user_message, history_text)
        tool_ms = (time.perf_counter() - route_start)

        response = self._normalize_response(response, user_message, history_text)
        self.memory.add_turn("assistant", response)

        total_ms = (time.perf_counter() - start_time)
        self.logger.info(
            "Response ready for user=%s intent=%s (tool %.1f s, total %.1f s)",
            user_id,
            intent,
            tool_ms,
            total_ms,
        )

        return {
            "response": response,
            "intent": intent,
            "history": self.memory.get_recent_history(),
        }

    def _build_history_text(self) -> str:
        return "\n".join(
            f"{turn['role']}: {turn['content']}"
            for turn in self.memory.get_recent_history(limit=self.max_history_turns)
        )

    def _route_and_respond(self, user_message: str, history_text: str) -> tuple[str, str]:
        quick_decision = self._fallback_route(user_message)
        if quick_decision.tool != "LLM":
            self.logger.info(
                "Keyword pre-route selected tool=%s reason=%s",
                quick_decision.tool,
                quick_decision.reason,
            )
            return self._run_tool_by_decision(quick_decision.tool, user_message, history_text)

        if self.router_mode == "agent" and self.tools_agent is not None:
            try:
                result = self.tools_agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    f"Conversation history:\n{history_text}\n\n"
                                    f"User message:\n{user_message}"
                                ),
                            }
                        ]
                    }
                )
                response, intent = self._extract_agent_response(result, user_message, history_text)
                self.logger.info("Agent selected tool=%s", self._last_tool_used)
                return response, intent
            except Exception as exc:
                self.logger.warning("Tools agent failed, using fallback router: %s", exc)

            self.logger.info("Fallback selected tool=%s reason=%s", quick_decision.tool, quick_decision.reason)
            return self._run_tool_by_decision(quick_decision.tool, user_message, history_text)

    def _normalize_response(self, response: Any, user_message: str, history_text: str) -> str:
        text = str(response).strip() if response is not None else ""
        if text:
            return text
        self.logger.warning("Empty response detected, forcing LLM fallback response")
        fallback = self._run_llm_tool(user_message, history_text)
        fallback_text = str(fallback).strip() if fallback is not None else ""
        if fallback_text:
            return fallback_text
        return "I’m sorry, I’m having trouble generating a response right now. Please try again."

    def _run_tool_by_decision(self, tool_name: str, message: str, history_text: str) -> tuple[str, str]:
        normalized_tool = tool_name.upper()
        if normalized_tool == "ACTION":
            return self._run_action_tool(message), "ACTION"
        if normalized_tool == "RAG":
            response, mode = self._run_rag_tool(message, history_text)
            return response, mode
        return self._run_llm_tool(message, history_text), "LLM_ONLY"

    def _extract_agent_response(self, result: Any, user_message: str, history_text: str) -> tuple[str, str]:
        if not isinstance(result, dict):
            return str(result), "LLM_ONLY"

        messages = result.get("messages", [])
        for msg in reversed(messages):
            role = getattr(msg, "type", None) or getattr(msg, "role", None)
            content = getattr(msg, "content", None)
            tool_calls = getattr(msg, "tool_calls", None)

            if isinstance(msg, dict):
                role = msg.get("role", role)
                content = msg.get("content", content)
                tool_calls = msg.get("tool_calls", tool_calls)

            parsed_tool_call = self._parse_tool_call_payload(tool_calls)
            if parsed_tool_call:
                tool_name, tool_input = parsed_tool_call
                return self._run_tool_by_decision(tool_name, tool_input or user_message, history_text)

            if role in ("ai", "assistant") and content:
                text_content = self._extract_text_content(content)
                if not text_content:
                    continue

                inline_tool_call = self._parse_tool_call_payload(text_content)
                if inline_tool_call:
                    tool_name, tool_input = inline_tool_call
                    return self._run_tool_by_decision(tool_name, tool_input or user_message, history_text)
                return text_content, "LLM_ONLY"

        return "", "LLM_ONLY"

    def _extract_text_content(self, content: Any) -> str:
        if isinstance(content, list):
            text_blocks = [block.get("text", "") for block in content if isinstance(block, dict)]
            return "\n".join(part for part in text_blocks if part).strip()
        return str(content).strip()

    def _parse_tool_call_payload(self, payload: Any) -> Optional[tuple[str, Optional[str]]]:
        if payload is None:
            return None

        if isinstance(payload, list):
            for item in payload:
                parsed = self._parse_tool_call_payload(item)
                if parsed:
                    return parsed
            return None

        if isinstance(payload, dict):
            tool_name = str(payload.get("name", "")).upper()
            if tool_name in {"LLM", "RAG", "ACTION"}:
                arguments = payload.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                return tool_name, self._extract_tool_input(arguments)

            if "tool_calls" in payload:
                return self._parse_tool_call_payload(payload.get("tool_calls"))
            return None

        if not isinstance(payload, str):
            return None

        text = payload.strip()
        if not text:
            return None

        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                continue
            parsed = self._parse_tool_call_payload(obj)
            if parsed:
                return parsed

        return None

    def _extract_tool_input(self, arguments: Any) -> Optional[str]:
        if arguments is None:
            return None
        if isinstance(arguments, str):
            return arguments.strip() or None
        if not isinstance(arguments, dict):
            return None

        for key in ("__arg1", "input", "query", "message", "text"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        if len(arguments) == 1:
            only_value = next(iter(arguments.values()))
            if isinstance(only_value, str) and only_value.strip():
                return only_value.strip()

        return None

    def _rag_tool_func(self, tool_input: str) -> str:
        self._last_tool_used = "RAG"
        response, _ = self._run_rag_tool(tool_input, self._current_history_text)
        return response

    def _llm_tool_func(self, tool_input: str) -> str:
        self._last_tool_used = "LLM"
        return self._run_llm_tool(tool_input, self._current_history_text)

    def _action_tool_func(self, tool_input: str) -> str:
        self._last_tool_used = "ACTION"
        return self._run_action_tool(tool_input)

    def _route_with_llm(self, message: str, history_text: str) -> RouteDecision:
        """Deprecated in favor of tools agent; kept for compatibility."""
        return self._fallback_route(message)

    def _fallback_route(self, message: str) -> RouteDecision:
        """Deterministic fallback routing when LLM router is unavailable."""
        self.logger.info("Running fallback router for message: %s", message)
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
            "Keep your response concise (max 2 sentences).\n"
            f"Conversation history:\n{history_text}\n\n"
            f"User: {message}\n"
            "Assistant:"
        )
        response = self.llm.generate(prompt)
        if response and str(response).strip():
            return str(response).strip()
        return "I’m sorry, I’m having trouble generating a response right now. Please try again."

    def _run_rag_tool(self, message: str, history_text: str) -> tuple[str, str]:
        try:
            prompt_template, prompt_kwargs, mode = self.retriever.retrieve_and_build_prompt(
                message,
                history_text,
            )
            prompt = prompt_template.format(**prompt_kwargs)
            response = self.llm.generate(prompt)
            if not response or not str(response).strip():
                self.logger.warning("Empty RAG generation, falling back to LLM")
                return self._run_llm_tool(message, history_text), "LLM_ONLY"
            return str(response).strip(), mode.upper()
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

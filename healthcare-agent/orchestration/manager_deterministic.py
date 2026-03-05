import logging
import os
import re
import time
import json
from typing import Dict, Any, Optional, Literal

from pydantic import BaseModel, Field
# from langchain.agents import create_agent
# from langchain_core.tools import Tool

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


import json
from pydantic import ValidationError

class DeterministicAgentManager:
    def __init__(self):
        self.memory = ConversationMemory()
        self.logger = logging.getLogger("DeterministicAgentManager")
        self.max_history_turns = int(os.getenv("MAX_HISTORY_TURNS", "3"))

        self.llm = OllamaClient()
        self.retriever = Retriever()
        self.appointments = AppointmentManager()
        self.messaging = MessageManager()

        # optional: keep this deterministic shortcut
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
        """Very simple keyword router to avoid unnecessary LLM calls."""
        
        msg = message.lower()

        action_keywords = [
            "appointment",
            "book",
            "schedule",
            "cancel",
            "reschedule",
            "slot",
            "doctor",
            "message doctor"
        ]

        rag_keywords = [
            "prepare",
            "preparation",
            "clinic",
            "policy",
            "insurance",
            "treatment",
            "procedure",
            "how does"
        ]

        if any(k in msg for k in action_keywords):
            return RouteDecision(
                tool="ACTION",
                reason="Detected appointment/doctor action keywords"
            )

        if any(k in msg for k in rag_keywords):
            return RouteDecision(
                tool="RAG",
                reason="Detected knowledge/policy question"
            )

        return RouteDecision(
            tool="LLM",
            reason="Default conversational response"
        )

    def _run_llm_tool(self, message: str, history_text: str) -> str:
        """
        Plain LLM response (no retrieval, no action).
        """
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
        """
        Retrieval-augmented answer. Returns (response, mode).
        mode is useful for your UI/analytics.
        """
        try:
            # Adjust these names if your Retriever API differs.
            # Common patterns: retrieve(), search(), get_context(), query()
            docs = self.retriever.retrieve(message)  # <-- if this breaks, see note below
            context = self._format_retrieved_docs(docs)

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
            # fallback to plain LLM if retrieval breaks
            return self._run_llm_tool(message, history_text), "RAG_FALLBACK_LLM"

    def _format_retrieved_docs(self, docs: Any) -> str:
        """
        Makes a readable context string from whatever your Retriever returns.
        Supports common shapes: list[Document], list[str], list[dict], plain str.
        """
        if docs is None:
            return ""

        # If retriever returns a string context already
        if isinstance(docs, str):
            return docs.strip()

        # If it's a list of things
        if isinstance(docs, list):
            chunks = []
            for d in docs[:6]:  # cap context size a bit
                if isinstance(d, str):
                    chunks.append(d.strip())
                elif isinstance(d, dict):
                    # common keys: "page_content", "content", "text"
                    chunks.append(str(d.get("page_content") or d.get("content") or d.get("text") or d).strip())
                else:
                    # langchain Document often has .page_content
                    page_content = getattr(d, "page_content", None)
                    chunks.append((page_content if page_content is not None else str(d)).strip())
            return "\n\n---\n\n".join([c for c in chunks if c])

        # Otherwise just stringify
        return str(docs).strip()

    def _run_action_tool(self, message: str) -> str:
        msg = message.lower().strip()

        # helpers
        def find_appt_id(text: str) -> Optional[str]:
            m = re.search(r"\b(appt_\d+)\b", text, re.IGNORECASE)
            return m.group(1).lower() if m else None

        def find_slot_ids(text: str) -> list[str]:
            return [s.lower() for s in re.findall(r"\b(slot_\d+)\b", text, re.IGNORECASE)]

        # 1) list/show commands
        if any(k in msg for k in ["available", "free slots", "list slots", "slots"]):
            slots = self.appointments.list_available_slots()
            if not slots:
                return "No available slots found."

            # keep it short for chat UI
            preview = slots[:10]
            lines = [f"{s['id']} — day {s['day']} at {s['time']}" for s in preview]
            more = "" if len(slots) <= 10 else f"\n…and {len(slots) - 10} more."
            return "Here are available slots:\n" + "\n".join(lines) + more

        if any(k in msg for k in ["my appointments", "show appointments", "list appointments"]):
            appts = self.appointments.get_patient_appointments()
            if not appts:
                return "You have no appointments scheduled."

            lines = [f"{a['id']} — {a['time']} ({a['slot_id']})" for a in appts]
            return "Your appointments:\n" + "\n".join(lines)

        # 2) cancel
        if any(k in msg for k in ["cancel", "delete appointment", "remove appointment"]):
            appt_id = find_appt_id(message)
            if not appt_id:
                return "To cancel, please include the appointment id (e.g., appt_2)."
            ok = self.appointments.cancel_appointment(appt_id)
            return "Canceled." if ok else f"Could not cancel {appt_id}."

        # 3) reschedule
        if any(k in msg for k in ["reschedule", "move appointment", "change appointment"]):
            appt_id = find_appt_id(message)
            slots = find_slot_ids(message)

            if not appt_id:
                return "To reschedule, please include the appointment id (e.g., appt_2)."
            if len(slots) < 1:
                return "To reschedule, please include the new slot id (e.g., slot_17)."

            new_slot_id = slots[-1]  # last mentioned slot
            ok = self.appointments.reschedule_appointment(appt_id, new_slot_id)
            return "Rescheduled." if ok else f"Could not reschedule {appt_id} to {new_slot_id}."

        # 4) book
        if any(k in msg for k in ["book", "schedule", "make appointment", "reserve"]):
            slots = find_slot_ids(message)
            if not slots:
                return (
                    "To book, please include the slot id (e.g., slot_17). "
                    "You can also ask: 'list available slots'."
                )
            slot_id = slots[-1]
            ok = self.appointments.book_appointment(slot_id)
            return "Appointment booked successfully." if ok else f"Could not book {slot_id} (it may already be taken)."

        # 5) messaging (wire this to your MessageManager API)
        if any(k in msg for k in ["message doctor", "contact doctor", "send message"]):
            # Replace this with your real MessageManager call:
            # return self.messaging.send_message_from_text(message)
            return "Messaging is routed correctly, but MessageManager is not wired in here yet."

        return (
            "I can help with appointments. Try:\n"
            "- 'list available slots'\n"
            "- 'book slot_17'\n"
            "- 'cancel appt_2'\n"
            "- 'reschedule appt_2 to slot_19'\n"
            "- 'my appointments'"
        )

    def _normalize_response(self, response: Any, message: str, history_text: str) -> str:
        """
        Ensures output is a clean string for the UI.
        """
        if response is None:
            return ""
        if isinstance(response, str):
            return response.strip()
        # Some libs return dicts like {"output": "..."} or {"response": "..."}
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
        # 1) optional deterministic pre-route (fast and predictable)
        if self.enable_keyword_preroute:
            quick = self._fallback_route(message)
            if quick.tool != "LLM":
                return quick

        # 2) LLM router (must output strict JSON)
        prompt = (
            "You are a router for a healthcare assistant. Choose exactly ONE tool.\n\n"
            "Tools:\n"
            "- ACTION: booking/canceling/rescheduling appointments, listing slots, sending doctor messages.\n"
            "- RAG: questions about clinic rules/policies/procedures/preparation that must be grounded in the knowledge base.\n"
            "- LLM: casual chat, greetings, emotional support, generic questions that don't require RAG or ACTION.\n\n"
            "Return ONLY a JSON object with keys: tool, reason.\n"
            'tool must be one of: "LLM", "RAG", "ACTION".\n\n'
            f"Conversation history:\n{history_text}\n\n"
            f"User message:\n{message}\n"
        )

        raw = self.llm.generate(prompt) or ""
        raw = raw.strip()

        # Try direct JSON parse
        try:
            obj = json.loads(raw)
            return RouteDecision(**obj)
        except (json.JSONDecodeError, ValidationError, TypeError):
            pass

        # Try to salvage JSON if model adds extra text
        obj = self._extract_first_json_object(raw)
        if obj is not None:
            try:
                return RouteDecision(**obj)
            except (ValidationError, TypeError):
                pass

        # Final deterministic fallback
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
import logging
import os
import re
import json
from typing import Dict, Any, Optional, Literal

from pydantic import BaseModel, Field, ValidationError

from orchestration.memory import ConversationMemory
from orchestration.tool_implementations import ToolExecutor
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

        self.tools_exec = ToolExecutor(self.retriever, self.appointments, self.messaging)

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
            context_str, mode = self.tools_exec.run_rag_tool(message, history_text)

            prompt = (
                "You are a healthcare assistant. Answer using ONLY the provided context.\n"
                "If the context is insufficient, say you don't know and suggest what to ask next.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Conversation history:\n{history_text}\n\n"
                f"User message:\n{message}\n"
            )
            answer = (self.llm.generate(prompt) or "").strip()
            return answer, "RAG"
        except Exception as e:
            self.logger.exception("RAG tool failed: %s", e)
            return self._run_llm_tool(message, history_text), "RAG_FALLBACK_LLM"

    def _run_action_tool(self, message: str) -> str:
        message = self._normalize_tool_input(message)
        return self.tools_exec.run_action_tool(message)

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
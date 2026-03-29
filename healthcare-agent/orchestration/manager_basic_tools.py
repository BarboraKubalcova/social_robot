import logging
import os
import re
import time
from typing import Any, Dict, Optional

from langchain_core.tools import Tool
from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError

from orchestration.memory import ConversationMemory
from orchestration.tool_implementations import ToolExecutor
from execution.ollama_client import OllamaClient
from execution.rag.retrieve import Retriever
from execution.actions.appointments import AppointmentManager
from execution.actions.messaging import MessageManager

from contextlib import contextmanager


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

        self.tools_exec = ToolExecutor(self.retriever, self.appointments, self.messaging)

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

        try:
            result = self.tools_agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 6},
            )
        except GraphRecursionError:
            logger.warning("Agent hit recursion limit for: %s", user_message)
            return "I'm sorry, I had trouble processing that. Could you rephrase?", "ERROR"
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

    def _action_tool_func(self, tool_input: str) -> str:
        self._last_tool_used = "ACTION"
        return self.tools_exec.run_action_tool(tool_input)

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

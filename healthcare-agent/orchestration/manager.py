import logging
import os
import re
import time
import json
from typing import Any, Dict, Optional
from pydantic import BaseModel
from typing import Literal

from langchain_core.tools import Tool

from orchestration.memory import ConversationMemory
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

class RouteDecision(BaseModel):
    tool: Literal["RAG", "LLM", "ACTION"]

ROUTER_PROMPT = """You are a router for a healthcare assistant.
Choose exactly one tool: RAG, LLM, ACTION.

Rules:
- ACTION: booking/cancel/reschedule/list slots/send message/email
- RAG: questions needing clinic/procedure/policy knowledge base (also ML math chunks exist)
- LLM: greetings, chit-chat, empathetic replies, or uncertain

Return ONLY valid JSON like: {"tool":"RAG"}.

Message: """


class AgentManager:
    """
    Router + tools wrapper.

    - Preferred: LangChain agent decides which tool to call (RAG / LLM / ACTION).
    - Fallback: simple heuristic routing if agent creation/invocation fails.
    """

    def __init__(self) -> None:
        self.memory = ConversationMemory()
        self.max_history_turns = int(os.getenv("MAX_HISTORY_TURNS", "3"))
        self.router_mode = os.getenv("ROUTER_MODE", "agent")  # "agent" | "keyword"

        # Clients / subsystems
        self.llm = OllamaClient()
        self.retriever = Retriever()
        self.appointments = AppointmentManager()
        self.messaging = MessageManager()

        # Mutable per-request state (kept minimal)
        self._last_tool_used: Optional[str] = None
        self._current_history_text: str = ""

        self.tools = [
            Tool(
                name="RAG",
                description=(
                "Use for questions that should be grounded in the knowledge base, such as medical procedures, "
                "clinic rules/policies, and preparation guidelines (e.g., 'What is an MRI?', "
                "'How do I prepare for an ultrasound?', 'Can I bring someone with me?'). "
                "Note: the knowledge base also contains some chunks about machine learning mathematics "
                "(included for testing/experimentation), so RAG may also answer those if asked."
                ),
                func=self._rag_tool_func,
            ),
            Tool(
                name="LLM",
                description=(
                    "Use for casual chat, greetings, and general questions not requiring retrieval "
                    "or operational actions. Use when unsure."
                    "Also use this tool when the patient feels emotional and would benefit from a more empathetic, free-form response."
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

        self.tools_agent = self._build_agent() if self.router_mode == "agent" else None
        logger.info("Router mode: %s", self.router_mode)

    # -------- Public API --------

    async def process_message(self, user_message: str, user_id: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        logger.info("User=%s msg=%s", user_id, user_message)

        self.memory.add_turn("user", user_message)

        self._current_history_text = self._build_history_text()
        self._last_tool_used = None

        response, intent = self._route_and_respond(user_message)

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

    # -------- Agent wiring --------

    def _build_agent(self):
        """
        LangChain APIs have shifted over time. Prefer `langchain.agents.create_agent`,
        but fall back to LangGraph's prebuilt `create_react_agent` if needed.
        """
        system_prompt = (
            "You are a healthcare assistant router.\n"
            "Pick EXACTLY ONE tool: RAG, LLM, or ACTION.\n"
            "Call it once, then return the tool output as the final answer.\n"
            "Do not call a second tool. Do not rephrase tool output."
        )

        try:
            from langchain.agents import create_agent  # LangChain v1 style

            return create_agent(
                model=self.llm.llm,
                tools=self.tools,
                system_prompt=system_prompt,
                debug=False,
            )
        except Exception:
            # Fallback for installations where create_agent is missing/moved
            try:
                from langgraph.prebuilt import create_react_agent  # deprecated but common

                # create_react_agent typically returns a graph/executor with `.invoke(...)`
                return create_react_agent(
                    model=self.llm.llm,
                    tools=self.tools,
                    prompt=system_prompt,
                )
            except Exception as exc:
                logger.warning("Could not build agent, will use keyword routing: %s", exc)
                return None
    
    def _route_once_with_llm(self, user_message: str) -> str:
        prompt = f"{ROUTER_PROMPT}{user_message}"
        raw = self.llm.generate(prompt)

        try:
            data = json.loads(str(raw).strip())
            decision = RouteDecision(**data)
            return decision.tool
        except Exception:
            # hard fallback
            return self._fallback_tool_choice(user_message)
    
    def _route_and_respond(self, user_message: str) -> tuple[str, str]:
        # 0) ultra-fast deterministic ACTION detection first (saves router call)
        tool = self._fallback_tool_choice(user_message)
        if tool != "LLM":
            return self._run_tool(tool, user_message)

        # 1) single LLM routing call (no loops)
        tool = self._route_once_with_llm(user_message)
        return self._run_tool(tool, user_message)

    def _run_tool(self, tool: str, user_message: str) -> tuple[str, str]:
        if tool == "ACTION":
            self._last_tool_used = "ACTION"
            return self._run_action_tool(user_message), "ACTION"
        if tool == "RAG":
            self._last_tool_used = "RAG"
            text, mode = self._run_rag_tool(user_message)
            return text, mode
        self._last_tool_used = "LLM"
        return self._run_llm_tool(user_message), "LLM_ONLY"

    # def _route_and_respond(self, user_message: str) -> tuple[str, str]:
    #     # 1) Agent router
    #     if self.tools_agent is not None:
    #         try:
    #             with timed("agent.invoke"):
    #                 result = self.tools_agent.invoke(
    #                     {
    #                         "messages": [
    #                             {
    #                                 "role": "user",
    #                                 "content": (
    #                                     f"Conversation history:\n{self._current_history_text}\n\n"
    #                                     f"User message:\n{user_message}"
    #                                 ),
    #                             }
    #                         ]
    #                     },
    #                     # Keep the loop tight: model -> tool -> model -> stop
    #                     config={"recursion_limit": 4},
    #                 )
    #                 response = self._extract_final_text(result)
    #                 return response, self._resolve_intent()
    #         except Exception as exc:
    #             logger.warning("Agent invocation failed, falling back: %s", exc)

        # # 2) Fallback router
        # tool = self._fallback_tool_choice(user_message)
        # if tool == "ACTION":
        #     self._last_tool_used = "ACTION"
        #     return self._run_action_tool(user_message), "ACTION"
        # if tool == "RAG":
        #     self._last_tool_used = "RAG"
        #     text, mode = self._run_rag_tool(user_message)
        #     return text, mode
        # self._last_tool_used = "LLM"
        # return self._run_llm_tool(user_message), "LLM_ONLY"

    # -------- Tool callbacks (called by the agent) --------

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
        prompt = (
            "You are a helpful healthcare assistant.\n"
            "Keep your response concise.\n\n"
            f"Conversation history:\n{self._current_history_text}\n\n"
            f"Current message: {message}\n"
        )
        with timed("llm.generate (llm)"):
            response = self.llm.generate(prompt)
        return str(response).strip() if response and str(response).strip() else (
            "I’m sorry, I’m having trouble generating a response right now. Please try again."
        )

    def _run_rag_tool(self, message: str) -> tuple[str, str]:
        try:
            with timed("retriever.retrieve_and_build_prompt"):
                prompt_template, prompt_kwargs, mode = self.retriever.retrieve_and_build_prompt(
                    message,
                    self._current_history_text,
                )
            prompt = prompt_template.format(**prompt_kwargs)
            with timed("llm.generate (rag)"):
                response = self.llm.generate(prompt)
            if response and str(response).strip():
                return str(response).strip(), str(mode).upper()
            logger.warning("Empty RAG response, falling back to LLM")
        except Exception as exc:
            logger.warning("RAG failed, falling back to LLM: %s", exc)
        return self._run_llm_tool(message), "LLM_ONLY"

    def _run_action_tool(self, message: str) -> str:
        msg = message.lower()

        slot_id = self._extract_slot_id(msg)
        appt_id = self._extract_appointment_id(msg)

        if self._wants_slots(msg):
            slots = self.appointments.list_available_slots()
            if not slots:
                return "There are no available slots in the current timetable."
            preview = ", ".join(f"{s['id']} ({s['time']})" for s in slots[:8])
            return f"Here are available slots: {preview}"

        if re.search(r"\b(reschedule|rebook)\b", msg):
            new_slot_id = slot_id
            if not appt_id or not new_slot_id:
                return "Please provide both appointment id (e.g. appt_1) and new slot id (e.g. slot_12)."
            ok = self.appointments.reschedule_appointment(appt_id, new_slot_id)
            return (
                f"Appointment {appt_id} was rescheduled to {new_slot_id}."
                if ok
                else f"I could not reschedule {appt_id} to {new_slot_id}."
            )

        if re.search(r"\bcancel\b", msg):
            if not appt_id:
                appts = self.appointments.get_patient_appointments()
                if not appts:
                    return "There is no active appointment to cancel."
                appt_id = appts[0]["id"]
            ok = self.appointments.cancel_appointment(appt_id)
            return (
                f"Appointment {appt_id} has been canceled."
                if ok
                else f"I could not cancel appointment {appt_id}."
            )

        # "book ..." OR just mentioning a slot
        if re.search(r"\bbook\b", msg) or slot_id:
            if not slot_id:
                available = self.appointments.list_available_slots()
                if not available:
                    return "There are no available slots to book right now."
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
            return "I have sent your message to the doctor." if ok else "I could not send the message."

        return "I can help with booking, canceling, rescheduling appointments, listing slots, or sending a doctor message."

    # -------- Helpers --------

    def _extract_final_text(self, result: Any) -> str:
        """
        Pull the final assistant/tool text from various possible return shapes.
        Keeps your original idea, but tighter.
        """
        if not isinstance(result, dict):
            text = str(result).strip()
            return text if text else "I'm sorry, I couldn't process your request."

        messages = result.get("messages") or []
        for msg in reversed(messages):
            # dict or LC message object
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None) or getattr(msg, "type", None)
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)

            tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
            if tool_calls and not content:
                continue

            text = self._extract_text_content(content)
            if role in ("assistant", "ai", "tool") and text:
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
            return "\n".join(parts).strip()
        return str(content).strip()

    def _resolve_intent(self) -> str:
        return (self._last_tool_used or "LLM_ONLY").upper()

    def _fallback_tool_choice(self, message: str) -> str:
        """
        A simple, deterministic backup router.
        """
        msg = message.lower()

        if re.search(r"\b(book|cancel|reschedule|rebook|slot|appointment|send message|email)\b", msg):
            return "ACTION"

        # RAG-ish cues: informational / policy / procedure questions
        if re.search(r"\b(what is|how do i|how to|prepare|policy|procedure|mri|ultrasound|ct|x-ray)\b", msg):
            return "RAG"

        return "LLM"

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
            )
        )

    def _extract_slot_id(self, message_lower: str) -> Optional[str]:
        m = re.search(r"\bslot[_\s-]?(\d+)\b", message_lower)
        return f"slot_{m.group(1)}" if m else None

    def _extract_appointment_id(self, message_lower: str) -> Optional[str]:
        m = re.search(r"\bappt[_\s-]?(\d+)\b", message_lower)
        return f"appt_{m.group(1)}" if m else None



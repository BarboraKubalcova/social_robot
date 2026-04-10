import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from orchestration.memory import ConversationMemory
from orchestration.tool_implementations import ToolExecutor
from execution.ollama_client import OllamaClient
from execution.rag.retrieve import Retriever
from execution.actions.appointments import AppointmentManager
from execution.actions.messaging import MessageManager

from contextlib import contextmanager


logger = logging.getLogger("MultiAgentManager")

PROMPTS_DIR = Path(__file__).parent / "prompts"


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s took %.2fs", label, time.perf_counter() - t0)


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text()


def _parse_json(text: str) -> Optional[Dict]:
    """Extract the first JSON object from LLM output."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        pass

    # Find first { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

class AppointmentAgent:
    """Handles all appointment-related operations."""

    def __init__(self, llm: OllamaClient, appointments: AppointmentManager, tools_exec: ToolExecutor):
        self.llm = llm
        self.appointments = appointments
        self.tools_exec = tools_exec
        self.prompt_template = _load_prompt("appointment_agent_prompt.md")

    def execute(self, task: str, history_text: str) -> Dict[str, Any]:
        """Run the appointment agent and return a structured report."""
        available_slots = self.appointments.list_available_slots()
        booked = self.appointments.get_patient_appointments()

        slots_preview = ", ".join(
            f"{s['id']} ({s.get('day', '')} {s.get('date', '')} {s.get('time', '')})"
            for s in available_slots[:12]
        )
        appts_preview = ", ".join(
            f"{a['id']} ({a.get('day', '')} {a.get('date', '')} {a.get('time', '')})"
            for a in booked[:8]
        )
        available_data = (
            f"Available slots: {slots_preview or 'none'}\n"
            f"Booked appointments: {appts_preview or 'none'}"
        )

        prompt = self.prompt_template.format(
            available_data=available_data,
            history=history_text,
            task=task,
        )

        with timed("appointment_agent LLM"):
            raw = self.llm.generate(prompt)

        logger.info("appointment_agent raw LLM output: %s", raw[:500])

        decision = _parse_json(raw)
        if not decision:
            return {"success": False, "report": "Could not understand the appointment request. Please try again."}

        action = decision.get("action", "")
        if not action:
            # LLM returned JSON without an "action" key.
            # Try to infer the action from the content instead of blindly trusting LLM text.
            action = self._infer_action_from_decision(decision, task)
            if action:
                decision["action"] = action
            else:
                fallback = decision.get("answer") or decision.get("message") or json.dumps(decision, indent=2)
                return {"success": True, "report": fallback}

        # Resolve slot_id from day+time if the LLM didn't provide one directly
        if not decision.get("slot_id") and (decision.get("day") or decision.get("time")):
            computed = self._compute_slot_id(decision)
            if computed:
                decision["slot_id"] = computed
                logger.info("Computed slot_id=%s from day=%s time=%s", computed, decision.get("day"), decision.get("time"))

        return self._execute_action(action, decision, available_slots, booked)

    def _execute_action(
        self, action: str, decision: Dict, available_slots: list, booked: list
    ) -> Dict[str, Any]:
        if action == "list_slots":
            result = self.tools_exec.run_list_slots()
            return {"success": True, "report": result}

        if action == "list_appointments":
            result = self.tools_exec.run_list_appointments()
            return {"success": True, "report": result}

        if action == "book":
            slot_id = decision.get("slot_id") or self._infer_slot(decision, available_slots)
            if not slot_id:
                return {"success": False, "report": "Could not determine which slot to book. Please specify a slot or day."}
            result = self.tools_exec.run_book_appointment_by_id(slot_id)
            return {"success": "successfully" in result.lower(), "report": result}

        if action == "cancel":
            appt_id = decision.get("appointment_id") or self._infer_appointment(decision, booked)
            if not appt_id:
                return {"success": False, "report": "Could not determine which appointment to cancel. Please specify."}
            result = self.tools_exec.run_cancel_appointment_by_id(appt_id)
            return {"success": "canceled" in result.lower() or "successfully" in result.lower(), "report": result}

        if action == "reschedule":
            appt_id = decision.get("appointment_id") or self._infer_appointment(decision, booked)
            new_slot = decision.get("new_slot_id") or decision.get("slot_id") or self._infer_slot(decision, available_slots)
            if not appt_id:
                return {"success": False, "report": "Could not determine which appointment to reschedule."}
            if not new_slot:
                return {"success": False, "report": "Please specify the new day or slot for rescheduling."}
            result = self.tools_exec.run_reschedule_appointment_by_id(appt_id, new_slot)
            return {"success": "successfully" in result.lower(), "report": result}

        if action == "clarify":
            msg = decision.get("message", "Could you please provide more details about your appointment request?")
            return {"success": True, "report": msg, "needs_clarification": True}

        return {"success": False, "report": f"Unknown appointment action: {action}"}

    @staticmethod
    def _infer_action_from_decision(decision: Dict, task: str) -> Optional[str]:
        """Infer the intended action when the LLM omits the 'action' key."""
        # If the decision has a slot_id or mentions booking keywords, treat as book
        text = (decision.get("message") or decision.get("answer") or "").lower() + " " + task.lower()
        if decision.get("slot_id") or "book" in text:
            return "book"
        if decision.get("appointment_id") and ("cancel" in text):
            return "cancel"
        if decision.get("appointment_id") and decision.get("new_slot_id"):
            return "reschedule"
        if "list" in text and "slot" in text:
            return "list_slots"
        if "list" in text and "appointment" in text:
            return "list_appointments"
        if decision.get("available_slots"):
            return "list_slots"
        return None

    # Day+time → slot_id deterministic mapping
    _TIMES = ["07:30", "08:30", "09:30", "10:30", "11:30", "13:00", "14:00", "15:00", "16:00"]
    _DAY_OFFSETS = {"monday": 0, "tuesday": 9, "wednesday": 18}

    def _compute_slot_id(self, decision: Dict) -> Optional[str]:
        """Deterministically compute slot_id from day and time fields."""
        day = (decision.get("day") or "").strip().lower()
        time_str = (decision.get("time") or "").strip()

        if not day or day not in self._DAY_OFFSETS:
            return None
        if not time_str:
            return None

        # Normalize time: "10:30" or "10:30 AM" → "10:30"
        time_str = re.sub(r"\s*(AM|PM)\s*", "", time_str, flags=re.IGNORECASE).strip()
        # Zero-pad: "7:30" → "07:30"
        if len(time_str) == 4:
            time_str = "0" + time_str

        if time_str not in self._TIMES:
            return None

        slot_number = self._DAY_OFFSETS[day] + self._TIMES.index(time_str) + 1
        return f"slot_{slot_number}"

    def _infer_slot(self, decision: Dict, available_slots: list) -> Optional[str]:
        """Try to match a day+time to an available slot."""
        # First try deterministic computation
        computed = self._compute_slot_id(decision)
        if computed:
            return computed

        day = decision.get("day", "")
        time_str = decision.get("time", "")
        if not day:
            return available_slots[0]["id"] if available_slots else None

        # Match by day and time if both provided
        if time_str:
            for s in available_slots:
                if (day.lower() in (s.get("day", "") or "").lower()
                        and time_str in (s.get("time", "") or "")):
                    return s["id"]

        # Match by day only — return first available on that day
        for s in available_slots:
            if day.lower() in (s.get("day", "") or "").lower():
                return s["id"]
        return None

    def _infer_appointment(self, decision: Dict, booked: list) -> Optional[str]:
        """Try to find the appointment by day or return the only one."""
        day = decision.get("day", "")
        if day:
            for a in booked:
                if day.lower() in (a.get("day", "") or "").lower():
                    return a["id"]
        if len(booked) == 1:
            return booked[0]["id"]
        return None


class MedicalKnowledgeAgent:
    """Answers medical/clinic questions using RAG."""

    def __init__(self, llm: OllamaClient, retriever: Retriever, tools_exec: ToolExecutor):
        self.llm = llm
        self.retriever = retriever
        self.tools_exec = tools_exec
        self.prompt_template = _load_prompt("medical_knowledge_agent_prompt.md")

    def execute(self, task: str, history_text: str) -> Dict[str, Any]:
        with timed("medical_knowledge_agent RAG retrieval"):
            context_str, mode = self.tools_exec.run_rag_tool(task, history_text)

        prompt = self.prompt_template.format(
            context=context_str,
            history=history_text,
            task=task,
        )

        with timed("medical_knowledge_agent LLM"):
            raw = self.llm.generate(prompt)

        parsed = _parse_json(raw)
        if parsed:
            answer = parsed.get("answer", raw)
            sources = parsed.get("sources", [])
            grounded = parsed.get("grounded", mode == "rag")
            source_note = f" (Sources: {', '.join(sources)})" if sources else ""
            return {
                "success": True,
                "report": f"{answer}{source_note}",
                "grounded": grounded,
                "mode": mode,
            }

        # Fallback: use raw text if JSON parsing fails
        return {"success": True, "report": raw.strip(), "grounded": mode == "rag", "mode": mode}


class MessagingAgent:
    """Handles doctor messaging."""

    def __init__(self, llm: OllamaClient, messaging: MessageManager, tools_exec: ToolExecutor):
        self.llm = llm
        self.messaging = messaging
        self.tools_exec = tools_exec
        self.prompt_template = _load_prompt("messaging_agent_prompt.md")

    def execute(self, task: str, history_text: str) -> Dict[str, Any]:
        doctors = self.messaging.list_doctors()
        doctors_text = ", ".join(f"{d['id']}: {d['name']}" for d in doctors)

        prompt = self.prompt_template.format(
            doctors=doctors_text,
            history=history_text,
            task=task,
        )

        with timed("messaging_agent LLM"):
            raw = self.llm.generate(prompt)

        decision = _parse_json(raw)
        if not decision:
            return {"success": False, "report": "Could not process the messaging request."}

        if decision.get("action") == "clarify":
            return {"success": True, "report": decision.get("message", "What would you like to say to the doctor?"), "needs_clarification": True}

        recipient = decision.get("recipient", doctors[0]["id"] if doctors else "doc_1")
        subject = decision.get("subject", "Patient message")
        body = decision.get("body", task)

        result = self.tools_exec.run_send_doctor_message(
            f"To: {recipient}, Subject: {subject}, Body: {body}"
        )
        if "sent" in result.lower():
            doctor_name = next((d["name"] for d in doctors if d["id"] == recipient), recipient)
            return {"success": True, "report": f"Message sent to {doctor_name}. Subject: {subject}"}
        return {"success": False, "report": "Could not send the message. Please try again."}


class ChatAgent:
    """Handles casual conversation and emotional support."""

    def __init__(self, llm: OllamaClient):
        self.llm = llm
        self.prompt_template = _load_prompt("chat_agent_prompt.md")

    def execute(self, task: str, history_text: str) -> Dict[str, Any]:
        prompt = self.prompt_template.format(
            history=history_text,
            task=task,
        )

        with timed("chat_agent LLM"):
            raw = self.llm.generate(prompt)

        parsed = _parse_json(raw)
        if parsed:
            return {"success": True, "report": parsed.get("answer", raw)}

        # Fallback: use raw text
        return {"success": True, "report": raw.strip()}


# ---------------------------------------------------------------------------
# Coordinator / main manager
# ---------------------------------------------------------------------------

FINAL_ANSWER_PROMPT = """You are a healthcare assistant coordinator.
A specialist agent just completed a task for the user. Compose a final, friendly answer for the patient based on the agent's report.

User's original message: {user_message}
Agent that handled the task: {agent_name}
Agent's report: {report}

Instructions:
- Rewrite the agent's report into a natural, patient-friendly response.
- Be concise and clear.
- If the task was successful, confirm it.
- If it failed or needs clarification, communicate that kindly.
- Do NOT add information that is not in the report.
"""

AGENT_NAME_TO_INTENT = {
    "appointment_agent": "ACTION",
    "medical_knowledge_agent": "RAG",
    "messaging_agent": "ACTION",
    "chat_agent": "CHAT",
}

VALID_AGENTS = {"appointment_agent", "medical_knowledge_agent", "messaging_agent", "chat_agent"}


class MultiAgentManager:
    """
    Multi-agent healthcare assistant.

    A coordinator agent routes user messages to specialist agents,
    each of which executes its task and returns a structured report.
    The coordinator then composes the final user-facing answer.
    """

    def __init__(self) -> None:
        self.memory = ConversationMemory()
        self.max_history_turns = int(os.getenv("MAX_HISTORY_TURNS", "3"))

        self.llm = OllamaClient()
        self.retriever = Retriever()
        self.appointments = AppointmentManager()
        self.messaging = MessageManager()

        self.tools_exec = ToolExecutor(self.retriever, self.appointments, self.messaging)

        self.coordinator_prompt = _load_prompt("coordinator_prompt.md")

        # Initialize specialist agents
        self.agents: Dict[str, Any] = {
            "appointment_agent": AppointmentAgent(self.llm, self.appointments, self.tools_exec),
            "medical_knowledge_agent": MedicalKnowledgeAgent(self.llm, self.retriever, self.tools_exec),
            "messaging_agent": MessagingAgent(self.llm, self.messaging, self.tools_exec),
            "chat_agent": ChatAgent(self.llm),
        }

        logger.info("MultiAgentManager initialized with agents: %s", list(self.agents.keys()))

    async def process_message(self, user_message: str, user_id: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        logger.info("User=%s msg=%s", user_id, user_message)

        self.memory.add_turn("user", user_message)
        history_text = self._build_history_text()

        # Step 1: Coordinator decides which agent to delegate to
        with timed("coordinator routing"):
            delegation = self._route(user_message, history_text)

        agent_name = delegation.get("agent", "chat_agent")
        task = delegation.get("task", user_message)

        if agent_name not in VALID_AGENTS:
            logger.warning("Coordinator selected unknown agent '%s', falling back to chat_agent", agent_name)
            agent_name = "chat_agent"

        logger.info("Delegating to %s: %s", agent_name, task[:80])

        # Step 2: Specialist agent executes the task
        with timed(f"{agent_name} execution"):
            report = self.agents[agent_name].execute(task, history_text)

        logger.info("Agent %s report: success=%s", agent_name, report.get("success"))

        # Step 3: Coordinator composes the final answer
        with timed("coordinator final answer"):
            response = self._compose_final_answer(user_message, agent_name, report)

        intent = AGENT_NAME_TO_INTENT.get(agent_name, "CHAT")
        self.memory.add_turn("assistant", response)

        logger.info(
            "Done user=%s agent=%s intent=%s total=%.2fs",
            user_id, agent_name, intent, time.perf_counter() - t0,
        )

        return {
            "response": response,
            "intent": intent,
            "agent": agent_name,
            "history": self.memory.get_recent_history(),
        }

    def _route(self, user_message: str, history_text: str) -> Dict:
        """Ask the coordinator LLM which agent should handle this message."""
        prompt = self.coordinator_prompt.format(
            history=history_text,
            user_message=user_message,
        )
        raw = self.llm.generate(prompt)
        parsed = _parse_json(raw)
        if parsed and parsed.get("agent") in VALID_AGENTS:
            return parsed

        # Keyword fallback if LLM output is unparseable
        return self._keyword_fallback(user_message)

    def _keyword_fallback(self, message: str) -> Dict:
        """Simple keyword-based routing as a safety net."""
        msg = message.lower()

        appointment_keywords = [
            "appointment", "book", "cancel", "reschedule", "slot",
            "schedule", "available", "rebook", "move", "free time",
        ]
        if any(kw in msg for kw in appointment_keywords):
            return {"agent": "appointment_agent", "task": message}

        medical_keywords = [
            "prepare", "procedure", "mri", "ultrasound", "blood",
            "surgery", "exam", "test", "diagnosis", "treatment",
            "fasting", "medication", "policy", "rule",
        ]
        if any(kw in msg for kw in medical_keywords):
            return {"agent": "medical_knowledge_agent", "task": message}

        messaging_keywords = ["message", "email", "send", "doctor", "write to"]
        if any(kw in msg for kw in messaging_keywords):
            return {"agent": "messaging_agent", "task": message}

        return {"agent": "chat_agent", "task": message}

    def _compose_final_answer(self, user_message: str, agent_name: str, report: Dict) -> str:
        """Let the coordinator LLM compose a patient-friendly final answer."""
        report_text = report.get("report", "No report available.")

        prompt = FINAL_ANSWER_PROMPT.format(
            user_message=user_message,
            agent_name=agent_name.replace("_", " ").title(),
            report=report_text,
        )

        raw = self.llm.generate(prompt)
        text = raw.strip() if raw else ""

        # Strip thinking tags if present (some models like qwen3 emit these)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        return text if text else report_text

    def _build_history_text(self) -> str:
        turns = self.memory.get_recent_history(limit=self.max_history_turns)
        return "\n".join(f"{t['role']}: {t['content']}" for t in turns)

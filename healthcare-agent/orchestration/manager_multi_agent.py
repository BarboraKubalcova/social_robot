"""
Multi-Agent Manager with Coordinator pattern.

Architecture:
  1. Coordinator agent reads the user message and delegates to one of four sub-agents.
  2. Each sub-agent uses the LLM to decide on its action, then executes the
     corresponding tool.  Every decision and execution step is timed and printed.

Sub-agents:
  - appointment_agent   → list_slots, list_appointments, book, cancel, reschedule
  - medical_knowledge_agent → RAG retrieval + grounded answer
  - messaging_agent     → send message to a doctor
  - chat_agent          → casual conversation / emotional support
"""

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

logger = logging.getLogger("MultiAgentManager")

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def _timed(label: str):
    """Simple context-manager that prints and logs elapsed time."""
    class _Timer:
        def __init__(self):
            self.elapsed = 0.0
        def __enter__(self):
            self._t0 = time.perf_counter()
            return self
        def __exit__(self, *_):
            self.elapsed = time.perf_counter() - self._t0
            msg = f"[TIMER] {label}: {self.elapsed:.3f}s"
            print(msg)
            logger.info(msg)
    return _Timer()


def _safe_format(template: str, **kwargs) -> str:
    """Replace {key} placeholders without touching other braces (safe for JSON)."""
    for key, value in kwargs.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def _parse_json(text: str) -> Optional[Dict]:
    """Extract the first JSON object from LLM output (tolerates markdown fences)."""
    # Strip <think>…</think> blocks
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"^[\s\S]*?</think>", "", text)

    # Try to find JSON in code fences first
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


class MultiAgentManager:
    """
    Coordinator-based multi-agent healthcare assistant.
    """

    def __init__(self) -> None:
        self.memory = ConversationMemory()
        self.max_history_turns = int(os.getenv("MAX_HISTORY_TURNS", "5"))

        self.llm = OllamaClient()
        self.retriever = Retriever()
        self.appointments = AppointmentManager()
        self.messaging = MessageManager()

        self.tools = ToolExecutor(self.retriever, self.appointments, self.messaging)

        # Load prompt templates once
        self._coordinator_prompt = _load_prompt("coordinator_prompt.md")
        self._appointment_prompt = _load_prompt("appointment_agent_prompt.md")
        self._medical_prompt = _load_prompt("medical_knowledge_agent_prompt.md")
        self._messaging_prompt = _load_prompt("messaging_agent_prompt.md")
        self._chat_prompt = _load_prompt("chat_agent_prompt.md")

        logger.info("MultiAgentManager initialized")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def process_message(self, user_message: str, user_id: str) -> Dict[str, Any]:
        total_t0 = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"[MULTI-AGENT] User={user_id} Message: {user_message}")
        print(f"{'='*60}")

        self.memory.add_turn("user", user_message)
        history_text = self._build_history_text()

        # --- Step 1: Coordinator decides which agent ---
        with _timed("Coordinator decision") as coord_timer:
            coordinator_result = self._run_coordinator(user_message, history_text)

        agent_name = coordinator_result.get("agent", "chat_agent")
        task = coordinator_result.get("task", user_message)
        reasoning = coordinator_result.get("reasoning", "")

        print(f"[COORDINATOR] Agent: {agent_name}")
        print(f"[COORDINATOR] Task: {task}")
        print(f"[COORDINATOR] Reasoning: {reasoning}")

        # --- Step 2: Delegate to the chosen sub-agent ---
        with _timed(f"Sub-agent '{agent_name}' execution") as agent_timer:
            if agent_name == "appointment_agent":
                response = self._run_appointment_agent(task, history_text, user_message)
            elif agent_name == "medical_knowledge_agent":
                response = self._run_medical_knowledge_agent(task, history_text, user_message)
            elif agent_name == "messaging_agent":
                response = self._run_messaging_agent(task, history_text)
            else:
                response = self._run_chat_agent(task, history_text)

        self.memory.add_turn("assistant", response)

        total_elapsed = time.perf_counter() - total_t0
        print(f"\n[MULTI-AGENT] Total time: {total_elapsed:.3f}s")
        print(f"[MULTI-AGENT] Final response: {response[:200]}")
        print(f"{'='*60}\n")

        return {
            "response": response,
            "intent": agent_name,
            "history": self.memory.get_recent_history(),
        }

    # ------------------------------------------------------------------
    # Coordinator
    # ------------------------------------------------------------------

    def _run_coordinator(self, user_message: str, history_text: str) -> Dict:
        prompt = _safe_format(
            self._coordinator_prompt,
            history=history_text,
            user_message=user_message,
        )
        raw = self.llm.generate(prompt)
        print(f"[COORDINATOR] Raw LLM output: {raw}")

        parsed = _parse_json(raw)
        if parsed and "agent" in parsed:
            return parsed

        # Fallback: try to extract agent name from free text
        print("[COORDINATOR] JSON parsing failed, attempting text recovery")
        agent = self._recover_agent_from_text(raw)
        if agent:
            return {"agent": agent, "task": user_message, "reasoning": "recovered from free-text LLM output"}

        # Final fallback: keyword-based routing
        print("[COORDINATOR] Recovery failed, falling back to keyword routing")
        return self._fallback_route(user_message)

    def _recover_agent_from_text(self, text: str) -> Optional[str]:
        """Try to extract agent name from free-text coordinator output."""
        text_lower = text.lower()
        agents = ["messaging_agent", "appointment_agent", "medical_knowledge_agent", "chat_agent"]
        for agent in agents:
            if agent in text_lower:
                print(f"[COORDINATOR] Recovered agent from text: {agent}")
                return agent
        # Also try partial matches
        if re.search(r"\bmessaging\b", text_lower):
            return "messaging_agent"
        if re.search(r"\bappointment\b", text_lower):
            return "appointment_agent"
        if re.search(r"\bmedical.knowledge\b", text_lower):
            return "medical_knowledge_agent"
        if re.search(r"\bchat\b", text_lower):
            return "chat_agent"
        return None

    def _fallback_route(self, message: str) -> Dict:
        msg = message.lower()

        if re.search(r"\b(appointment|book|cancel|reschedule|slot|schedule|rebook)\b", msg):
            return {"agent": "appointment_agent", "task": message, "reasoning": "keyword fallback: appointment-related"}
        if re.search(r"\b(message|email|write to|contact|send)\b.*\b(doctor|dr)\b", msg):
            return {"agent": "messaging_agent", "task": message, "reasoning": "keyword fallback: messaging"}
        if re.search(
            r"\b(mri|ct|ultrasound|x[\s-]?ray|procedure|preparation|prepare|examination|"
            r"treatment|policy|insurance|clinic|diagnos|roentgen)\b", msg
        ):
            return {"agent": "medical_knowledge_agent", "task": message, "reasoning": "keyword fallback: medical knowledge"}

        return {"agent": "chat_agent", "task": message, "reasoning": "keyword fallback: no specific match"}

    # ------------------------------------------------------------------
    # Appointment Agent
    # ------------------------------------------------------------------

    def _run_appointment_agent(self, task: str, history_text: str, user_message: str) -> str:
        # Gather available data for the prompt
        available_slots = self.appointments.list_available_slots()
        booked_appointments = self.appointments.get_patient_appointments()
        available_data = self._format_available_data(available_slots, booked_appointments)

        # Step 1: LLM decides which appointment action to take
        with _timed("Appointment agent LLM decision"):
            prompt = _safe_format(
                self._appointment_prompt,
                available_data=available_data,
                history=history_text,
                task=task,
            )
            raw = self.llm.generate(prompt)
            print(f"[APPOINTMENT AGENT] Raw LLM output: {raw}")

        parsed = _parse_json(raw)
        if not parsed:
            print("[APPOINTMENT AGENT] JSON parsing failed, attempting text recovery")
            parsed = self._recover_appointment_action_from_text(raw, task, user_message)

        if not parsed:
            print("[APPOINTMENT AGENT] Recovery failed, falling back to keyword action routing")
            return self._appointment_fallback(user_message)

        action = parsed.get("action", "clarify")
        print(f"[APPOINTMENT AGENT] Decided action: {action}")
        print(f"[APPOINTMENT AGENT] Parsed details: {json.dumps(parsed, indent=2)}")

        # Step 2: Execute the chosen action
        with _timed(f"Appointment tool execution ({action})"):
            result = self._execute_appointment_action(parsed, user_message)

        print(f"[APPOINTMENT AGENT] Tool result: {result}")
        return result

    def _execute_appointment_action(self, parsed: Dict, user_message: str) -> str:
        action = parsed.get("action", "clarify")

        if action == "list_slots":
            return self.tools.run_list_slots()

        if action == "list_appointments":
            return self.tools.run_list_appointments()

        if action == "book":
            slot_id = parsed.get("slot_id")
            if slot_id:
                return self.tools.run_book_appointment_by_id(slot_id)
            # Fallback: try to extract from day/time in parsed data
            day = parsed.get("day")
            time_val = parsed.get("time")
            if day or time_val:
                return self._book_by_day_time(day, time_val)
            return self.tools.run_book_appointment(user_message)

        if action == "cancel":
            appt_id = parsed.get("appointment_id")
            if appt_id:
                return self.tools.run_cancel_appointment_by_id(appt_id)
            return self.tools.run_cancel_appointment(user_message)

        if action == "reschedule":
            appt_id = parsed.get("appointment_id")
            new_slot_id = parsed.get("new_slot_id")
            if appt_id and new_slot_id:
                return self.tools.run_reschedule_appointment_by_id(appt_id, new_slot_id)
            return self.tools.run_reschedule_appointment(user_message)

        # clarify or unknown action
        message = parsed.get("message", "Could you please provide more details about your appointment request?")
        return message

    def _book_by_day_time(self, day: Optional[str], time_val: Optional[str]) -> str:
        """Attempt to find a matching slot by day and/or time and book it."""
        available = self.appointments.list_available_slots()
        if not available:
            return "There are no available slots to book right now."

        candidates = available
        if day:
            candidates = [s for s in candidates if s.get("day", "").lower() == day.lower()]
        if time_val:
            candidates = [s for s in candidates if s.get("time", "") == time_val]

        if candidates:
            slot_id = candidates[0]["id"]
            return self.tools.run_book_appointment_by_id(slot_id)

        label = " ".join(filter(None, [day, time_val]))
        return f"I could not find an available slot for {label}."

    def _appointment_fallback(self, message: str) -> str:
        """Keyword-based fallback if appointment LLM parsing fails."""
        return self.tools.run_action_tool(message)

    def _recover_appointment_action_from_text(
        self, llm_text: str, task: str, user_message: str
    ) -> Optional[Dict]:
        """Try to extract a usable action from free-text LLM output.

        If the LLM ignored the JSON-only instruction but still reasoned its way
        to a slot/time/appointment, we can salvage that into a proper action dict.
        """
        text = llm_text.lower()
        combined = (task + " " + user_message).lower()

        # Try to extract a slot_id mentioned in the text
        slot_match = re.search(r"\bslot[_\s-]?(\d+)\b", text)
        slot_id = f"slot_{slot_match.group(1)}" if slot_match else None

        # Try to extract an appointment_id mentioned in the text
        appt_match = re.search(r"\bappt[_\s-]?(\d+)\b", text)
        appt_id = f"appt_{appt_match.group(1)}" if appt_match else None

        # Try to extract a time (HH:MM) — take the last one as the "answer"
        time_matches = re.findall(r"\b(\d{1,2}:\d{2})\b", text)
        found_time = time_matches[-1] if time_matches else None

        # Try to extract a day
        day_match = re.search(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text
        )
        found_day = day_match.group(1) if day_match else None

        # Determine intended action from the original task/user_message
        if re.search(r"\b(reschedule|rebook|move|change)\b", combined):
            action = "reschedule"
        elif re.search(r"\bcancel\b", combined):
            action = "cancel"
        elif re.search(r"\b(book|schedule|reserve|slot for)\b", combined):
            action = "book"
        elif re.search(r"\b(list|show|available|free)\b.*\bslot", combined):
            action = "list_slots"
        elif re.search(r"\b(my|list|show)\b.*\bappointment", combined):
            action = "list_appointments"
        else:
            action = "book"  # default for appointment agent context

        # If we found a time but no slot_id, try to resolve it from available slots
        if not slot_id and found_time:
            available = self.appointments.list_available_slots()
            candidates = available
            if found_day:
                candidates = [s for s in candidates if s.get("day", "").lower() == found_day]
            candidates = [s for s in candidates if s.get("time", "") == found_time]
            if candidates:
                slot_id = candidates[0]["id"]
                found_day = found_day or candidates[0].get("day", "")

        result = {"action": action, "message": "Recovered from free-text LLM output"}
        if slot_id:
            result["slot_id"] = slot_id
        if appt_id:
            result["appointment_id"] = appt_id
        if found_day:
            result["day"] = found_day
        if found_time:
            result["time"] = found_time

        # Only return if we have enough info to act
        if action in ("list_slots", "list_appointments"):
            return result
        if action == "book" and (slot_id or found_time):
            return result
        if action == "cancel" and appt_id:
            return result
        if action == "reschedule" and appt_id:
            if slot_id:
                result["new_slot_id"] = slot_id
            return result

        print(f"[APPOINTMENT AGENT] Recovery extracted: action={action}, slot={slot_id}, "
              f"appt={appt_id}, day={found_day}, time={found_time} — insufficient to act")
        return None

    def _format_available_data(self, slots: list, appointments: list) -> str:
        parts = []
        if slots:
            slot_strs = [
                f"  {s['id']}: {s.get('day', '')} {s.get('date', '')} at {s.get('time', '')}"
                for s in slots[:30]
            ]
            more = f"\n  ... and {len(slots) - 30} more slots." if len(slots) > 30 else ""
            parts.append("Available slots:\n" + "\n".join(slot_strs) + more)
        else:
            parts.append("Available slots: None")

        if appointments:
            appt_strs = [
                f"  {a['id']}: {a.get('day', '')} {a.get('date', '')} at {a.get('time', '')}"
                for a in appointments
            ]
            parts.append("Booked appointments:\n" + "\n".join(appt_strs))
        else:
            parts.append("Booked appointments: None")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Medical Knowledge Agent
    # ------------------------------------------------------------------

    def _run_medical_knowledge_agent(self, task: str, history_text: str, user_message: str) -> str:
        # Step 1: RAG retrieval — get raw document context directly from the retriever
        with _timed("RAG retrieval"):
            _, prompt_kwargs, mode = self.retriever.retrieve_and_build_prompt(
                user_message, history_text,
            )
            context_str = prompt_kwargs.get("context", "")

        print(f"[MEDICAL AGENT] RAG mode: {mode}")
        print(f"[MEDICAL AGENT] Context length: {len(context_str)} chars")

        # Step 2: LLM generates answer from context
        with _timed("Medical knowledge LLM answer"):
            prompt = _safe_format(
                self._medical_prompt,
                context=context_str,
                history=history_text,
                task=task,
            )
            raw = self.llm.generate(prompt)
            print(f"[MEDICAL AGENT] Raw LLM output: {raw}")

        parsed = _parse_json(raw)
        if parsed and "answer" in parsed:
            answer = parsed["answer"]
            sources = parsed.get("sources", [])
            grounded = parsed.get("grounded", False)
            print(f"[MEDICAL AGENT] Grounded: {grounded}, Sources: {sources}")
            return answer

        # If JSON parsing fails, return the raw text (cleaned)
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        cleaned = re.sub(r"^[\s\S]*?</think>", "", cleaned).strip()
        return cleaned if cleaned else "I'm sorry, I couldn't find an answer to your question."

    # ------------------------------------------------------------------
    # Messaging Agent
    # ------------------------------------------------------------------

    def _run_messaging_agent(self, task: str, history_text: str) -> str:
        doctors = self.messaging.list_doctors()
        doctors_str = "\n".join(
            f"  {d['id']}: {d['name']}" for d in doctors
        )

        # Step 1: LLM decides message details
        with _timed("Messaging agent LLM decision"):
            prompt = _safe_format(
                self._messaging_prompt,
                doctors=doctors_str,
                history=history_text,
                task=task,
            )
            raw = self.llm.generate(prompt)
            print(f"[MESSAGING AGENT] Raw LLM output: {raw}")

        parsed = _parse_json(raw)
        if not parsed:
            print("[MESSAGING AGENT] JSON parsing failed")
            return "I couldn't understand the message details. Could you please rephrase?"

        action = parsed.get("action", "clarify")
        print(f"[MESSAGING AGENT] Action: {action}")

        if action == "send_message":
            recipient = parsed.get("recipient", doctors[0]["id"] if doctors else "doc_1")
            subject = parsed.get("subject", "Patient message")
            body = parsed.get("body", task)

            with _timed("Send message execution"):
                ok = self.messaging.send_message(
                    recipient_id=recipient,
                    subject=subject,
                    body=body,
                )

            if ok:
                # Find doctor name for user-friendly response
                doc_name = next((d["name"] for d in doctors if d["id"] == recipient), recipient)
                return f"Your message has been sent to {doc_name}."
            return "I'm sorry, I couldn't send the message. Please try again."

        # clarify
        message = parsed.get("message", "Could you please provide more details about the message you want to send?")
        return message

    # ------------------------------------------------------------------
    # Chat Agent
    # ------------------------------------------------------------------

    def _run_chat_agent(self, task: str, history_text: str) -> str:
        with _timed("Chat agent LLM response"):
            prompt = _safe_format(
                self._chat_prompt,
                history=history_text,
                task=task,
            )
            raw = self.llm.generate(prompt)
            print(f"[CHAT AGENT] Raw LLM output: {raw}")

        parsed = _parse_json(raw)
        if parsed and "answer" in parsed:
            return parsed["answer"]

        # Clean up raw text
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        cleaned = re.sub(r"^[\s\S]*?</think>", "", cleaned).strip()
        return cleaned if cleaned else "Hello! How can I help you today?"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_history_text(self) -> str:
        turns = self.memory.get_recent_history(limit=self.max_history_turns)
        return "\n".join(f"{t['role']}: {t['content']}" for t in turns)

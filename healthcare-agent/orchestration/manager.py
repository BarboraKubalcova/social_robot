import json
import logging
from typing import Dict, Any, Optional

from orchestration.memory import ConversationMemory
from execution.ollama_client import OllamaClient
from execution.rag.retrieve import Retriever
from execution.actions.appointments import AppointmentManager
from execution.actions.messaging import MessageManager

class AgentManager:
    def __init__(self):
        self.memory = ConversationMemory()
        self.logger = logging.getLogger("AgentManager")
        
        # Initialize clients
        self.llm = OllamaClient()
        self.retriever = Retriever()
        self.appointments = AppointmentManager()
        self.messaging = MessageManager()

    async def process_message(self, user_message: str, user_id: str) -> Dict[str, Any]:
        """
        Main entry point for processing a user message.
        """
        self.logger.info(f"Processing message from {user_id}: {user_message}")
        
        # 1. Update Memory
        self.memory.add_turn("user", user_message)
        
        # 2. RAG + Router (Combined logic similar to minimalMCP)
        # We try to find relevant docs first.
        # If found, we use RAG mode.
        # If not, we fall back to generic chat or we check for specific action keywords.
        
        # Note: In a more complex system, we might want a dedicated router first.
        # But to match minimalMCP style where RAG is primary if context exists:
        
        history_text = "\n".join([f"{turn['role']}: {turn['content']}" for turn in self.memory.get_recent_history()])
        
        # Check for Actions first (primitive keyword matching for v1)
        # In a real system, use an LLM router or function calling
        action_response = self._check_actions(user_message)
        if action_response:
            response = action_response
            intent = "ACTION"
        else:
            # RAG / Chat flow
            # Retrieve relevant docs and build prompt
            prompt_template, prompt_kwargs, mode = self.retriever.retrieve_and_build_prompt(user_message, history_text)
            
            # Generate response
            prompt = prompt_template.format(**prompt_kwargs)
            response = self.llm.generate(prompt)
            intent = mode.upper() # RAG or LLM_ONLY

        # 3. Update Memory with Response
        self.memory.add_turn("assistant", response)

        return {
            "response": response,
            "intent": intent,
            "history": self.memory.get_recent_history()
        }

    def _check_actions(self, message: str) -> Optional[str]:
        """
        Simple keyword-based action dispatcher for demonstration.
        """
        msg_lower = message.lower()
        
        if "available slots" in msg_lower or "when can i come" in msg_lower:
            slots = self.appointments.list_available_slots()
            return f"Here are the available slots: " + ", ".join([s["time"] for s in slots])
            
        if "book" in msg_lower and "slot" in msg_lower:
            # Very naive extraction
            # In real life, extract slot_id via LLM
            return "I have booked that slot for you."
            
        if "cancel" in msg_lower:
            return "I have canceled your appointment."
            
        if "send message" in msg_lower or "email" in msg_lower:
            return "I have sent your message to the doctor."
            
        return None

from typing import List, Dict, Any
import datetime

class ConversationMemory:
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.context: Dict[str, Any] = {}

    def add_turn(self, role: str, content: str):
        """Add a message to the history."""
        timestamp = datetime.datetime.now().isoformat()
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })

    def get_recent_history(self, limit: int = 5) -> List[Dict[str, str]]:
        """Get the last N turns."""
        return self.history[-limit:]

    def set_context(self, key: str, value: Any):
        """Set a session context variable (e.g., patient_id)."""
        self.context[key] = value

    def get_context(self, key: str) -> Any:
        return self.context.get(key)

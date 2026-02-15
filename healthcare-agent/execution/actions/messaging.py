import logging

logger = logging.getLogger("Messaging")

class MessageManager:
    def list_doctors(self) -> list:
        # Single clinic context
        return [{"id": "doc_1", "name": "Dr. Smith"}, {"id": "doc_2", "name": "Dr. House"}]

    def send_message(self, recipient_id: str, subject: str, body: str) -> bool:
        """Directly send the message without draft/confirmation steps."""
        # Simplified: no draft_id, no confirmation
        print(f"Message sent to {recipient_id}. Subject: {subject}. Body: {body}")
        return True

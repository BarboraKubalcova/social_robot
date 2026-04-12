"""
Test cases for tool/route selection evaluation.

Each test case: (user_message, expected_category)
Categories: "CHAT", "RAG", "ACTION"
"""

TEST_CASES = [
    # ── CHAT: greetings, casual, emotional support ──────────────────────
    ("Hello", "CHAT"),
    ("Hi there!", "CHAT"),
    ("Good morning", "CHAT"),
    ("How are you?", "CHAT"),
    ("Thank you for your help", "CHAT"),
    ("Tell me a joke", "CHAT"),
    ("I'm feeling anxious about my visit", "CHAT"),
    ("Goodbye, see you later", "CHAT"),
    ("What is your name?", "CHAT"),
    ("Can you help me?", "CHAT"),

    # ── RAG: medical knowledge, procedures, policies ────────────────────
    ("How should I prepare for an MRI?", "RAG"),
    ("Can I wear jewelry during an MRI?", "RAG"),
    ("How does a CT scan work?", "RAG"),
    ("What should I do before a ultrasound examination?", "RAG"),
    ("Tell me about X-ray preparation", "RAG"),
    ("What is the fasting policy before taking CT?", "RAG"),
    ("How does an ultrasound procedure work?", "RAG"),
    ("Can I drink before ultrasound?", "RAG"),
    ("Is MRI examination safe?", "RAG"),
    ("Can I eat before MRI?", "RAG"),

    # ── ACTION: appointments, slots, messaging ──────────────────────────
    ("Book me an appointment for slot_1", "ACTION"),
    ("Show me available slots", "ACTION"),
    ("Cancel my appointment", "ACTION"),
    ("Reschedule my appointment to Thursday", "ACTION"),
    ("What appointments do I have?", "ACTION"),
    ("Book slot_5", "ACTION"),
    ("Cancel appt_2", "ACTION"),
    ("I need to reschedule appt_1 to slot_9", "ACTION"),
    ("List my scheduled appointments", "ACTION"),
    ("Are there any free slots on Monday?", "ACTION"),
]

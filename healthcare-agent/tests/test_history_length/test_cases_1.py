"""
Test cases for tool selection with 1-turn history (2 messages: user + assistant).

Each test case has:
  - history: list of past messages (1 user + 1 assistant)
  - user_query: the current user message to classify
  - expected_tool: the correct tool for the current query
  - description: human-readable label

8 tools × 3 conversations = 24 test cases.
The fake history deliberately discusses OTHER tools to test whether the
agent gets distracted by prior context.
"""

TEST_CASES = [
    # ── LIST_SLOTS ──────────────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "How should I prepare for an MRI scan?"},
            {"role": "assistant", "content": "For an MRI scan, please remove all metal objects and jewelry. You should wear comfortable clothing. If you have any implants, let us know beforehand."},
        ],
        "user_query": "What is available this week?",
        "expected_tool": "LIST_SLOTS",
        "description": "LIST_SLOTS after RAG history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Cancel my appointment appt_2"},
            {"role": "assistant", "content": "Your appointment appt_2 has been successfully cancelled."},
        ],
        "user_query": "Show me the free slots",
        "expected_tool": "LIST_SLOTS",
        "description": "LIST_SLOTS after CANCEL history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Send a message to my doctor saying I have a headache"},
            {"role": "assistant", "content": "Your message has been sent to your doctor."},
        ],
        "user_query": "I want to come for a visit. Can you show me when?",
        "expected_tool": "LIST_SLOTS",
        "description": "LIST_SLOTS after SEND_DOCTOR_MESSAGE history (3/3)",
    },

    # ── LIST_APPOINTMENTS ───────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Book me slot_5 please"},
            {"role": "assistant", "content": "Done! Your appointment for slot_5 on Monday at 11:30 has been booked."},
        ],
        "user_query": "What appointments do I have scheduled?",
        "expected_tool": "LIST_APPOINTMENTS",
        "description": "LIST_APPOINTMENTS after BOOK history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Is MRI safe for pregnant women?"},
            {"role": "assistant", "content": "MRI is generally considered safe during pregnancy, but you should always inform your doctor and the technician about your pregnancy beforehand."},
        ],
        "user_query": "Show my booked appointments",
        "expected_tool": "LIST_APPOINTMENTS",
        "description": "LIST_APPOINTMENTS after RAG history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "Hello! I'm doing well, thank you. How can I help you today?"},
        ],
        "user_query": "List my scheduled appointments",
        "expected_tool": "LIST_APPOINTMENTS",
        "description": "LIST_APPOINTMENTS after LLM history (3/3)",
    },

    # ── BOOK_APPOINTMENT ────────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You currently have one appointment: appt_1 on Monday at 09:30."},
        ],
        "user_query": "Book me an appointment for slot_12",
        "expected_tool": "BOOK_APPOINTMENT",
        "description": "BOOK after LIST_APPOINTMENTS history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "I want to book an appointment on Monday at 11:30"},
            {"role": "assistant", "content": "That would be slot_5 on Monday at 11:30. Shall I book it for you?"},
        ],
        "user_query": "Yes, book it",
        "expected_tool": "BOOK_APPOINTMENT",
        "description": "BOOK with pronoun 'it' after slot suggestion (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Can I eat before a CT scan?"},
            {"role": "assistant", "content": "For a CT scan, you should avoid eating for at least 4 hours before the procedure if contrast dye will be used."},
        ],
        "user_query": "Reserve slot_3 for me please",
        "expected_tool": "BOOK_APPOINTMENT",
        "description": "BOOK after RAG history (3/3)",
    },

    # ── CANCEL_APPOINTMENT ──────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Here are the available slots: Monday 07:30 (slot_1), Monday 08:30 (slot_2), Tuesday 09:30 (slot_12)..."},
        ],
        "user_query": "Cancel my appointment appt_1",
        "expected_tool": "CANCEL_APPOINTMENT",
        "description": "CANCEL after LIST_SLOTS history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "I don't think I can make my appointment on Monday"},
            {"role": "assistant", "content": "You have appt_2 scheduled for Monday at 09:30. Would you like to cancel it?"},
        ],
        "user_query": "Yes, cancel it",
        "expected_tool": "CANCEL_APPOINTMENT",
        "description": "CANCEL with pronoun 'it' after confirmation prompt (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Tell my doctor that my symptoms got worse"},
            {"role": "assistant", "content": "I've sent your message to the doctor informing them about your worsening symptoms."},
        ],
        "user_query": "Please cancel my booking appt_3",
        "expected_tool": "CANCEL_APPOINTMENT",
        "description": "CANCEL after SEND_DOCTOR_MESSAGE history (3/3)",
    },

    # ── RESCHEDULE_APPOINTMENT ──────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 on Monday at 09:30 and appt_2 on Tuesday at 14:00."},
        ],
        "user_query": "Reschedule appt_1 to slot_20",
        "expected_tool": "RESCHEDULE_APPOINTMENT",
        "description": "RESCHEDULE after LIST_APPOINTMENTS history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "I need to change my Tuesday appointment to a later time"},
            {"role": "assistant", "content": "Your appointment appt_2 is on Tuesday at 14:00. I can move it to slot_15 at 13:00. Would you like me to reschedule it?"},
        ],
        "user_query": "Yes, move it there",
        "expected_tool": "RESCHEDULE_APPOINTMENT",
        "description": "RESCHEDULE with 'it there' after suggestion (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Cancel my appointment appt_3"},
            {"role": "assistant", "content": "Your appointment appt_3 has been cancelled."},
        ],
        "user_query": "I want to change appt_1 to slot_9",
        "expected_tool": "RESCHEDULE_APPOINTMENT",
        "description": "RESCHEDULE after CANCEL history (3/3)",
    },

    # ── SEND_DOCTOR_MESSAGE ─────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Book me slot_5"},
            {"role": "assistant", "content": "Your appointment for slot_5 (Monday 11:30) has been booked."},
        ],
        "user_query": "Send a message to my doctor saying I feel dizzy",
        "expected_tool": "SEND_DOCTOR_MESSAGE",
        "description": "SEND_DOCTOR_MESSAGE after BOOK history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "What is the fasting policy before a CT?"},
            {"role": "assistant", "content": "You should fast for at least 4 hours before a CT scan if contrast dye is used. Water is usually allowed."},
        ],
        "user_query": "Tell my doctor about my test results",
        "expected_tool": "SEND_DOCTOR_MESSAGE",
        "description": "SEND_DOCTOR_MESSAGE after RAG history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Show me my appointments"},
            {"role": "assistant", "content": "You have appt_1 on Monday at 09:30."},
        ],
        "user_query": "I want to contact my doctor about my medication",
        "expected_tool": "SEND_DOCTOR_MESSAGE",
        "description": "SEND_DOCTOR_MESSAGE after LIST_APPOINTMENTS history (3/3)",
    },

    # ── RAG ─────────────────────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Book me slot_12"},
            {"role": "assistant", "content": "Your appointment for slot_12 (Tuesday 09:30) is confirmed."},
        ],
        "user_query": "How should I prepare for an MRI?",
        "expected_tool": "RAG",
        "description": "RAG after BOOK history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "What should I know about CT scans?"},
            {"role": "assistant", "content": "CT scans use X-rays to create cross-sectional images. There are preparation guidelines you should follow. Would you like to know more?"},
        ],
        "user_query": "Yes, tell me more about that",
        "expected_tool": "RAG",
        "description": "RAG with pronoun 'that' referencing CT info (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Reschedule appt_2 to slot_15"},
            {"role": "assistant", "content": "Done! Your appointment appt_2 has been moved to slot_15 (Tuesday 13:00)."},
        ],
        "user_query": "What should I do before an ultrasound examination?",
        "expected_tool": "RAG",
        "description": "RAG after RESCHEDULE history (3/3)",
    },

    # ── LLM ─────────────────────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Here are the available slots: slot_1 Monday 07:30, slot_2 Monday 08:30, slot_10 Tuesday 07:30..."},
        ],
        "user_query": "Thank you for your help",
        "expected_tool": "LLM",
        "description": "LLM after LIST_SLOTS history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Cancel my appointment appt_2"},
            {"role": "assistant", "content": "Your appointment appt_2 has been cancelled successfully."},
        ],
        "user_query": "Hello, how are you doing?",
        "expected_tool": "LLM",
        "description": "LLM after CANCEL history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "How does an MRI work?"},
            {"role": "assistant", "content": "MRI uses strong magnetic fields and radio waves to produce detailed images of organs and tissues in your body."},
        ],
        "user_query": "Tell me a joke",
        "expected_tool": "LLM",
        "description": "LLM after RAG history (3/3)",
    },
]

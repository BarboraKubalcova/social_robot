"""
Test cases for tool selection with 2-turn history (4 messages: 2×user + 2×assistant).

Each test case has:
  - history: list of past messages (2 user + 2 assistant turns)
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
            {"role": "assistant", "content": "For an MRI scan, remove all metal objects and jewelry. Wear comfortable clothing and inform staff about any implants."},
            {"role": "user", "content": "Can I eat before the MRI?"},
            {"role": "assistant", "content": "Generally yes, unless your doctor has given specific fasting instructions. Check with your clinic for details."},
        ],
        "user_query": "What appointment slots are available this week?",
        "expected_tool": "LIST_SLOTS",
        "description": "LIST_SLOTS after 2-turn RAG history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Cancel my appointment appt_2"},
            {"role": "assistant", "content": "Your appointment appt_2 has been successfully cancelled."},
            {"role": "user", "content": "Thank you for cancelling it"},
            {"role": "assistant", "content": "You're welcome! Is there anything else I can help you with?"},
        ],
        "user_query": "Show me the free time slots",
        "expected_tool": "LIST_SLOTS",
        "description": "LIST_SLOTS after CANCEL+LLM history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Send a message to my doctor saying I have a headache"},
            {"role": "assistant", "content": "Your message has been sent to your doctor."},
            {"role": "user", "content": "Book me slot_3"},
            {"role": "assistant", "content": "Your appointment for slot_3 (Monday 09:30) has been booked."},
        ],
        "user_query": "Are there any available appointment times left?",
        "expected_tool": "LIST_SLOTS",
        "description": "LIST_SLOTS after MSG+BOOK history (3/3)",
    },

    # ── LIST_APPOINTMENTS ───────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "What times can I come in this week?"},
            {"role": "assistant", "content": "Here are the available slots: slot_1 Monday 07:30, slot_2 Monday 08:30, slot_10 Tuesday 07:30..."},
            {"role": "user", "content": "Book me slot_5 please"},
            {"role": "assistant", "content": "Done! Your appointment for slot_5 (Monday 11:30) has been booked."},
        ],
        "user_query": "What appointments do I have scheduled?",
        "expected_tool": "LIST_APPOINTMENTS",
        "description": "LIST_APPOINTMENTS after LIST_SLOTS+BOOK history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Is MRI safe for pregnant women?"},
            {"role": "assistant", "content": "MRI is generally considered safe during pregnancy, but always inform your doctor and technician beforehand."},
            {"role": "user", "content": "What about CT scans?"},
            {"role": "assistant", "content": "CT scans use radiation and are generally avoided during pregnancy unless absolutely necessary."},
        ],
        "user_query": "Show my booked appointments",
        "expected_tool": "LIST_APPOINTMENTS",
        "description": "LIST_APPOINTMENTS after 2-turn RAG history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "Tell my doctor I need a refill on my prescription"},
            {"role": "assistant", "content": "I've sent your message about the prescription refill to your doctor."},
        ],
        "user_query": "Can you list my scheduled appointments?",
        "expected_tool": "LIST_APPOINTMENTS",
        "description": "LIST_APPOINTMENTS after LLM+MSG history (3/3)",
    },

    # ── BOOK_APPOINTMENT ────────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You currently have one appointment: appt_1 on Monday at 09:30."},
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_2 Mon 08:30, slot_5 Mon 11:30, slot_10 Tue 07:30, slot_12 Tue 09:30..."},
        ],
        "user_query": "Book me an appointment for slot_12",
        "expected_tool": "BOOK_APPOINTMENT",
        "description": "BOOK after LIST_APPTS+LIST_SLOTS history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_5 Mon 11:30, slot_12 Tue 09:30, slot_15 Tue 13:00, slot_20 Wed 08:30..."},
            {"role": "user", "content": "I like the Tuesday afternoon one"},
            {"role": "assistant", "content": "That's slot_15 on Tuesday at 13:00. Would you like me to book that for you?"},
        ],
        "user_query": "Yes please",
        "expected_tool": "BOOK_APPOINTMENT",
        "description": "BOOK with 'that one' after slot confirmation (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Cancel my appointment appt_3"},
            {"role": "assistant", "content": "Your appointment appt_3 has been cancelled."},
            {"role": "user", "content": "Can I eat before a CT scan?"},
            {"role": "assistant", "content": "You should avoid eating for at least 4 hours before a CT scan if contrast dye will be used."},
        ],
        "user_query": "Reserve slot_3 for me please",
        "expected_tool": "BOOK_APPOINTMENT",
        "description": "BOOK after CANCEL+RAG history (3/3)",
    },

    # ── CANCEL_APPOINTMENT ──────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_2 Mon 08:30, slot_10 Tue 07:30..."},
            {"role": "user", "content": "Book me slot_10"},
            {"role": "assistant", "content": "Your appointment for slot_10 (Tuesday 07:30) has been booked."},
        ],
        "user_query": "Cancel my appointment appt_1",
        "expected_tool": "CANCEL_APPOINTMENT",
        "description": "CANCEL after LIST_SLOTS+BOOK history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "How should I prepare for a CT scan?"},
            {"role": "assistant", "content": "For a CT scan, you may need to fast for 4 hours. Remove any metal jewelry and wear comfortable clothes."},
            {"role": "user", "content": "Is the contrast dye safe?"},
            {"role": "assistant", "content": "Yes, contrast dye is generally safe. Inform your doctor if you have allergies or kidney problems."},
        ],
        "user_query": "I need to cancel appointment appt_2",
        "expected_tool": "CANCEL_APPOINTMENT",
        "description": "CANCEL after 2-turn RAG history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_2 Tue 14:00, appt_3 Wed 08:30."},
            {"role": "user", "content": "I can't make the Wednesday one"},
            {"role": "assistant", "content": "That's appt_3 on Wednesday at 08:30. Do you want me to cancel it?"},
        ],
        "user_query": "Yes, go aheadt",
        "expected_tool": "CANCEL_APPOINTMENT",
        "description": "CANCEL with 'it' after confirmation prompt (3/3)",
    },

    # ── RESCHEDULE_APPOINTMENT ──────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "I need to move my Monday appointment to later in the week"},
            {"role": "assistant", "content": "Your Monday appointment is appt_1 at 09:30. I can move it to Wednesday. slot_20 at 08:30 is free. Shall I reschedule?"},
            {"role": "user", "content": "What other slots do you have on Wednesday?"},
            {"role": "assistant", "content": "On Wednesday I have: slot_19 at 07:30, slot_20 at 08:30, slot_21 at 09:30, slot_22 at 10:30. Would you like any of these?"},
        ],
        "user_query": "Yes, the 09:30 one please",
        "expected_tool": "RESCHEDULE_APPOINTMENT",
        "description": "RESCHEDULE with 'the 09:30 one' after options (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "How does an ultrasound work?"},
            {"role": "assistant", "content": "An ultrasound uses sound waves to image internal organs. It's painless and non-invasive."},
            {"role": "user", "content": "Can I bring someone with me?"},
            {"role": "assistant", "content": "Yes, you can usually bring one companion for support during the procedure."},
        ],
        "user_query": "Move my appointment appt_2 to slot_15",
        "expected_tool": "RESCHEDULE_APPOINTMENT",
        "description": "RESCHEDULE after 2-turn RAG history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Cancel my appointment appt_3"},
            {"role": "assistant", "content": "Your appointment appt_3 has been cancelled."},
            {"role": "user", "content": "Send a message to my doctor that I'm feeling better"},
            {"role": "assistant", "content": "I've sent your message to the doctor."},
        ],
        "user_query": "I want to change appt_1 to slot_9",
        "expected_tool": "RESCHEDULE_APPOINTMENT",
        "description": "RESCHEDULE after CANCEL+MSG history (3/3)",
    },

    # ── SEND_DOCTOR_MESSAGE ─────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_5 Mon 11:30, slot_10 Tue 07:30..."},
            {"role": "user", "content": "Book me slot_5"},
            {"role": "assistant", "content": "Your appointment for slot_5 (Monday 11:30) has been booked."},
        ],
        "user_query": "Send a message to my doctor saying I feel dizzy",
        "expected_tool": "SEND_DOCTOR_MESSAGE",
        "description": "SEND_DOCTOR_MESSAGE after LIST_SLOTS+BOOK history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "I've been having severe headaches lately"},
            {"role": "assistant", "content": "I'm sorry to hear that. I can relay this to your doctor. Would you like me to send them a message about it?"},
            {"role": "user", "content": "What would you say in the message?"},
            {"role": "assistant", "content": "I would inform your doctor that you've been experiencing severe headaches and request their guidance. Shall I send it?"},
        ],
        "user_query": "Yes, send it",
        "expected_tool": "SEND_DOCTOR_MESSAGE",
        "description": "SEND_DOCTOR_MESSAGE with 'it' after msg draft (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Cancel my appointment appt_1"},
            {"role": "assistant", "content": "Your appointment appt_1 has been cancelled."},
            {"role": "user", "content": "Show my remaining appointments"},
            {"role": "assistant", "content": "You still have: appt_2 on Tuesday at 14:00."},
        ],
        "user_query": "I want to contact my doctor about my medication side effects",
        "expected_tool": "SEND_DOCTOR_MESSAGE",
        "description": "SEND_DOCTOR_MESSAGE after CANCEL+LIST_APPTS history (3/3)",
    },

    # ── RAG ─────────────────────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Book me slot_12"},
            {"role": "assistant", "content": "Your appointment for slot_12 (Tuesday 09:30) is confirmed."},
            {"role": "user", "content": "What appointments do I have now?"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_4 Tue 09:30."},
        ],
        "user_query": "How should I prepare for an MRI?",
        "expected_tool": "RAG",
        "description": "RAG after BOOK+LIST_APPTS history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Cancel appt_1"},
            {"role": "assistant", "content": "Your appointment appt_1 has been cancelled."},
            {"role": "user", "content": "Reschedule appt_2 to slot_20"},
            {"role": "assistant", "content": "Done! appt_2 has been moved to slot_20 (Wednesday 08:30)."},
        ],
        "user_query": "Can I wear jewelry during a CT scan?",
        "expected_tool": "RAG",
        "description": "RAG after CANCEL+RESCHEDULE history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Send my doctor a message about chest pain"},
            {"role": "assistant", "content": "I've sent your message about chest pain to your doctor."},
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_5 Mon 11:30, slot_19 Wed 07:30..."},
        ],
        "user_query": "What should I do before an ultrasound examination?",
        "expected_tool": "RAG",
        "description": "RAG after MSG+LIST_SLOTS history (3/3)",
    },

    # ── LLM ─────────────────────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_2 Mon 08:30, slot_10 Tue 07:30..."},
            {"role": "user", "content": "Book me slot_2"},
            {"role": "assistant", "content": "Your appointment for slot_2 (Monday 08:30) has been booked."},
        ],
        "user_query": "Thank you so much for your help!",
        "expected_tool": "LLM",
        "description": "LLM after LIST_SLOTS+BOOK history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Cancel my appointment appt_2"},
            {"role": "assistant", "content": "Your appointment appt_2 has been cancelled."},
            {"role": "user", "content": "How should I prepare for an MRI?"},
            {"role": "assistant", "content": "Remove all metal objects, wear comfortable clothing, and inform staff about any implants."},
        ],
        "user_query": "Good morning, how are you?",
        "expected_tool": "LLM",
        "description": "LLM after CANCEL+RAG history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Reschedule appt_1 to slot_15"},
            {"role": "assistant", "content": "Done! appt_1 moved to slot_15 (Tuesday 13:00)."},
            {"role": "user", "content": "Tell my doctor I'll bring my test results"},
            {"role": "assistant", "content": "I've sent your message to the doctor about your test results."},
        ],
        "user_query": "What is your name?",
        "expected_tool": "LLM",
        "description": "LLM after RESCHEDULE+MSG history (3/3)",
    },
]

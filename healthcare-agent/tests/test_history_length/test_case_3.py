"""
Test cases for tool selection with 3-turn history (6 messages: 3×user + 3×assistant).

Each test case has:
  - history: list of past messages (3 user + 3 assistant turns)
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
            {"role": "assistant", "content": "Generally yes, unless your doctor has given specific fasting instructions."},
            {"role": "user", "content": "Thank you for the information"},
            {"role": "assistant", "content": "You're welcome! Let me know if you have any other questions."},
        ],
        "user_query": "What appointment slots are available?",
        "expected_tool": "LIST_SLOTS",
        "description": "LIST_SLOTS after RAG+RAG+LLM history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Cancel my appointment appt_2"},
            {"role": "assistant", "content": "Your appointment appt_2 has been successfully cancelled."},
            {"role": "user", "content": "Send a message to my doctor about my recovery"},
            {"role": "assistant", "content": "Your message has been sent to your doctor."},
            {"role": "user", "content": "Book me slot_12 please"},
            {"role": "assistant", "content": "Your appointment for slot_12 (Tuesday 09:30) has been booked."},
        ],
        "user_query": "Show me the free slots",
        "expected_tool": "LIST_SLOTS",
        "description": "LIST_SLOTS after CANCEL+MSG+BOOK history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 on Monday at 09:30 and appt_2 on Tuesday at 14:00."},
            {"role": "user", "content": "Reschedule appt_1 to slot_6"},
            {"role": "assistant", "content": "Done! appt_1 has been rescheduled to slot_6 (Monday 13:00)."},
        ],
        "user_query": "Are there any open appointment times?",
        "expected_tool": "LIST_SLOTS",
        "description": "LIST_SLOTS after LLM+LIST_APPTS+RESCHEDULE history (3/3)",
    },

    # ── LIST_APPOINTMENTS ───────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_5 Mon 11:30, slot_10 Tue 07:30, slot_12 Tue 09:30..."},
            {"role": "user", "content": "Book me slot_5"},
            {"role": "assistant", "content": "Your appointment for slot_5 (Monday 11:30) has been booked."},
            {"role": "user", "content": "How should I prepare for a CT scan?"},
            {"role": "assistant", "content": "Fast for 4 hours before a CT scan if contrast dye is used. Remove metal jewelry."},
        ],
        "user_query": "What appointments do I have scheduled?",
        "expected_tool": "LIST_APPOINTMENTS",
        "description": "LIST_APPOINTMENTS after LIST_SLOTS+BOOK+RAG history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Is MRI safe during pregnancy?"},
            {"role": "assistant", "content": "MRI is generally considered safe during pregnancy, but always inform your doctor."},
            {"role": "user", "content": "Send my doctor a note about my pregnancy"},
            {"role": "assistant", "content": "I've sent your message to the doctor. Would you also like to check your upcoming appointments?"},
            {"role": "user", "content": "How many do I have?"},
            {"role": "assistant", "content": "Let me look that up for you. Would you like me to show your scheduled appointments?"},
        ],
        "user_query": "Yes, show me those",
        "expected_tool": "LIST_APPOINTMENTS",
        "description": "LIST_APPOINTMENTS with 'those' after prompt (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Cancel my appointment appt_3"},
            {"role": "assistant", "content": "Your appointment appt_3 has been cancelled."},
            {"role": "user", "content": "Book me slot_19"},
            {"role": "assistant", "content": "Your appointment for slot_19 (Wednesday 07:30) has been booked."},
            {"role": "user", "content": "Can I wear jewelry during an ultrasound?"},
            {"role": "assistant", "content": "You should remove jewelry near the area being examined, as it can interfere with the procedure."},
        ],
        "user_query": "List my scheduled appointments",
        "expected_tool": "LIST_APPOINTMENTS",
        "description": "LIST_APPOINTMENTS after CANCEL+BOOK+RAG history (3/3)",
    },

    # ── BOOK_APPOINTMENT ────────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have appt_1 on Monday at 09:30."},
            {"role": "user", "content": "Cancel appt_1"},
            {"role": "assistant", "content": "Your appointment appt_1 has been cancelled. Would you like to book a new one?"},
            {"role": "user", "content": "Yes, show me what's available on Tuesday"},
            {"role": "assistant", "content": "On Tuesday I have: slot_10 at 07:30, slot_11 at 08:30, slot_12 at 09:30. Would you like any of these?"},
        ],
        "user_query": "I'll take the 09:30 one",
        "expected_tool": "BOOK_APPOINTMENT",
        "description": "BOOK with 'the 09:30 one' after options (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "How does an ultrasound work?"},
            {"role": "assistant", "content": "An ultrasound uses high-frequency sound waves to create images of internal organs."},
            {"role": "user", "content": "Tell my doctor I need an ultrasound referral"},
            {"role": "assistant", "content": "I've sent your message to the doctor requesting an ultrasound referral."},
            {"role": "user", "content": "What times are available this week?"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_10 Tue 07:30, slot_19 Wed 07:30..."},
        ],
        "user_query": "I'd like to Tue 7:30",
        "expected_tool": "BOOK_APPOINTMENT",
        "description": "BOOK after RAG+MSG+LIST_SLOTS history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Reschedule appt_2 to slot_6"},
            {"role": "assistant", "content": "Done! appt_2 rescheduled to slot_6 (Monday 13:00)."},
            {"role": "user", "content": "Can I eat before a CT scan?"},
            {"role": "assistant", "content": "Avoid eating for 4 hours before a CT if contrast dye is used."},
            {"role": "user", "content": "Hello, I need another appointment"},
            {"role": "assistant", "content": "Of course! Would you like to see available slots or book a specific one?"},
        ],
        "user_query": "Reserve Wednesday 07:30 for me please",
        "expected_tool": "BOOK_APPOINTMENT",
        "description": "BOOK after RESCHEDULE+RAG+LLM history (3/3)",
    },

    # ── CANCEL_APPOINTMENT ──────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_5 Mon 11:30, slot_10 Tue 07:30..."},
            {"role": "user", "content": "Book me slot_5"},
            {"role": "assistant", "content": "Your appointment for slot_5 (Monday 11:30) has been booked."},
            {"role": "user", "content": "How should I prepare for it?"},
            {"role": "assistant", "content": "For a general appointment, arrive 10 minutes early and bring your ID and insurance card."},
        ],
        "user_query": "Cancel my appointment appt_1",
        "expected_tool": "CANCEL_APPOINTMENT",
        "description": "CANCEL after LIST_SLOTS+BOOK+RAG history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Tell my doctor about my headaches"},
            {"role": "assistant", "content": "I've sent your message about headaches to the doctor."},
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_2 Tue 14:00, appt_3 Wed 08:30."},
            {"role": "user", "content": "I'm not going to make the Wednesday one"},
            {"role": "assistant", "content": "That's appt_3 on Wednesday at 08:30. Would you like me to cancel it or reschedule it?"},
        ],
        "user_query": "Just cancel it",
        "expected_tool": "CANCEL_APPOINTMENT",
        "description": "CANCEL with 'it' after disambiguation prompt (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "Can I wear metal during an MRI?"},
            {"role": "assistant", "content": "No, all metal objects must be removed before an MRI scan."},
            {"role": "user", "content": "Good to know. Also show me my appointments."},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_2 Tue 14:00."},
        ],
        "user_query": "Please cancel my booking appt_2",
        "expected_tool": "CANCEL_APPOINTMENT",
        "description": "CANCEL after LLM+RAG+LIST_APPTS history (3/3)",
    },

    # ── RESCHEDULE_APPOINTMENT ──────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_2 Tue 14:00."},
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_5 Mon 11:30, slot_12 Tue 09:30, slot_20 Wed 08:30..."},
            {"role": "user", "content": "I'm feeling anxious about my visit"},
            {"role": "assistant", "content": "I understand your anxiety. Take deep breaths and know that our staff will take good care of you."},
        ],
        "user_query": "Reschedule appt_1 to slot_20",
        "expected_tool": "RESCHEDULE_APPOINTMENT",
        "description": "RESCHEDULE after LIST_APPTS+LIST_SLOTS+LLM history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Book me slot_3"},
            {"role": "assistant", "content": "Your appointment for slot_3 (Monday 09:30) has been booked."},
            {"role": "user", "content": "How does an ultrasound work?"},
            {"role": "assistant", "content": "An ultrasound uses sound waves to image internal organs. It's painless and non-invasive."},
            {"role": "user", "content": "Cancel appt_3"},
            {"role": "assistant", "content": "Your appointment appt_3 has been cancelled."},
        ],
        "user_query": "Move my appointment appt_2 to slot_15",
        "expected_tool": "RESCHEDULE_APPOINTMENT",
        "description": "RESCHEDULE after BOOK+RAG+CANCEL history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Show me my appointments"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_2 Wed 08:30."},
            {"role": "user", "content": "The Wednesday one doesn't work for me anymore"},
            {"role": "assistant", "content": "I understand. Would you like to reschedule appt_2 to a different time? Monday slot_9 at 16:00 is available."},
            {"role": "user", "content": "Is there anything earlier on Monday?"},
            {"role": "assistant", "content": "Yes, slot_5 at 11:30 and slot_6 at 13:00 are available on Monday."},
        ],
        "user_query": "The 11:30 one works, reschedule it there",
        "expected_tool": "RESCHEDULE_APPOINTMENT",
        "description": "RESCHEDULE with 'it there' after time negotiation (3/3)",
    },

    # ── SEND_DOCTOR_MESSAGE ─────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_5 Mon 11:30, slot_10 Tue 07:30..."},
            {"role": "user", "content": "Book me slot_5"},
            {"role": "assistant", "content": "Your appointment for slot_5 (Monday 11:30) has been booked."},
            {"role": "user", "content": "How should I prepare for the visit?"},
            {"role": "assistant", "content": "Arrive 10 minutes early, bring your ID and insurance card. Follow any specific instructions from your doctor."},
        ],
        "user_query": "Send a message to my doctor saying I feel dizzy",
        "expected_tool": "SEND_DOCTOR_MESSAGE",
        "description": "SEND_DOCTOR_MESSAGE after LIST_SLOTS+BOOK+RAG history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30."},
            {"role": "user", "content": "Reschedule appt_1 to slot_15"},
            {"role": "assistant", "content": "Done! appt_1 moved to slot_15 (Tuesday 13:00)."},
            {"role": "user", "content": "Cancel appt_1"},
            {"role": "assistant", "content": "Your appointment appt_1 has been cancelled."},
        ],
        "user_query": "Tell my doctor about my allergy to penicillin",
        "expected_tool": "SEND_DOCTOR_MESSAGE",
        "description": "SEND_DOCTOR_MESSAGE after LIST_APPTS+RESCHEDULE+CANCEL history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Can I eat before an MRI?"},
            {"role": "assistant", "content": "Generally yes, unless your doctor has given specific fasting instructions."},
            {"role": "user", "content": "Hello, I have a question"},
            {"role": "assistant", "content": "Of course! What would you like to know?"},
            {"role": "user", "content": "Show me my appointments"},
            {"role": "assistant", "content": "You have: appt_2 on Tuesday at 14:00."},
        ],
        "user_query": "I want to contact my doctor about changing my medication",
        "expected_tool": "SEND_DOCTOR_MESSAGE",
        "description": "SEND_DOCTOR_MESSAGE after RAG+LLM+LIST_APPTS history (3/3)",
    },

    # ── RAG ─────────────────────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Book me slot_12"},
            {"role": "assistant", "content": "Your appointment for slot_12 (Tuesday 09:30) is confirmed."},
            {"role": "user", "content": "Show me my appointments"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_4 Tue 09:30."},
            {"role": "user", "content": "Cancel appt_1"},
            {"role": "assistant", "content": "Your appointment appt_1 has been cancelled."},
        ],
        "user_query": "How should I prepare for an MRI?",
        "expected_tool": "RAG",
        "description": "RAG after BOOK+LIST_APPTS+CANCEL history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Hello, good morning"},
            {"role": "assistant", "content": "Good morning! How can I help you today?"},
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_5 Mon 11:30, slot_10 Tue 07:30..."},
            {"role": "user", "content": "Book me slot_1"},
            {"role": "assistant", "content": "Your appointment for slot_1 (Monday 07:30) has been booked."},
        ],
        "user_query": "Can I wear jewelry during a CT scan?",
        "expected_tool": "RAG",
        "description": "RAG after LLM+LIST_SLOTS+BOOK history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Reschedule appt_2 to slot_15"},
            {"role": "assistant", "content": "Done! appt_2 moved to slot_15 (Tuesday 13:00)."},
            {"role": "user", "content": "Tell my doctor I'll be coming in earlier"},
            {"role": "assistant", "content": "I've sent your message to the doctor."},
            {"role": "user", "content": "Thank you for your help"},
            {"role": "assistant", "content": "You're welcome! Let me know if you need anything else."},
        ],
        "user_query": "What should I do before an ultrasound?",
        "expected_tool": "RAG",
        "description": "RAG after RESCHEDULE+MSG+LLM history (3/3)",
    },

    # ── LLM ─────────────────────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_2 Mon 08:30, slot_10 Tue 07:30..."},
            {"role": "user", "content": "Book me slot_2"},
            {"role": "assistant", "content": "Your appointment for slot_2 (Monday 08:30) has been booked."},
            {"role": "user", "content": "How should I prepare for the visit?"},
            {"role": "assistant", "content": "Arrive 10 minutes early, bring your ID and insurance card."},
        ],
        "user_query": "Thank you so much for everything!",
        "expected_tool": "LLM",
        "description": "LLM after LIST_SLOTS+BOOK+RAG history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Cancel my appointment appt_2"},
            {"role": "assistant", "content": "Your appointment appt_2 has been cancelled."},
            {"role": "user", "content": "Send a message to my doctor about my leg pain"},
            {"role": "assistant", "content": "I've sent your message to the doctor about your leg pain."},
            {"role": "user", "content": "What is the fasting policy before a CT?"},
            {"role": "assistant", "content": "Fast for at least 4 hours before a CT scan if contrast dye is used."},
        ],
        "user_query": "Good morning, how are you doing?",
        "expected_tool": "LLM",
        "description": "LLM after CANCEL+MSG+RAG history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_2 Tue 14:00."},
            {"role": "user", "content": "Reschedule appt_1 to slot_6"},
            {"role": "assistant", "content": "Done! appt_1 moved to slot_6 (Monday 13:00)."},
            {"role": "user", "content": "Book me slot_12"},
            {"role": "assistant", "content": "Your appointment for slot_12 (Tuesday 09:30) has been booked."},
        ],
        "user_query": "Tell me a joke",
        "expected_tool": "LLM",
        "description": "LLM after LIST_APPTS+RESCHEDULE+BOOK history (3/3)",
    },
]

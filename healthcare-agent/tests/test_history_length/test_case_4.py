"""
Test cases for tool selection with 4-turn history (8 messages: 4×user + 4×assistant).

Each test case has:
  - history: list of past messages (4 user + 4 assistant turns)
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
            {"role": "assistant", "content": "For an MRI scan, remove all metal objects and jewelry. Wear comfortable clothing."},
            {"role": "user", "content": "Can I eat before the MRI?"},
            {"role": "assistant", "content": "Generally yes, unless your doctor has given specific fasting instructions."},
            {"role": "user", "content": "Book me slot_5"},
            {"role": "assistant", "content": "Your appointment for slot_5 (Monday 11:30) has been booked."},
            {"role": "user", "content": "Cancel my appointment appt_2"},
            {"role": "assistant", "content": "Your appointment appt_2 has been cancelled."},
        ],
        "user_query": "What appointment slots are available?",
        "expected_tool": "LIST_SLOTS",
        "description": "LIST_SLOTS after RAG+RAG+BOOK+CANCEL history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "Send my doctor a message about my back pain"},
            {"role": "assistant", "content": "Your message about back pain has been sent to your doctor."},
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_3 Wed 08:30."},
            {"role": "user", "content": "Reschedule appt_3 to slot_22"},
            {"role": "assistant", "content": "Done! appt_3 rescheduled to slot_22 (Wednesday 10:30)."},
        ],
        "user_query": "Show me the free time slots",
        "expected_tool": "LIST_SLOTS",
        "description": "LIST_SLOTS after LLM+MSG+LIST_APPTS+RESCHEDULE history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Cancel my appointment appt_1"},
            {"role": "assistant", "content": "Your appointment appt_1 has been cancelled."},
            {"role": "user", "content": "Is MRI safe for people with pacemakers?"},
            {"role": "assistant", "content": "MRI can be dangerous for people with certain pacemakers. Always inform your doctor about any implants."},
            {"role": "user", "content": "Thank you for the information"},
            {"role": "assistant", "content": "You're welcome! Is there anything else I can help with?"},
            {"role": "user", "content": "Tell my doctor about my implant concern"},
            {"role": "assistant", "content": "I've sent your message about the implant concern to your doctor."},
        ],
        "user_query": "Are there any available appointment times this week?",
        "expected_tool": "LIST_SLOTS",
        "description": "LIST_SLOTS after CANCEL+RAG+LLM+MSG history (3/3)",
    },

    # ── LIST_APPOINTMENTS ───────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_5 Mon 11:30, slot_10 Tue 07:30, slot_19 Wed 07:30..."},
            {"role": "user", "content": "Book me slot_5"},
            {"role": "assistant", "content": "Your appointment for slot_5 (Monday 11:30) has been booked."},
            {"role": "user", "content": "How should I prepare for a CT scan?"},
            {"role": "assistant", "content": "Fast for 4 hours if contrast dye is used. Remove metal jewelry and wear comfortable clothes."},
            {"role": "user", "content": "Send my doctor a note about my allergy"},
            {"role": "assistant", "content": "I've sent your allergy information to the doctor."},
        ],
        "user_query": "What appointments do I have?",
        "expected_tool": "LIST_APPOINTMENTS",
        "description": "LIST_APPOINTMENTS after LIST_SLOTS+BOOK+RAG+MSG history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Good morning"},
            {"role": "assistant", "content": "Good morning! How can I assist you today?"},
            {"role": "user", "content": "Can I drink water before an ultrasound?"},
            {"role": "assistant", "content": "For some ultrasounds you may need a full bladder, so drinking water may be recommended. Check with your clinic."},
            {"role": "user", "content": "Reschedule appt_1 to slot_12"},
            {"role": "assistant", "content": "Done! appt_1 moved to slot_12 (Tuesday 09:30)."},
            {"role": "user", "content": "Cancel appt_3"},
            {"role": "assistant", "content": "Your appointment appt_3 has been cancelled."},
        ],
        "user_query": "Show my booked appointments",
        "expected_tool": "LIST_APPOINTMENTS",
        "description": "LIST_APPOINTMENTS after LLM+RAG+RESCHEDULE+CANCEL history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Book me slot_19"},
            {"role": "assistant", "content": "Your appointment for slot_19 (Wednesday 07:30) has been booked."},
            {"role": "user", "content": "Tell my doctor I need a follow-up"},
            {"role": "assistant", "content": "I've sent your follow-up request to the doctor."},
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_2 Mon 08:30, slot_10 Tue 07:30..."},
            {"role": "user", "content": "How does a CT scan work?"},
            {"role": "assistant", "content": "A CT scan uses X-rays taken from multiple angles to create cross-sectional images of your body."},
        ],
        "user_query": "List my scheduled appointments",
        "expected_tool": "LIST_APPOINTMENTS",
        "description": "LIST_APPOINTMENTS after BOOK+MSG+LIST_SLOTS+RAG history (3/3)",
    },

    # ── BOOK_APPOINTMENT ────────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have appt_1 on Monday at 09:30."},
            {"role": "user", "content": "Cancel appt_1"},
            {"role": "assistant", "content": "Your appointment appt_1 has been cancelled."},
            {"role": "user", "content": "How should I prepare for an ultrasound?"},
            {"role": "assistant", "content": "Your doctor may ask you to drink water beforehand for a full bladder. Wear loose clothing."},
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_3 Mon 09:30, slot_5 Mon 11:30, slot_12 Tue 09:30, slot_20 Wed 08:30..."},
        ],
        "user_query": "Book me an appointment for slot_12",
        "expected_tool": "BOOK_APPOINTMENT",
        "description": "BOOK after LIST_APPTS+CANCEL+RAG+LIST_SLOTS history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Hello, I need to make a new appointment"},
            {"role": "assistant", "content": "Of course! Would you like to see available slots?"},
            {"role": "user", "content": "Yes, what do you have on Wednesday?"},
            {"role": "assistant", "content": "On Wednesday: slot_19 at 07:30, slot_20 at 08:30, slot_21 at 09:30, slot_22 at 10:30."},
            {"role": "user", "content": "How about the afternoon?"},
            {"role": "assistant", "content": "Wednesday afternoon: slot_24 at 13:00, slot_25 at 14:00, slot_26 at 15:00, slot_27 at 16:00."},
            {"role": "user", "content": "The 14:00 one looks good"},
            {"role": "assistant", "content": "That's slot_25 on Wednesday at 14:00. Shall I book that for you?"},
        ],
        "user_query": "Yes, go ahead",
        "expected_tool": "BOOK_APPOINTMENT",
        "description": "BOOK with 'go ahead' after slot negotiation (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "What is the fasting policy for CT scans?"},
            {"role": "assistant", "content": "Fast for at least 4 hours before a CT if contrast dye is used. Water is usually allowed."},
            {"role": "user", "content": "Cancel my appointment appt_3"},
            {"role": "assistant", "content": "Your appointment appt_3 has been cancelled."},
            {"role": "user", "content": "What appointments do I have left?"},
            {"role": "assistant", "content": "You have: appt_2 on Tuesday at 14:00."},
            {"role": "user", "content": "Tell my doctor about my upcoming surgery"},
            {"role": "assistant", "content": "I've sent your message about the upcoming surgery to your doctor."},
        ],
        "user_query": "Reserve slot_3 for me please",
        "expected_tool": "BOOK_APPOINTMENT",
        "description": "BOOK after RAG+CANCEL+LIST_APPTS+MSG history (3/3)",
    },

    # ── CANCEL_APPOINTMENT ──────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_2 Tue 14:00, appt_3 Wed 08:30."},
            {"role": "user", "content": "I need to cancel one of them"},
            {"role": "assistant", "content": "Sure, which appointment would you like to cancel?"},
            {"role": "user", "content": "The Monday one"},
            {"role": "assistant", "content": "That's appt_1 on Monday at 09:30. Do you want me to cancel it?"},
            {"role": "user", "content": "Actually, wait. How should I prepare for an MRI?"},
            {"role": "assistant", "content": "Remove all metal objects, wear comfortable clothing, and inform staff about any implants."},
        ],
        "user_query": "OK thanks. Yes, go ahead",
        "expected_tool": "CANCEL_APPOINTMENT",
        "description": "CANCEL with 'that appointment' after digression (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Hello, I need some help"},
            {"role": "assistant", "content": "Hello! I'd be happy to help. What do you need?"},
            {"role": "user", "content": "How does an MRI work?"},
            {"role": "assistant", "content": "MRI uses strong magnetic fields and radio waves to produce detailed images of your organs and tissues."},
            {"role": "user", "content": "Book me slot_10"},
            {"role": "assistant", "content": "Your appointment for slot_10 (Tuesday 07:30) has been booked."},
            {"role": "user", "content": "Reschedule appt_2 to slot_20"},
            {"role": "assistant", "content": "Done! appt_2 moved to slot_20 (Wednesday 08:30)."},
        ],
        "user_query": "I need to cancel appointment appt_3",
        "expected_tool": "CANCEL_APPOINTMENT",
        "description": "CANCEL after LLM+RAG+BOOK+RESCHEDULE history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Tell my doctor about my test results"},
            {"role": "assistant", "content": "I've sent your message about test results to the doctor."},
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_2 Tue 14:00, appt_3 Wed 08:30."},
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_5 Mon 11:30, slot_12 Tue 09:30, slot_19 Wed 07:30..."},
            {"role": "user", "content": "Can I wear jewelry during a CT scan?"},
            {"role": "assistant", "content": "No, you should remove all jewelry and metal objects before a CT scan."},
        ],
        "user_query": "Please cancel my booking appt_2",
        "expected_tool": "CANCEL_APPOINTMENT",
        "description": "CANCEL after MSG+LIST_APPTS+LIST_SLOTS+RAG history (3/3)",
    },

    # ── RESCHEDULE_APPOINTMENT ──────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_2 Tue 14:00."},
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_5 Mon 11:30, slot_12 Tue 09:30, slot_20 Wed 08:30..."},
            {"role": "user", "content": "I'm feeling anxious about my appointment"},
            {"role": "assistant", "content": "I understand your anxiety. Our staff will make sure you're comfortable. Take deep breaths."},
            {"role": "user", "content": "How should I prepare for CT?"},
            {"role": "assistant", "content": "Fast for 4 hours if contrast dye is used. Remove metal and wear comfortable clothes."},
        ],
        "user_query": "Reschedule appt_1 to slot_20",
        "expected_tool": "RESCHEDULE_APPOINTMENT",
        "description": "RESCHEDULE after LIST_APPTS+LIST_SLOTS+LLM+RAG history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_2 Tue 14:00."},
            {"role": "user", "content": "I want to move the Tuesday one to a different day"},
            {"role": "assistant", "content": "I can reschedule appt_2. Would you prefer Monday or Wednesday?"},
            {"role": "user", "content": "Wednesday, what's available?"},
            {"role": "assistant", "content": "On Wednesday: slot_19 at 07:30, slot_20 at 08:30, slot_21 at 09:30."},
            {"role": "user", "content": "How about slot_20?"},
            {"role": "assistant", "content": "Slot_20 on Wednesday at 08:30 is available. Should I reschedule your appointment there?"},
        ],
        "user_query": "Yes, do that",
        "expected_tool": "RESCHEDULE_APPOINTMENT",
        "description": "RESCHEDULE with 'do that' after negotiation (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "Can I eat before an MRI?"},
            {"role": "assistant", "content": "Generally yes, unless your doctor gave specific fasting instructions."},
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_9 Mon 16:00, slot_15 Tue 13:00..."},
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_2 Wed 08:30."},
        ],
        "user_query": "I want to change appt_1 to slot_9",
        "expected_tool": "RESCHEDULE_APPOINTMENT",
        "description": "RESCHEDULE after LLM+RAG+LIST_SLOTS+LIST_APPTS history (3/3)",
    },

    # ── SEND_DOCTOR_MESSAGE ─────────────────────────────────────────────
    {
        "history": [
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_5 Mon 11:30, slot_10 Tue 07:30..."},
            {"role": "user", "content": "Book me slot_5"},
            {"role": "assistant", "content": "Your appointment for slot_5 (Monday 11:30) has been booked."},
            {"role": "user", "content": "What should I prepare for an MRI?"},
            {"role": "assistant", "content": "Remove all metal objects. Wear comfortable clothing. Inform staff about implants."},
            {"role": "user", "content": "Cancel my appointment appt_2"},
            {"role": "assistant", "content": "Your appointment appt_2 has been cancelled."},
        ],
        "user_query": "Send a message to my doctor saying I feel dizzy and nauseous",
        "expected_tool": "SEND_DOCTOR_MESSAGE",
        "description": "SEND_DOCTOR_MESSAGE after LIST_SLOTS+BOOK+RAG+CANCEL history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_2 Tue 14:00."},
            {"role": "user", "content": "Reschedule appt_1 to slot_15"},
            {"role": "assistant", "content": "Done! appt_1 moved to slot_15 (Tuesday 13:00)."},
            {"role": "user", "content": "How does a CT scan work?"},
            {"role": "assistant", "content": "A CT scan uses X-rays from multiple angles to create cross-sectional images of your body."},
            {"role": "user", "content": "Hello, I'm feeling worried"},
            {"role": "assistant", "content": "I understand your concern. Our medical staff is here to support you every step of the way."},
        ],
        "user_query": "Tell my doctor about my persistent headaches",
        "expected_tool": "SEND_DOCTOR_MESSAGE",
        "description": "SEND_DOCTOR_MESSAGE after LIST_APPTS+RESCHEDULE+RAG+LLM history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Book me slot_19"},
            {"role": "assistant", "content": "Your appointment for slot_19 (Wednesday 07:30) has been booked."},
            {"role": "user", "content": "Cancel appt_3"},
            {"role": "assistant", "content": "Your appointment appt_3 has been cancelled."},
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_3 Mon 09:30, slot_10 Tue 07:30..."},
            {"role": "user", "content": "Can I drink before an ultrasound?"},
            {"role": "assistant", "content": "For some ultrasounds you may need a full bladder, so drinking water may be recommended."},
        ],
        "user_query": "I want to contact my doctor about my lab results",
        "expected_tool": "SEND_DOCTOR_MESSAGE",
        "description": "SEND_DOCTOR_MESSAGE after BOOK+CANCEL+LIST_SLOTS+RAG history (3/3)",
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
            {"role": "user", "content": "Tell my doctor I need new test results"},
            {"role": "assistant", "content": "I've sent your message about test results to the doctor."},
        ],
        "user_query": "How should I prepare for an MRI?",
        "expected_tool": "RAG",
        "description": "RAG after BOOK+LIST_APPTS+CANCEL+MSG history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "Show me available slots"},
            {"role": "assistant", "content": "Available: slot_1 Mon 07:30, slot_5 Mon 11:30, slot_10 Tue 07:30..."},
            {"role": "user", "content": "Book me slot_10"},
            {"role": "assistant", "content": "Your appointment for slot_10 (Tuesday 07:30) has been booked."},
            {"role": "user", "content": "Reschedule appt_2 to slot_20"},
            {"role": "assistant", "content": "Done! appt_2 moved to slot_20 (Wednesday 08:30)."},
        ],
        "user_query": "Can I wear jewelry during a CT scan?",
        "expected_tool": "RAG",
        "description": "RAG after LLM+LIST_SLOTS+BOOK+RESCHEDULE history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "Cancel appt_3"},
            {"role": "assistant", "content": "Your appointment appt_3 has been cancelled."},
            {"role": "user", "content": "Send a message to my doctor about my knee pain"},
            {"role": "assistant", "content": "I've sent your message about knee pain to the doctor."},
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_2 Tue 14:00."},
            {"role": "user", "content": "Thank you"},
            {"role": "assistant", "content": "You're welcome! Let me know if you need anything else."},
        ],
        "user_query": "What should I do before an ultrasound examination?",
        "expected_tool": "RAG",
        "description": "RAG after CANCEL+MSG+LIST_APPTS+LLM history (3/3)",
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
            {"role": "user", "content": "Cancel my appointment appt_3"},
            {"role": "assistant", "content": "Your appointment appt_3 has been cancelled."},
        ],
        "user_query": "Thank you for all your help today!",
        "expected_tool": "LLM",
        "description": "LLM after LIST_SLOTS+BOOK+RAG+CANCEL history (1/3)",
    },
    {
        "history": [
            {"role": "user", "content": "What appointments do I have?"},
            {"role": "assistant", "content": "You have: appt_1 Mon 09:30, appt_2 Tue 14:00."},
            {"role": "user", "content": "Reschedule appt_1 to slot_6"},
            {"role": "assistant", "content": "Done! appt_1 moved to slot_6 (Monday 13:00)."},
            {"role": "user", "content": "Send my doctor a message about my blood pressure"},
            {"role": "assistant", "content": "I've sent your message about blood pressure to the doctor."},
            {"role": "user", "content": "Can I eat before an MRI?"},
            {"role": "assistant", "content": "Generally yes, unless your doctor gave specific fasting instructions."},
        ],
        "user_query": "Goodbye, have a great day!",
        "expected_tool": "LLM",
        "description": "LLM after LIST_APPTS+RESCHEDULE+MSG+RAG history (2/3)",
    },
    {
        "history": [
            {"role": "user", "content": "How does a CT scan work?"},
            {"role": "assistant", "content": "A CT scan uses X-rays from multiple angles to create cross-sectional images."},
            {"role": "user", "content": "Book me slot_19"},
            {"role": "assistant", "content": "Your appointment for slot_19 (Wednesday 07:30) has been booked."},
            {"role": "user", "content": "Cancel appt_2"},
            {"role": "assistant", "content": "Your appointment appt_2 has been cancelled."},
            {"role": "user", "content": "Tell my doctor about my allergy to latex"},
            {"role": "assistant", "content": "I've sent your latex allergy information to the doctor. Is there anything else you need?"},
        ],
        "user_query": "No, that's everything, thanks!",
        "expected_tool": "LLM",
        "description": "LLM with 'that's everything' closing (3/3)",
    },
]

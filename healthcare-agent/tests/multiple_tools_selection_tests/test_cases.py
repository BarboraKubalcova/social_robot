"""
Test cases for specific ACTION tool selection evaluation.

Each test case: (user_message, expected_tool)

The 6 action tools:
  LIST_SLOTS, LIST_APPOINTMENTS, BOOK_APPOINTMENT,
  CANCEL_APPOINTMENT, RESCHEDULE_APPOINTMENT, SEND_DOCTOR_MESSAGE

3 test cases per tool = 18 total.
"""

TEST_CASES = [
    # ── LIST_SLOTS ──────────────────────────────────────────────────────
    ("Show me available slots", "LIST_SLOTS"),
    ("Are there any free slots on Monday?", "LIST_SLOTS"),
    ("What times can I come in this week?", "LIST_SLOTS"),

    # ── LIST_APPOINTMENTS ───────────────────────────────────────────────
    ("What appointments do I have?", "LIST_APPOINTMENTS"),
    ("Show my booked appointments", "LIST_APPOINTMENTS"),
    ("List my scheduled appointments", "LIST_APPOINTMENTS"),

    # ── BOOK_APPOINTMENT ────────────────────────────────────────────────
    ("Book me an appointment for slot_5", "BOOK_APPOINTMENT"),
    ("I'd like to schedule slot_12", "BOOK_APPOINTMENT"),
    ("Reserve slot_3 for me please", "BOOK_APPOINTMENT"),

    # ── CANCEL_APPOINTMENT ──────────────────────────────────────────────
    ("Cancel my appointment appt_2", "CANCEL_APPOINTMENT"),
    ("I need to cancel appointment appt_1", "CANCEL_APPOINTMENT"),
    ("Please cancel my booking appt_3", "CANCEL_APPOINTMENT"),

    # ── RESCHEDULE_APPOINTMENT ──────────────────────────────────────────
    ("Reschedule appt_1 to slot_9", "RESCHEDULE_APPOINTMENT"),
    ("Move my appointment appt_2 to slot_15", "RESCHEDULE_APPOINTMENT"),
    ("I want to change appt_3 to slot_20", "RESCHEDULE_APPOINTMENT"),

    # ── SEND_DOCTOR_MESSAGE ─────────────────────────────────────────────
    ("Send a message to my doctor saying I feel unwell", "SEND_DOCTOR_MESSAGE"),
    ("Say my doctor about my test results", "SEND_DOCTOR_MESSAGE"),
    ("I want to contact my doctor about my medication", "SEND_DOCTOR_MESSAGE"),
]

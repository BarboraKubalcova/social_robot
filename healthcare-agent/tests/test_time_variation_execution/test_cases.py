"""
Actual execution test cases for appointment booking.

Two groups:
  1. DIRECT SLOT ID — user mentions an explicit slot_id (slot_1, slot_5, …)
  2. DAY + TIME     — user mentions a day and time (Monday 9:30, Wednesday 13:00, …)

Each test case:
    (user_message, expected_slot_id)

The expected_slot_id is the slot that should end up booked in the
AppointmentManager table after the agent processes the message.

Slot mapping (start_date = 2026-06-01 = Monday, 3 days, 9 slots/day):
  Monday    slot_1=07:30  slot_2=08:30  slot_3=09:30  slot_4=10:30  slot_5=11:30
            slot_6=13:00  slot_7=14:00  slot_8=15:00  slot_9=16:00
  Tuesday   slot_10=07:30 slot_11=08:30 slot_12=09:30 slot_13=10:30 slot_14=11:30
            slot_15=13:00 slot_16=14:00 slot_17=15:00 slot_18=16:00
  Wednesday slot_19=07:30 slot_20=08:30 slot_21=09:30 slot_22=10:30 slot_23=11:30
            slot_24=13:00 slot_25=14:00 slot_26=15:00 slot_27=16:00
"""

# ── Group 1: Direct slot ID ────────────────────────────────────────────────

DIRECT_SLOT_CASES = [
    ("Book slot_1 for me please", "slot_1"),
    ("I'd like to reserve slot_5", "slot_5"),
    ("Schedule me for slot_7", "slot_7"),
    ("Can you book slot_12 for me?", "slot_12"),
    ("Please book appointment slot_15", "slot_15"),
    ("I want slot_18", "slot_18"),
    ("Reserve slot_20 please", "slot_20"),
    ("Put me down for slot_23", "slot_23"),
    ("Book slot_25 for my appointment", "slot_25"),
    ("I'd like to book slot_27", "slot_27"),
]

# ── Group 2: Day + time ───────────────────────────────────────────────────

DAY_TIME_CASES = [
    ("Book me an appointment on Monday at 9:30", "slot_3"),
    ("I want an appointment on Mon at 14:00", "slot_7"),
    ("Schedule me for Tuesday at 8:30", "slot_11"),
    ("Can I get an appointment on Tuesday at 13:00?", "slot_15"),
    ("Book an appointment for wednesday at 7:30", "slot_19"),
    ("I'd like Wednesday at 10:30 please", "slot_22"),
    ("Reserve me a spot on Mon at 16:00", "slot_9"),
    ("I want to book Tue at 15:00", "slot_17"),
    ("Can you book Wed at 14:00 for me?", "slot_25"),
    ("Schedule an appointment for Monday at 11:30", "slot_5"),
]

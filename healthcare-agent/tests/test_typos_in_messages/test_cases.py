"""
Typo-focused test cases for tool selection evaluation.

Each message contains realistic misspellings, character swaps, missing letters,
or keyboard-adjacent errors. Tests whether agents can still route to the
correct *specific* tool despite garbled input.

5 test cases per tool = 40 total.

Each test case: (user_message, expected_tool)

Tools tested:
  LLM, RAG, LIST_SLOTS, LIST_APPOINTMENTS, BOOK_APPOINTMENT,
  CANCEL_APPOINTMENT, RESCHEDULE_APPOINTMENT, SEND_DOCTOR_MESSAGE
"""

TEST_CASES = [
    # ── LLM (CHAT) ─────────────────────────────────────────────────────
    ("Helo thre, how r you?", "LLM"),
    ("Goood mornig, hwo are you doin?", "LLM"),
    ("Thnk you for yoru hlep", "LLM"),
    ("Wat is yuor name?", "LLM"),
    ("Goodby, se you latre", "LLM"),

    # ── RAG ─────────────────────────────────────────────────────────────
    ("How shoudl I preprae for an MIR?", "RAG"),
    ("Can I waer jewlery durign an MRI?", "RAG"),
    ("Whta is the fasitng polcy befroe a CT?", "RAG"),
    ("Hwo does a ultrasond procedur wrk?", "RAG"),
    ("Can I drnk befroe ultrsound examinaiton?", "RAG"),

    # ── LIST_SLOTS ──────────────────────────────────────────────────────
    ("Shwo me avialble sltos", "LIST_SLOTS"),
    ("Are tehre any fre slots on Moday?", "LIST_SLOTS"),
    ("Waht tiems can I cme in tihs week?", "LIST_SLOTS"),
    ("Lst avalable apointment tmes", "LIST_SLOTS"),
    ("Do you haev any opne slots?", "LIST_SLOTS"),

    # ── LIST_APPOINTMENTS ───────────────────────────────────────────────
    ("Waht apointments do I hve?", "LIST_APPOINTMENTS"),
    ("Shwo my bokked apointments", "LIST_APPOINTMENTS"),
    ("Lsit my sceduled apointmnets", "LIST_APPOINTMENTS"),
    ("Can I se my upcomign apointments?", "LIST_APPOINTMENTS"),
    ("Dsplay my currnet bokings", "LIST_APPOINTMENTS"),

    # ── BOOK_APPOINTMENT ────────────────────────────────────────────────
    ("Boko me an apointmnet for slot_5", "BOOK_APPOINTMENT"),
    ("I'd liek to scedule slot_12", "BOOK_APPOINTMENT"),
    ("Resrve slot_3 for me plese", "BOOK_APPOINTMENT"),
    ("Bok slot_7 plase", "BOOK_APPOINTMENT"),
    ("I wnat to boko an apointment for slot_10", "BOOK_APPOINTMENT"),

    # ── CANCEL_APPOINTMENT ──────────────────────────────────────────────
    ("Cancle my apointment appt_2", "CANCEL_APPOINTMENT"),
    ("I ned to cancl apointmnet appt_1", "CANCEL_APPOINTMENT"),
    ("Plese cancle my bokking appt_3", "CANCEL_APPOINTMENT"),
    ("Cansel appt_2 plz", "CANCEL_APPOINTMENT"),
    ("Remov my apointmnet appt_1", "CANCEL_APPOINTMENT"),

    # ── RESCHEDULE_APPOINTMENT ──────────────────────────────────────────
    ("Reshedule appt_1 to slot_9", "RESCHEDULE_APPOINTMENT"),
    ("Mve my apointment appt_2 to slot_15", "RESCHEDULE_APPOINTMENT"),
    ("I wnat to chnage appt_3 to slot_20", "RESCHEDULE_APPOINTMENT"),
    ("Rescheudle appt_1 to slto_9 plz", "RESCHEDULE_APPOINTMENT"),
    ("Chagne the tiem of appt_2 to slot_11", "RESCHEDULE_APPOINTMENT"),

    # ── SEND_DOCTOR_MESSAGE ─────────────────────────────────────────────
    ("Sned a mesage to my doctr sayign I feel unwel", "SEND_DOCTOR_MESSAGE"),
    ("Tll my docor about my tset resutls", "SEND_DOCTOR_MESSAGE"),
    ("I wnat to contct my doctro abut my medicaiton", "SEND_DOCTOR_MESSAGE"),
    ("Sennd a mesage to the doctr plese", "SEND_DOCTOR_MESSAGE"),
    ("Wrte to my docotr that I hav a fver", "SEND_DOCTOR_MESSAGE"),
]

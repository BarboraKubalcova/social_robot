"""
Language variation / paraphrasing test cases for tool selection evaluation.

Same intent expressed in very different styles:
  - formal
  - slang / casual
  - typos / misspellings
  - non-native phrasing
  - verbose / roundabout

5 test cases per tool = 40 total.

Each test case: (user_message, expected_tool)

Tools tested:
  LLM, RAG, LIST_SLOTS, LIST_APPOINTMENTS, BOOK_APPOINTMENT,
  CANCEL_APPOINTMENT, RESCHEDULE_APPOINTMENT, SEND_DOCTOR_MESSAGE
"""

TEST_CASES = [
    # ── LLM (CHAT) ─────────────────────────────────────────────────────
    # formal
    ("Good evening, I wish to extend my warmest greetings to you.", "LLM"),
    # slang
    ("yo whats up buddy how u doing", "LLM"),
    # typos
    ("Helo, how are yuo todya?", "LLM"),
    # non-native
    ("I am having the good day, please to tell me how you are feeling?", "LLM"),
    # verbose
    ("I just wanted to drop by and say hello and see if you could maybe chat with me for a little while", "LLM"),

    # ── RAG ─────────────────────────────────────────────────────────────
    # formal
    ("I would like to enquire about the preparatory procedures required prior to undergoing an MRI examination.", "RAG"),
    # slang
    ("yo what do i gotta do before getting an mri", "RAG"),
    # typos
    ("How shoud I prepre for a CT scna?", "RAG"),
    # non-native
    ("Please can you explain me what is the procedure for ultrasound, what I must do before?", "RAG"),
    # verbose
    ("I have an X-ray coming up soon and I was wondering if there are any special instructions or things I should know about how to get ready for it", "RAG"),

    # ── LIST_SLOTS ──────────────────────────────────────────────────────
    # formal
    ("I wish to enquire about the currently available appointment slots.", "LIST_SLOTS"),
    # slang
    ("yo got any slots open?", "LIST_SLOTS"),
    # typos
    ("Shwo me avalable sltos", "LIST_SLOTS"),
    # non-native
    ("I am wanting to see what times are free for making the appointment please", "LIST_SLOTS"),
    # verbose
    ("Could you possibly check and let me know which time slots are still available for booking at the moment?", "LIST_SLOTS"),

    # ── LIST_APPOINTMENTS ───────────────────────────────────────────────
    # formal
    ("I would appreciate it if you could display my currently scheduled appointments.", "LIST_APPOINTMENTS"),
    # slang
    ("lemme see my bookings", "LIST_APPOINTMENTS"),
    # typos
    ("Shwo my apointments pleae", "LIST_APPOINTMENTS"),
    # non-native
    ("I need for to see what appointments I am having booked", "LIST_APPOINTMENTS"),
    # verbose
    ("I was wondering if you could pull up a list of all the appointments that I currently have on my schedule", "LIST_APPOINTMENTS"),

    # ── BOOK_APPOINTMENT ────────────────────────────────────────────────
    # formal
    ("I would like to formally request the reservation of slot_7 for my appointment.", "BOOK_APPOINTMENT"),
    # slang
    ("yo book me in for slot_7", "BOOK_APPOINTMENT"),
    # typos
    ("Boko slot_7 for me plase", "BOOK_APPOINTMENT"),
    # non-native
    ("I am wanting to make the booking for slot_7 please", "BOOK_APPOINTMENT"),
    # verbose
    ("If it's not too much trouble, I'd really like to go ahead and book slot_7 as my appointment time", "BOOK_APPOINTMENT"),

    # ── CANCEL_APPOINTMENT ──────────────────────────────────────────────
    # formal
    ("I hereby request the cancellation of my appointment appt_2.", "CANCEL_APPOINTMENT"),
    # slang
    ("cancle appt_2 pls", "CANCEL_APPOINTMENT"),
    # typos
    ("Cancl my apointmnet appt_2", "CANCEL_APPOINTMENT"),
    # non-native
    ("I need for to remove my appointment appt_2, it is no longer needed", "CANCEL_APPOINTMENT"),
    # verbose
    ("I've been thinking about it and I've decided that I really need to go ahead and cancel my appointment appt_2", "CANCEL_APPOINTMENT"),

    # ── RESCHEDULE_APPOINTMENT ──────────────────────────────────────────
    # formal
    ("I would like to formally request that appointment appt_1 be rescheduled to slot_9.", "RESCHEDULE_APPOINTMENT"),
    # slang
    ("move appt_1 to slot_9 yeah?", "RESCHEDULE_APPOINTMENT"),
    # typos
    ("Rescheudle appt_1 to slto_9 plz", "RESCHEDULE_APPOINTMENT"),
    # non-native
    ("I am needing to change the time of appt_1, please to put it in slot_9", "RESCHEDULE_APPOINTMENT"),
    # verbose
    ("I was hoping you could help me move my existing appointment appt_1 over to a different time, specifically slot_9 if that works", "RESCHEDULE_APPOINTMENT"),

    # ── SEND_DOCTOR_MESSAGE ─────────────────────────────────────────────
    # formal
    ("I wish to compose and transmit a message to my physician regarding my current symptoms.", "SEND_DOCTOR_MESSAGE"),
    # slang
    ("tell my doc i aint feeling great", "SEND_DOCTOR_MESSAGE"),
    # typos
    ("Sennd a mesage to my doctr about my headaces", "SEND_DOCTOR_MESSAGE"),
    # non-native
    ("I am wanting for to write the message to my doctor because I have the problem with medicine", "SEND_DOCTOR_MESSAGE"),
    # verbose
    ("Could you please help me get in touch with my doctor by sending them a message about how I've been feeling lately", "SEND_DOCTOR_MESSAGE"),
]

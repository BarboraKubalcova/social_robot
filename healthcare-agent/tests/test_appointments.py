import pytest
from execution.actions.appointments import AppointmentManager

def test_book_appointment():
    manager = AppointmentManager()
    # Updated to reflect simplified signature
    result = manager.book_appointment("slot_1")
    assert result is True

def test_cancel_appointment():
    manager = AppointmentManager()
    result = manager.cancel_appointment("appt_1")
    assert result is True

def test_reschedule_appointment():
    manager = AppointmentManager()
    # Should cancel appt_1 and book slot_2
    result = manager.reschedule_appointment("appt_1", "slot_2")
    assert result is True

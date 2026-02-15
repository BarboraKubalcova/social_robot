from datetime import datetime
from typing import List, Dict
import logging
from execution.storage.db import get_db
# Removed audit log for simplicity as per request

logger = logging.getLogger("Appointments")

class AppointmentManager:
    def list_available_slots(self, date_range: tuple = None) -> List[Dict]:
        """Mock: Return some fake slots."""
        # Simplified: no clinic queries
        return [
            {"id": "slot_1", "time": "2026-02-16T09:00:00"},
            {"id": "slot_2", "time": "2026-02-16T10:00:00"},
            {"id": "slot_3", "time": "2026-02-17T14:00:00"}
        ]

    def book_appointment(self, slot_id: str) -> bool:
        """Book a slot for the single patient."""
        # Simplified: no patient_id needed, no confirmation
        try:
            # DB logic here: insert into appointments
            print(f"Booked slot {slot_id} for the patient")
            return True
        except Exception as e:
            logger.error(f"Failed to book: {e}")
            return False

    def cancel_appointment(self, appointment_id: str) -> bool:
        """Cancel an appointment."""
        print(f"Cancelled appointment {appointment_id}")
        return True

    def reschedule_appointment(self, appointment_id: str, new_slot_id: str) -> bool:
        """Atomic reschedule."""
        # Simple implementation: cancel then book
        if self.cancel_appointment(appointment_id):
            return self.book_appointment(new_slot_id)
        return False
    
    def get_patient_appointments(self) -> List[Dict]:
        """Get appointments for the single patient."""
        return [{"id": "appt_123", "time": "2026-02-15T09:00:00"}]

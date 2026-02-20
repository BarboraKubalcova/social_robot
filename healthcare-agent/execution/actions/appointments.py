from calendar import monthrange
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger("Appointments")


class AppointmentManager:
    def __init__(self, year: Optional[int] = None, month: Optional[int] = None):
        now = datetime.now()
        self.year = year or now.year
        self.month = month or now.month

        # Columns are times, rows are days in month.
        # Example: table[0][0] == day 1, first slot "07:30".
        self.time_columns: List[str] = [
            "07:30",
            "08:30",
            "09:30",
            "10:30",
            "11:30",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
        ]

        self.days_in_month = monthrange(self.year, self.month)[1]
        self.table: List[List[Optional[str]]] = [
            [None for _ in self.time_columns]
            for _ in range(self.days_in_month)
        ]

        self._appointment_counter = 1
        self._appointments: Dict[str, Dict] = {}

    def _slot_position(self, slot_id: str) -> Optional[Tuple[int, int]]:
        """Resolve a sequential slot id (slot_1, slot_2, ...) to row/column."""
        if not slot_id.startswith("slot_"):
            return None

        raw_index = slot_id.replace("slot_", "", 1)
        if not raw_index.isdigit():
            return None

        one_based = int(raw_index)
        if one_based <= 0:
            return None

        zero_based = one_based - 1
        columns = len(self.time_columns)
        day_index = zero_based // columns
        time_index = zero_based % columns

        if day_index < 0 or day_index >= self.days_in_month:
            return None
        return day_index, time_index

    def _slot_datetime(self, day_index: int, time_index: int) -> datetime:
        hour, minute = map(int, self.time_columns[time_index].split(":"))
        return datetime(self.year, self.month, day_index + 1, hour, minute)

    def _is_in_range(self, slot_dt: datetime, date_range: Optional[tuple]) -> bool:
        if not date_range:
            return True
        start, end = date_range
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        return start <= slot_dt <= end

    def list_available_slots(self, date_range: tuple = None) -> List[Dict]:
        """Return currently free slots from the timetable."""
        available: List[Dict] = []
        columns = len(self.time_columns)

        for day_index, row in enumerate(self.table):
            for time_index, cell in enumerate(row):
                if cell is not None:
                    continue

                slot_dt = self._slot_datetime(day_index, time_index)
                if not self._is_in_range(slot_dt, date_range):
                    continue

                slot_number = day_index * columns + time_index + 1
                available.append(
                    {
                        "id": f"slot_{slot_number}",
                        "day": day_index + 1,
                        "time": slot_dt.isoformat(),
                    }
                )

        return available

    def book_appointment(self, slot_id: str) -> bool:
        """Book a free timetable slot for the single patient."""
        try:
            position = self._slot_position(slot_id)
            if position is None:
                logger.warning("Invalid slot id: %s", slot_id)
                return False

            day_index, time_index = position
            if self.table[day_index][time_index] is not None:
                logger.info("Slot %s is already booked", slot_id)
                return False

            appointment_id = f"appt_{self._appointment_counter}"
            self._appointment_counter += 1

            slot_dt = self._slot_datetime(day_index, time_index)
            self.table[day_index][time_index] = appointment_id
            self._appointments[appointment_id] = {
                "id": appointment_id,
                "slot_id": slot_id,
                "day": day_index + 1,
                "time": slot_dt.isoformat(),
            }

            logger.info("Booked %s for slot %s", appointment_id, slot_id)
            return True
        except Exception as e:
            logger.error(f"Failed to book: {e}")
            return False

    def cancel_appointment(self, appointment_id: str) -> bool:
        """Cancel an existing appointment and free the timetable slot."""
        appointment = self._appointments.get(appointment_id)
        if appointment is None:
            logger.info("Appointment not found (already canceled): %s", appointment_id)
            return True

        position = self._slot_position(appointment["slot_id"])
        if position is None:
            logger.error("Corrupted appointment slot reference: %s", appointment_id)
            return False

        day_index, time_index = position
        self.table[day_index][time_index] = None
        del self._appointments[appointment_id]
        logger.info("Cancelled appointment %s", appointment_id)
        return True

    def reschedule_appointment(self, appointment_id: str, new_slot_id: str) -> bool:
        """Move an existing appointment to a new free slot."""
        appointment = self._appointments.get(appointment_id)
        if appointment is None:
            logger.info("Appointment not found for reschedule, booking new slot: %s", appointment_id)
            return self.book_appointment(new_slot_id)

        new_position = self._slot_position(new_slot_id)
        if new_position is None:
            logger.warning("Invalid new slot id for reschedule: %s", new_slot_id)
            return False

        new_day_index, new_time_index = new_position
        if self.table[new_day_index][new_time_index] is not None:
            logger.info("Target slot %s is already booked", new_slot_id)
            return False

        old_position = self._slot_position(appointment["slot_id"])
        if old_position is None:
            logger.error("Corrupted old slot id in appointment: %s", appointment_id)
            return False

        old_day_index, old_time_index = old_position
        self.table[old_day_index][old_time_index] = None
        self.table[new_day_index][new_time_index] = appointment_id

        new_dt = self._slot_datetime(new_day_index, new_time_index)
        appointment["slot_id"] = new_slot_id
        appointment["day"] = new_day_index + 1
        appointment["time"] = new_dt.isoformat()
        logger.info("Rescheduled appointment %s to slot %s", appointment_id, new_slot_id)
        return True
    
    def get_patient_appointments(self) -> List[Dict]:
        """Get all appointments for the single patient."""
        return sorted(self._appointments.values(), key=lambda item: item["time"])

    def get_timetable(self) -> List[List[Optional[str]]]:
        """Return the timetable matrix (rows=days, columns=times)."""
        return [row[:] for row in self.table]

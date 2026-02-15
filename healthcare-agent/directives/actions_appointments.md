# Action: Appointments

**Purpose**: Manage patient appointments.

## Available Tools
- `list_available_slots(clinic_id, date_range)`: Find free time slots.
- `book_appointment(patient_id, slot_id)`: Reserve a slot.
- `cancel_appointment(appointment_id)`: Remove a reservation.
- `reschedule_appointment(appointment_id, new_slot_id)`: Atomic cancel + book.
- `get_patient_appointments(patient_id)`: See what is currently booked.

## Workflow: Rescheduling
1.  **Identify**: User wants to reschedule. Ask for current appointment details if unknown (or lookup context).
2.  **Search**: "When would you like to come in instead?" -> Call `list_available_slots`.
3.  **Propose**: "I found a slot on [Date] at [Time]. Does that work?"
4.  **Confirm**: "Ok, I will move your appointment from [Old] to [New]. Confirm?"
5.  **Execute**: Call `reschedule_appointment` ONLY after explicit "yes".

## Dependencies
- Requires `patient_id` (authenticated user).
- Cannot book slots in the past.

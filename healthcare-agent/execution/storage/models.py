from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class Patient(BaseModel):
    id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None

class Appointment(BaseModel):
    id: str
    patient_id: str
    start_time: datetime
    end_time: datetime
    status: str

class Slot(BaseModel):
    id: str
    start_time: datetime
    end_time: datetime
    is_available: bool

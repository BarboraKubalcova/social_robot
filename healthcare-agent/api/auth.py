from fastapi import Depends

async def get_current_user():
    """
    Simplified auth: always returns a default mock user.
    """
    return {"id": "default_patient", "role": "patient"}

async def get_admin_user():
    """
    Simplified admin auth: always returns default admin.
    """
    return {"id": "admin", "role": "admin"}

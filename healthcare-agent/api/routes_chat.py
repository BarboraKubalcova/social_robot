from fastapi import APIRouter, Depends
from pydantic import BaseModel
from api.auth import get_current_user
from orchestration.manager_basic_tools import AgentManager
from orchestration.manager_deterministic import DeterministicAgentManager
from orchestration.manager_commands import PlannedAgentManager
from orchestration.manager_multiple_tools import AgentManagerMultiTools
from orchestration.manager_multi_agent import MultiAgentManager

router = APIRouter(prefix="/chat", tags=["chat"])

# Singleton manager
agent_manager = AgentManager()
deterministic_manager = DeterministicAgentManager()
planned_manager = PlannedAgentManager()
agent_multi_tools_manager = AgentManagerMultiTools()
multi_agent_manager = MultiAgentManager()   


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    intent: str
    history: list


@router.get("/appointments")
async def get_appointments():
    """Return all appointment slots with their free/occupied status."""
    return multi_agent_manager.appointments.get_all_slots()


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, user: dict = Depends(get_current_user)):
    """
    Main chat endpoint.
    """
    # result = await agent_manager.process_message(request.message, user["id"])
    # result = await deterministic_manager.process_message(request.message, user["id"])
    # result = await planned_manager.process_message(request.message, user["id"])
    result = await agent_multi_tools_manager.process_message(request.message, user["id"])
    # result = await multi_agent_manager.process_message(request.message, user["id"])
    print(f"[API DEBUG] Manager result: {result}")

    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        history=result["history"]
    )

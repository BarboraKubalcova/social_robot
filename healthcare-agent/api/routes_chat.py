from fastapi import APIRouter, Depends
from pydantic import BaseModel
from api.auth import get_current_user
from orchestration.manager import AgentManager

router = APIRouter(prefix="/chat", tags=["chat"])

# Singleton manager
agent_manager = AgentManager()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    intent: str
    history: list

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, user: dict = Depends(get_current_user)):
    """
    Main chat endpoint.
    """
    result = await agent_manager.process_message(request.message, user["id"])
    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        history=result["history"]
    )

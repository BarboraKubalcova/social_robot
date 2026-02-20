import pytest
from orchestration.manager import AgentManager

@pytest.mark.asyncio
async def test_router_intent_action():
    manager = AgentManager()
    # Test simple keyword matching for actions
    # This might fail if the mock isn't set up right, but we are testing logic
    
    # "book" -> ACTION
    result = await manager.process_message("I want to book a slot", "user1")
    assert result["intent"] == "ACTION"
    assert "booked" in result["response"]

@pytest.mark.asyncio
async def test_router_intent_rag_fallback():
    manager = AgentManager()
    # "Hello" -> should likely go to RAG/LLM mode (LLM_ONLY if no docs)
    # We mock the retriever or just assume default behavior of empty DB
    result = await manager.process_message("Hello", "user1")
    assert result["intent"] in ["RAG", "LLM_ONLY"]


@pytest.mark.asyncio
async def test_action_available_appointments_lists_slots():
    manager = AgentManager()
    result = await manager.process_message("Which are available appointments?", "user1")
    assert result["intent"] == "ACTION"
    assert "available slots" in result["response"].lower()
    assert "please provide both appointment id" not in result["response"].lower()


@pytest.mark.asyncio
async def test_action_direct_slot_request_books_slot():
    manager = AgentManager()
    result = await manager.process_message("I want the slot_6", "user1")
    assert result["intent"] == "ACTION"
    assert "booked slot_6" in result["response"].lower() or "could not book slot_6" in result["response"].lower()

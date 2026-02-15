# UI & Debug Directives

## Frontend Requirements
- **Chat Interface**: Standard list of messages (User right, Agent left).
- **Tool Traces**: To build trust, show a collapsible "Thought Process" or "Debug" view that shows:
    - Intent detected (Chat/RAG/Action).
    - Tools called and their arguments.
    - Retrieved documents (for RAG).

## Debug Panel (for Developers/Admins)
- Toggle to enable/disable specific tools.
- View raw JSON logs of the conversation.
- Button to clear session memory.

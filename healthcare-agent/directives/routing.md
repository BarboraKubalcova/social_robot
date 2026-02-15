# Routing Logic

This directive defines how to triage user inputs.

## Intents

### 1. `CHAT` (Support / Chit-chat)
- **Definition**: The user is engaging in casual conversation, expressing emotions, or asking non-procedural questions.
- **Examples**:
    - "I'm feeling anxious."
    - "Hello, are you a robot?"
    - "Thanks for your help."
- **Action**: Route to `Support Chat Agent`.

### 2. `RAG` (Procedure Q&A)
- **Definition**: The user asks a question about clinic rules, preparations, or specific medical procedures documented in the knowledge base.
- **Examples**:
    - "Can I drink water before the MRI?"
    - "What is an ultrasound?"
    - "Do I need to bring my ID?"
- **Action**: Route to `RAG Agent` (retrieve -> generate).

### 3. `ACTION` (Tool Use)
- **Definition**: The user wants to perform a specific task involving their data or clinic scheduling.
- **Examples**:
    - "Book an appointment."
    - "Reschedule my visit."
    - "Cancel my slot."
    - "Send a message to Dr. House."
- **Action**: Route to `Action Agent` (plan -> confirm -> execute).

## Ambiguity
- If the intent is unclear between RAG and ACTION, prefer **RAG** (information first).
- If the user expresses an emergency (see `system_policy.md`), override all routing and trigger **EMERGENCY_RESPONSE**.

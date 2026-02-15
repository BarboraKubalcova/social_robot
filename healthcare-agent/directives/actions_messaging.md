# Action: Messaging

**Purpose**: Send non-urgent messages to healthcare providers.

## Available Tools
- `list_doctors(clinic_id)`: See who can be messaged.
- `draft_message(recipient_id, subject, body)`: Create a draft.
- `send_message(draft_id)`: Finalize and send.

## Workflow
1.  **Drafting**: Ask user for the recipient (or infer from context) and the message content.
2.  **Review**: "Here is the drafted message to Dr. [Name]: '[Body]'. Should I send it?"
3.  **Execute**: Call `send_message` ONLY after explicit "yes".

## Constraints
- **Severity Check**: If the message contains emergency keywords, **refuse to send** and direct the user to emergency services.
- **Length**: Suggest keeping messages concise.

import logging
from openai import OpenAI
from app.config import settings
from app.services.dynamodb import get_history, get_summary, update_summary

log = logging.getLogger(__name__)

RECENT_WINDOW = 5  


def build_context(chat_id: str) -> str:
    """
    Build context for the LLM from stored summary + recent messages.
    No LLM call here — this is just a DynamoDB read.
    """
    summary = get_summary(chat_id)
    recent = list(reversed(get_history(chat_id, limit=RECENT_WINDOW)))  

    if not summary and not recent:
        return ""

    parts = []
    if summary:
        parts.append(f"[Conversation summary so far]\n{summary}")
    if recent:
        parts.append("[Recent messages]")
        for msg in recent:
            parts.append(f"User: {msg['query']}")
            parts.append(f"Assistant: {msg['answer']}")

    return "\n".join(parts)


def roll_summary(chat_id: str):
    history = get_history(chat_id, limit=RECENT_WINDOW + 1)

    if len(history) <= RECENT_WINDOW:
        return 

    oldest = history[RECENT_WINDOW]
    current_summary = get_summary(chat_id)

    new_summary = _merge_into_summary(current_summary, oldest)
    update_summary(chat_id, new_summary)
    log.info(f"Rolled summary for chat_id={chat_id}")


def _merge_into_summary(current_summary: str, message: dict) -> str:
    client = OpenAI(api_key=settings.openai_api_key)

    parts = []
    if current_summary:
        parts.append(f"Existing summary: {current_summary}")
    parts.append(
        f"New exchange to add:\nUser: {message['query']}\nAssistant: {message['answer']}"
    )

    resp = client.chat.completions.create(
        model=settings.openai_model_id,
        messages=[
            {
                "role": "system",
                "content": (
                    "Update the conversation summary to include the new exchange. "
                    "Keep it under 3 concise sentences. Preserve key facts."
                ),
            },
            {"role": "user", "content": "\n\n".join(parts)},
        ],
        max_tokens=120,
    )
    return resp.choices[0].message.content.strip()

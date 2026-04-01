import os
import json
import time
import logging
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
API_URL = os.getenv("API_URL", "http://api:8000")

EDIT_INTERVAL = 1.0


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Salem! I'm your AI assistant.\n\n"
        "You can ask me about:\n"
        "- Users, transactions, and products (from the database)\n"
        "- Policies, documentation, and guides (from the knowledge base)\n\n"
        "Just send me any question!"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    query = update.message.text

    msg = await update.message.reply_text("Thinking... ▌", parse_mode="HTML")

    full_text = ""
    last_edit = 0.0

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{API_URL}/invoke/stream",
                json={"query": query, "chat_id": chat_id},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    full_text += json.loads(payload).get("token", "")

                    now = time.monotonic()
                    if now - last_edit >= EDIT_INTERVAL and full_text:
                        await msg.edit_text(full_text + "▌", parse_mode="HTML")
                        last_edit = now

        if full_text:
            await msg.edit_text(full_text, parse_mode="HTML")

    except httpx.HTTPError as e:
        log.error(f"HTTP error: {e}")
        await msg.edit_text("Sorry, I couldn't reach the server. Please try again later =)")
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        await msg.edit_text("Something went wrong. Please try again later =)")


def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable is not set.")

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Bot started, polling for messages...")
    app.run_polling()


if __name__ == "__main__":
    main()

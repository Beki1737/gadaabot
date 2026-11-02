"""Telegram front end for GadaaBot 2.0 (Telegram is widely used in Ethiopia and works on low-end phones).

It forwards each text message to the GadaaBot HTTP API and replies with the answer.
Run:  TELEGRAM_TOKEN=<token from @BotFather> GADAABOT_API=http://127.0.0.1:8080 python -m gadaabot.serve.telegram_bot
Needs: python-telegram-bot >= 21, httpx.
"""

from __future__ import annotations

import logging
import os

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

API = os.environ.get("GADAABOT_API", "http://127.0.0.1:8080").rstrip("/")
WELCOME = ("Akkam! Ani GadaaBot. Gaaffii fayyaa fi barnootaa Afaan Oromootiin na gaafadhu. "
           "Deebiin koo maddeewwan isaa waliin dhufa. Odeeffannoon fayyaa gorsa ogeessa fayyaa bakka hin bu'u; "
           "haala ariifachiisaa keessatti hatattamaan buufata fayyaa deemi.")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


async def start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(WELCOME)


async def on_text(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    await update.message.chat.send_action("typing")
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{API}/api/ask", json={"question": update.message.text[:1000]})
        text = r.json().get("answer") if r.status_code == 200 else r.json().get("detail", "Dogoggora.")
    except (httpx.HTTPError, ValueError):
        text = "Dhiifama, tajaajilli amma hin hojjetu. Booda irra deebi'ii yaali."
    for i in range(0, len(text), 4000):  # Telegram message limit 4096 characters
        await update.message.reply_text(text[i:i + 4000])


def main() -> None:
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_TOKEN (create a bot with @BotFather).")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling()


if __name__ == "__main__":
    main()

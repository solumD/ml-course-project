from __future__ import annotations

import logging
from pathlib import Path

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from bot.bot_engine import BotEngine
from bot.config import load_settings
from bot.state import UserProfile


logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
SETTINGS = load_settings()
ENGINE = BotEngine(BASE_DIR / "data", BASE_DIR / "models" / "intent_classifier.pkl")
USER_STATES: dict[int, UserProfile] = {}


def get_profile(user_id: int) -> UserProfile:
    if user_id not in USER_STATES:
        USER_STATES[user_id] = UserProfile()
    return USER_STATES[user_id]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я бот, который умеет поддержать разговор и помочь подобрать музыкальный инструмент под твои интересы и бюджет."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Напиши мне что угодно: можно просто пообщаться, а можно попросить подобрать гитару, клавиши или другой инструмент. Команда /reset сбрасывает контекст."
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    USER_STATES[update.effective_user.id] = UserProfile()
    await update.message.reply_text("Контекст сброшен. Можем начать заново.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    profile = get_profile(update.effective_user.id)
    reply = ENGINE.reply(update.message.text, profile)
    await update.message.reply_text(reply.text)


def main() -> None:
    if not SETTINGS.telegram_token:
        raise RuntimeError("Не найден TELEGRAM_BOT_TOKEN. Добавь его в переменные окружения или .env.")
    application = Application.builder().token(SETTINGS.telegram_token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()


if __name__ == "__main__":
    main()

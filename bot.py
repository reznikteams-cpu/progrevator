import os
import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pvk_bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-2025-04-14").strip()
PAYMENT_URL = os.getenv("PAYMENT_URL", "").strip()

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

DB_PATH = "pvk_bot.sqlite3"

SYSTEM_INSTRUCTIONS = """
Ты — «Продюсер в кармане» (ПВК).
Ты не ИИ, не бот, не сервис. Ты — внешнее продюсерское мышление рядом с человеком.

Цель: вести пользователя от хаоса к ясности и далее к формату ПВК (постоянное сопровождение).

Стиль:
- 4–7 коротких предложений (строго).
- без инфостиля, без списков, без лекций.
- используй 2–6 точных слов пользователя как «якоря» (цитируй фрагменты).
- ощущение: «меня поняли глубже, чем я сказал(а)».

Запрещено:
- напрямую продавать до шага offer_allowed=true,
- говорить «подписка», «тариф», «ИИ», «сервис»,
- давать инструкции/чеклисты/пошаговые планы.

Каждый ответ:
- отражение состояния,
- 1 сдвиг (инсайт/переформулировка/снятие перегруза),
- завершение мягким вопросом или кнопкой-продолжением.

Твоя задача — двигать по лестнице Бена Ханта:
Unaware → Problem-Aware → Solution-Aware → Product-Aware → Most Aware
"""

START_TEXT = (
    "Здесь не учат и не мотивируют.\n"
    "Здесь помогают *думать* и принимать решения, когда ты перегружен(а).\n\n"
    "Что сейчас больше всего давит?"
)

TOPIC_BUTTONS = [
    ("Перегруз решений", "topic:decisions"),
    ("Деньги / нестабильность", "topic:money"),
    ("Контент и смыслы", "topic:content"),
    ("Хаос в запуске", "topic:launch"),
    ("Не понимаю, куда идти дальше", "topic:direction"),
]

# ---------- Storage (SQLite) ----------

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_state (
            user_id INTEGER PRIMARY KEY,
            state_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn

def load_state(user_id: int) -> Dict[str, Any]:
    conn = db()
    cur = conn.execute("SELECT state_json FROM user_state WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {
            "stage_ben_hunt": "Unaware",
            "topic": None,
            "anchors": [],
            "emotion": None,
            "resistance": None,
            "offer_allowed": False,
            "last_offer_shown": False,
        }
    return json.loads(row[0])

def save_state(user_id: int, state: Dict[str, Any]) -> None:
    conn = db()
    conn.execute(
        "INSERT INTO user_state(user_id, state_json, updated_at) VALUES(?,?,?) "
        "ON CONFLICT(user_id) DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at",
        (user_id, json.dumps(state, ensure_ascii=False), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()

def add_msg(user_id: int, role: str, text: str) -> None:
    conn = db()
    conn.execute(
        "INSERT INTO messages(user_id, role, text, created_at) VALUES(?,?,?,?)",
        (user_id, role, text, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()

def get_recent_dialog(user_id: int, limit: int = 10) -> str:
    conn = db()
    cur = conn.execute(
        "SELECT role, text FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    rows.reverse()
    # Простое склеивание в «диалог»
    parts = []
    for role, text in rows:
        prefix = "Пользователь" if role == "user" else "ПВК"
        parts.append(f"{prefix}: {text}")
    return "\n".join(parts).strip()

# ---------- LLM call (single pass JSON) ----------

JSON_CONTRACT = """
Верни СТРОГО JSON (без markdown и без комментариев) такого вида:
{
  "analysis": {
    "emotion": "…",
    "anchors": ["…", "…"],
    "stage_ben_hunt": "Unaware|Problem-Aware|Solution-Aware|Product-Aware|Most Aware",
    "resistance": "none|trust|money|time|skeptic|overwhelm",
    "warmup_step": "calibrate|anchor_problem|insight|micro_story|pre_sell|offer",
    "offer_allowed": true|false
  },
  "reply": "текст ответа пользователю (4–7 предложений)"
}

Правила:
- reply: 4–7 предложений, без списков.
- Используй 2–6 слов пользователя (anchors) как якоря.
- Если offer_allowed=false — не продавай.
- Если warmup_step=offer — можно мягко предложить «попробовать формат ПВК» и дать вопрос/кнопку.
"""

def safe_json_parse(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None

async def generate_reply(user_id: int, user_text: str) -> str:
    state = load_state(user_id)
    dialog = get_recent_dialog(user_id, limit=10)

    # Контекст, который «видит» модель: состояние + последние реплики
    input_text = (
        f"STATE:\n{json.dumps(state, ensure_ascii=False)}\n\n"
        f"RECENT_DIALOG:\n{dialog}\n\n"
        f"NEW_MESSAGE:\n{user_text}\n\n"
        f"{JSON_CONTRACT}"
    )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        instructions=SYSTEM_INSTRUCTIONS,
        input=input_text,
    )
    out = (resp.output_text or "").strip()

    parsed = safe_json_parse(out)
    if not parsed or "reply" not in parsed:
        # fallback: если модель вдруг сломала JSON — просто вернём текст
        return out[:1500] or "Скажи, что именно сейчас давит сильнее всего — решения, контент или деньги?"

    analysis = parsed.get("analysis", {})
    reply = (parsed.get("reply") or "").strip()

    # Обновим state
    state["emotion"] = analysis.get("emotion", state.get("emotion"))
    state["anchors"] = analysis.get("anchors", state.get("anchors"))
    state["stage_ben_hunt"] = analysis.get("stage_ben_hunt", state.get("stage_ben_hunt"))
    state["resistance"] = analysis.get("resistance", state.get("resistance"))
    state["offer_allowed"] = bool(analysis.get("offer_allowed", state.get("offer_allowed")))

    # Простейшее правило разрешения оффера:
    # если пользователь дошёл до Product-Aware / Most Aware — разрешаем оффер
    if state["stage_ben_hunt"] in ("Product-Aware", "Most Aware"):
        state["offer_allowed"] = True

    save_state(user_id, state)
    return reply[:1500]

# ---------- Telegram handlers ----------

def topic_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(text, callback_data=data)] for text, data in TOPIC_BUTTONS]
    )

def offer_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Да, хочу попробовать формат", callback_data="offer:yes")],
            [InlineKeyboardButton("Хочу понять формат", callback_data="offer:about")],
            [InlineKeyboardButton("Пока сомневаюсь", callback_data="offer:doubt")],
        ]
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    state = load_state(user_id)
    state["topic"] = None
    save_state(user_id, state)

    await update.message.reply_text(START_TEXT, parse_mode="Markdown", reply_markup=topic_keyboard())

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    state = {
        "stage_ben_hunt": "Unaware",
        "topic": None,
        "anchors": [],
        "emotion": None,
        "resistance": None,
        "offer_allowed": False,
        "last_offer_shown": False,
    }
    save_state(user_id, state)
    await update.message.reply_text("Ок. Начнём заново.", reply_markup=topic_keyboard())

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    state = load_state(user_id)
    data = query.data or ""

    if data.startswith("topic:"):
        topic = data.split(":", 1)[1]
        state["topic"] = topic
        # лёгкий буст: считаем, что человек уже Problem-Aware, если выбирает проблему
        state["stage_ben_hunt"] = "Problem-Aware"
        save_state(user_id, state)

        await query.edit_message_text(
            "Ок. Скажи одной фразой: что именно сейчас происходит — как ты это называешь своими словами?"
        )
        return

    if data.startswith("offer:"):
        action = data.split(":", 1)[1]
        if action == "yes":
            if PAYMENT_URL:
                await query.edit_message_text(
                    "Тогда без обещаний и обязательств.\n"
                    "Просто попробуй формат и посмотри: стало ли тебе проще думать и действовать.\n\n"
                    f"Ссылка: {PAYMENT_URL}\n\n"
                    "Хочешь, я помогу сформулировать первую задачу, с которой мы начнём?"
                )
            else:
                await query.edit_message_text(
                    "Ок. Тогда скажи: с какой задачей ты хочешь, чтобы я был рядом уже сегодня?"
                )
            return

        if action == "about":
            await query.edit_message_text(
                "Формат простой: ты не остаёшься одна с решениями.\n"
                "Ты приносишь реальную ситуацию — я помогаю быстро увидеть суть, выбрать ход и снять перегруз.\n"
                "Без лекций. Без морали. По делу.\n\n"
                "Что тебе важнее почувствовать в первые 3 дня — ясность, стабильность или движение?"
            )
            return

        if action == "doubt":
            await query.edit_message_text(
                "Нормально сомневаться.\n"
                "Обычно сомнение — это либо доверие, либо страх «опять не сработает», либо деньги.\n\n"
                "Что ближе: доверие, прошлый опыт или бюджет?"
            )
            return

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    text = update.message.text.strip()

    add_msg(user_id, "user", text)

    # Генерация ответа
    reply = await generate_reply(user_id, text)
    add_msg(user_id, "assistant", reply)

    state = load_state(user_id)

    # Если модель разрешила оффер и мы ещё не показывали кнопки — покажем.
    if state.get("offer_allowed") and not state.get("last_offer_shown"):
        state["last_offer_shown"] = True
        save_state(user_id, state)
        await update.message.reply_text(reply, reply_markup=offer_keyboard())
        return

    await update.message.reply_text(reply)

def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    log.info("Bot started")
    app.run_polling(
    allowed_updates=Update.ALL_TYPES,
    drop_pending_updates=True,
)

from telegram.error import Conflict

async def error_handler(update, context):
    err = context.error
    if isinstance(err, Conflict):
        # Обычно это короткий конфликт при рестарте/деплое.
        return
    log.exception("Unhandled error", exc_info=err)

if __name__ == "__main__":
    main()

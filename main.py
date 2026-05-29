# -*- coding: utf-8 -*-
import os
import io
import asyncio
import logging
import tempfile
import threading
import base64
import json
import urllib.error
import urllib.request

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify

from aiogram import Bot, Dispatcher, F
from aiogram.filters import StateFilter
from aiogram.types import (
    Message, CallbackQuery, PreCheckoutQuery,
    FSInputFile, KeyboardButton, ReplyKeyboardMarkup,
    InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice,
)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

from config import (
    BOT_NAME, BOT_USERNAME, ADMIN_IDS,
    FACE_REPORT_PRICE_RUB, PREMIUM_PLAN_PRICE_RUB,
    FACE_REPORT_PRICE_STARS, PREMIUM_PLAN_PRICE_STARS,
    PRODUCT_FACE_REPORT, PRODUCT_PREMIUM_PLAN,
    YOOKASSA_ENABLED, YOOKASSA_SHOP_ID, YOOKASSA_SECRET_KEY,
    YOOKASSA_RETURN_URL, YOOKASSA_WEBHOOK_PATH, TELEGRAM_STARS_CURRENCY,
)
from analysis import analyze_face
from pdf_builder import create_pdf_report, create_looksmaxxing_pdf
from counters import (
    increment_counter, get_user_count, get_today_count,
    set_selected_product, get_selected_product, clear_selected_product,
    create_order, get_order, update_order_payment, mark_order_paid,
    mark_order_failed, get_active_paid_order, consume_paid_order,
    find_order_by_provider_payment_id,
)


# ================== LOGGING ==================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ================== BOT INIT ==================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN is not set. Ð¡Ð¾Ð·Ð´Ð°Ð¹ .env Ñ BOT_TOKEN=...")

bot = Bot(token=BOT_TOKEN)
dp  = Dispatcher(storage=MemoryStorage())


# ================== FLASK (keep-alive for Render) ==================
app = Flask(__name__)

@app.route("/")
def home():
    return f"{BOT_NAME} bot is running"


@app.route(YOOKASSA_WEBHOOK_PATH, methods=["POST"])
def yookassa_webhook():
    payload = request.get_json(silent=True) or {}
    obj = payload.get("object") or {}
    payment_id = obj.get("id")
    metadata = obj.get("metadata") or {}
    order_id = metadata.get("order_id")
    event = payload.get("event")
    status = obj.get("status")

    logger.info("YooKassa webhook event=%s status=%s payment_id=%s order_id=%s",
                event, status, payment_id, order_id)

    if not order_id and payment_id:
        order = find_order_by_provider_payment_id(payment_id)
        order_id = order["order_id"] if order else None

    if not order_id:
        logger.warning("YooKassa webhook without known order_id: %s", payload)
        return jsonify({"ok": True})

    if event == "payment.succeeded" or status == "succeeded":
        verified = verify_yookassa_payment(payment_id, order_id)
        if verified:
            order = mark_order_paid(order_id, payment_id)
            if order:
                notify_paid_order_sync(order)
    elif event == "payment.canceled" or status == "canceled":
        mark_order_failed(order_id)

    return jsonify({"ok": True})


def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


# ================== FSM ==================
class AnalysisStates(StatesGroup):
    waiting_for_gender = State()
    waiting_for_photo  = State()


# ================== KEYBOARDS ==================
main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="ð Ð Ð°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ°")],
        [KeyboardButton(text="ð Premium Plan")],
        [KeyboardButton(text="Ð§ÑÐ¾ ÑÑÐ¾?"), KeyboardButton(text="Ð¢ÐµÑÐ¿Ð¾Ð´Ð´ÐµÑÐ¶ÐºÐ°")],
    ],
    resize_keyboard=True,
)
gender_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="ð¨ ÐÑÐ¶ÑÐºÐ¾Ð¹"), KeyboardButton(text="ð© ÐÐµÐ½ÑÐºÐ¸Ð¹")],
        [KeyboardButton(text="âï¸ ÐÐ°Ð·Ð°Ð´ Ð² Ð¼ÐµÐ½Ñ")],
    ],
    resize_keyboard=True,
)
cancel_keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="âï¸ ÐÐ°Ð·Ð°Ð´ Ð² Ð¼ÐµÐ½Ñ")]],
    resize_keyboard=True,
)


# ================== PRODUCTS / PAYMENT ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPUFF_GUIDE_PATH = os.path.join(BASE_DIR, "assets", "depuff_guide.pdf")
PROVIDER_YOOKASSA = "yookassa"
PROVIDER_STARS = "telegram_stars"
YOOKASSA_API_URL = "https://api.yookassa.ru/v3"


def payment_mode_text() -> str:
    if not YOOKASSA_ENABLED:
        return "ÐÐ¿Ð»Ð°ÑÐ° ÐºÐ°ÑÑÐ¾Ð¹ Ð¸ Ð¡ÐÐ Ð¿Ð¾ÑÐ²Ð¸ÑÑÑ Ð¿Ð¾ÑÐ»Ðµ Ð´Ð¾Ð±Ð°Ð²Ð»ÐµÐ½Ð¸Ñ ÐºÐ»ÑÑÐµÐ¹ Ð®Kassa. Telegram Stars Ð´Ð¾ÑÑÑÐ¿Ð½Ñ Ð²Ð½ÑÑÑÐ¸ Telegram."
    return "ÐÐ¾ÑÑÑÐ¿Ð½Ð° Ð¾Ð¿Ð»Ð°ÑÐ° ÐºÐ°ÑÑÐ¾Ð¹, Ð¡ÐÐ ÑÐµÑÐµÐ· Ð®Kassa Ð¸Ð»Ð¸ Telegram Stars Ð²Ð½ÑÑÑÐ¸ Telegram."


def product_title(product: str) -> str:
    if product == PRODUCT_PREMIUM_PLAN:
        return "Premium Plan"
    return "Ð Ð°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ°"


def product_price(product: str) -> int:
    if product == PRODUCT_PREMIUM_PLAN:
        return PREMIUM_PLAN_PRICE_RUB
    return FACE_REPORT_PRICE_RUB


def product_stars_price(product: str) -> int:
    if product == PRODUCT_PREMIUM_PLAN:
        return PREMIUM_PLAN_PRICE_STARS
    return FACE_REPORT_PRICE_STARS


def is_admin(user_id: int) -> bool:
    return int(user_id) in ADMIN_IDS


def log_admin_bypass(user_id: int):
    logger.info("[ADMIN_BYPASS] user_id=%s", user_id)


def product_description(product: str) -> str:
    if product == PRODUCT_PREMIUM_PLAN:
        return (
            f"<b>ð Premium Plan</b>\n"
            f"ÐÐ°ÑÑÐ° / Ð¡ÐÐ â {PREMIUM_PLAN_PRICE_RUB} â½  Â·  Stars â {PREMIUM_PLAN_PRICE_STARS} â­\n\n"
            "ÐÐ¾Ð»Ð½ÑÐ¹ ÑÐ°Ð·Ð±Ð¾Ñ Ð¸ Ð»Ð¸ÑÐ½ÑÐ¹ Ð¿Ð»Ð°Ð½: Ð½Ðµ ÑÐ¾Ð»ÑÐºÐ¾ ÑÐ¸ÑÑÑ, Ð½Ð¾ Ð¸ Ð¿Ð¾Ð½ÑÑÐ½ÑÐµ ÑÐ°Ð³Ð¸.\n\n"
            "Ð Ð¿Ð°ÐºÐµÑÐµ:\n"
            "â ÑÐ°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ° Ð½Ð° 23 ÑÑÑÐ°Ð½Ð¸ÑÑ (20 Ð¼ÐµÑÑÐ¸Ðº, ÑÐ¸Ð¼Ð¼ÐµÑÑÐ¸Ñ, tier)\n"
            "â Ð¿ÐµÑÑÐ¾Ð½Ð°Ð»ÑÐ½ÑÐ¹ Ð¿Ð»Ð°Ð½ ÑÐ»ÑÑÑÐµÐ½Ð¸Ñ Ð¾Ð±ÑÐ°Ð·Ð°\n"
            "â ÑÐµÐºÐ¾Ð¼ÐµÐ½Ð´Ð°ÑÐ¸Ð¸: Ð¿ÑÐ¸ÑÑÑÐºÐ°, ÐºÐ¾Ð¶Ð°, Ð±ÑÐ¾Ð²Ð¸, Ð½Ð¸Ð¶Ð½ÑÑ ÑÑÐµÑÑ\n"
            "â Ð¿Ð»Ð°Ð½Ñ Ð½Ð° 7 Ð¸ 30 Ð´Ð½ÐµÐ¹\n"
            "â Ð±Ð¾Ð½ÑÑ: Ð³Ð°Ð¹Ð´ Ð¿Ð¾ ÑÐ½Ð¸Ð¶ÐµÐ½Ð¸Ñ Ð¾ÑÑÑÐ½Ð¾ÑÑÐ¸\n\n"
            f"<i>{payment_mode_text()}</i>\n\n"
            "ÐÐ¾ÑÐ»Ðµ Ð¾Ð¿Ð»Ð°ÑÑ Ð±Ð¾Ñ Ð¾ÑÐºÑÐ¾ÐµÑ Ð²ÑÐ±Ð¾Ñ Ð¿Ð¾Ð»Ð° Ð¸ Ð¿ÑÐ¸ÑÐ¼ ÑÐ¾ÑÐ¾."
        )

    return (
        f"<b>ð Ð Ð°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ°</b>\n"
        f"ÐÐ°ÑÑÐ° / Ð¡ÐÐ â {FACE_REPORT_PRICE_RUB} â½  Â·  Stars â {FACE_REPORT_PRICE_STARS} â­\n\n"
        "Ð¢Ð¾ÑÐ½Ð°Ñ Ð³ÐµÐ¾Ð¼ÐµÑÑÐ¸Ñ Ð²Ð°ÑÐµÐ³Ð¾ Ð»Ð¸ÑÐ° Ð² premium-ÑÐ¾ÑÐ¼Ð°ÑÐµ.\n\n"
        "Ð Ð¾ÑÑÑÑÐµ:\n"
        "â 23 ÑÑÑÐ°Ð½Ð¸ÑÑ, 20 ÐºÐ»ÑÑÐµÐ²ÑÑ Ð¼ÐµÑÑÐ¸Ðº\n"
        "â ÑÐ¸Ð¼Ð¼ÐµÑÑÐ¸Ñ Ð¸ Ð¿ÑÐ¾Ð¿Ð¾ÑÑÐ¸Ð¸ Ñ ÑÐ°Ð·Ð¼ÐµÑÐºÐ¾Ð¹ Ð½Ð° ÑÐ¾ÑÐ¾\n"
        "â Ð¸ÑÐ¾Ð³Ð¾Ð²Ð°Ñ Ð¾ÑÐµÐ½ÐºÐ° Ð³Ð°ÑÐ¼Ð¾Ð½Ð¸Ð¸ Ð¸ tier\n\n"
        f"<i>{payment_mode_text()}</i>\n\n"
        "ÐÐ¾ÑÐ»Ðµ Ð¾Ð¿Ð»Ð°ÑÑ Ð±Ð¾Ñ Ð¾ÑÐºÑÐ¾ÐµÑ Ð²ÑÐ±Ð¾Ñ Ð¿Ð¾Ð»Ð° Ð¸ Ð¿ÑÐ¸ÑÐ¼ ÑÐ¾ÑÐ¾."
    )


def payment_keyboard(product: str) -> InlineKeyboardMarkup:
    rows = []
    if YOOKASSA_ENABLED:
        rows.extend([
            [InlineKeyboardButton(text="ÐÐ¿Ð»Ð°ÑÐ¸ÑÑ ÐºÐ°ÑÑÐ¾Ð¹", callback_data=f"pay:yookassa:bank_card:{product}")],
            [InlineKeyboardButton(text="ÐÐ¿Ð»Ð°ÑÐ¸ÑÑ ÑÐµÑÐµÐ· Ð¡ÐÐ", callback_data=f"pay:yookassa:sbp:{product}")],
        ])
    rows.append([
        InlineKeyboardButton(
            text=f"ÐÐ¿Ð»Ð°ÑÐ¸ÑÑ Stars Â· {product_stars_price(product)} â­",
            callback_data=f"pay:stars:{product}",
        )
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def admin_test_keyboard(product: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="ð  TEST MODE (ADMIN)", callback_data=f"admin_test:{product}")]
    ])


def paid_prompt(product: str) -> str:
    return (
        "<b>ÐÐ¿Ð»Ð°ÑÐ° Ð¿Ð¾Ð»ÑÑÐµÐ½Ð°.</b>\n\n"
        f"Ð¢Ð°ÑÐ¸Ñ: <b>{product_title(product)}</b>\n\n"
        "ÐÑÐ±ÐµÑÐ¸ÑÐµ Ð¿Ð¾Ð» â Ð½Ð¾ÑÐ¼Ñ Ð°Ð½Ð°Ð»Ð¸Ð·Ð° ÑÐ°Ð·Ð»Ð¸ÑÐ°ÑÑÑÑ Ð´Ð»Ñ Ð¼ÑÐ¶ÑÐ¸Ð½ Ð¸ Ð¶ÐµÐ½ÑÐ¸Ð½."
    )


def photo_prompt(product: str, gender: str) -> str:
    word = "Ð¼ÑÐ¶ÑÐºÐ¾Ð¹" if gender == "male" else "Ð¶ÐµÐ½ÑÐºÐ¸Ð¹"
    return (
        f"ÐÐ¾Ð»: <b>{word}</b>  Â·  Ð¢Ð°ÑÐ¸Ñ: <b>{product_title(product)}</b>\n\n"
        "<b>ÐÑÐ¸ÑÐ»Ð¸ÑÐµ ÑÐ¾ÑÐ¾ Ð»Ð¸ÑÐ°.</b>\n\n"
        "ÐÐ»Ñ ÑÐ¾ÑÐ½Ð¾Ð³Ð¾ ÑÐµÐ·ÑÐ»ÑÑÐ°ÑÐ°:\n"
        "â ÑÑÑÐ¾Ð³Ð¾ Ð°Ð½ÑÐ°Ñ, Ð¿ÑÑÐ¼Ð¾ Ð² ÐºÐ°Ð¼ÐµÑÑ\n"
        "â Ð½ÐµÐ¹ÑÑÐ°Ð»ÑÐ½Ð¾Ðµ Ð²ÑÑÐ°Ð¶ÐµÐ½Ð¸Ðµ, ÑÐ¾Ñ Ð·Ð°ÐºÑÑÑ\n"
        "â ÑÐ¾Ð²Ð½ÑÐ¹ ÑÐ²ÐµÑ, Ð±ÐµÐ· ÑÐµÐ½ÐµÐ¹\n"
        "â Ð±ÐµÐ· Ð¾ÑÐºÐ¾Ð², Ð¼Ð°ÑÐºÐ¸ Ð¸ Ð³Ð¾Ð»Ð¾Ð²Ð½Ð¾Ð³Ð¾ ÑÐ±Ð¾ÑÐ°\n"
        "â Ð»Ð¾Ð± Ð¸ Ð±ÑÐ¾Ð²Ð¸ Ð¾ÑÐºÑÑÑÑ"
    )


def gender_reply_markup_dict() -> dict:
    return {
        "keyboard": [
            [{"text": "ð¨ ÐÑÐ¶ÑÐºÐ¾Ð¹"}, {"text": "ð© ÐÐµÐ½ÑÐºÐ¸Ð¹"}],
            [{"text": "âï¸ ÐÐ°Ð·Ð°Ð´ Ð² Ð¼ÐµÐ½Ñ"}],
        ],
        "resize_keyboard": True,
    }


def yookassa_auth_header() -> str:
    raw = f"{YOOKASSA_SHOP_ID}:{YOOKASSA_SECRET_KEY}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def yookassa_request(method: str, path: str, body: dict | None = None,
                     idempotence_key: str | None = None) -> dict:
    if not YOOKASSA_ENABLED:
        raise RuntimeError("Ð®Kassa Ð½Ðµ Ð½Ð°ÑÑÑÐ¾ÐµÐ½Ð°: Ð½ÐµÑ YOOKASSA_SHOP_ID Ð¸Ð»Ð¸ YOOKASSA_SECRET_KEY.")

    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        YOOKASSA_API_URL + path,
        data=data,
        method=method,
        headers={
            "Authorization": yookassa_auth_header(),
            "Content-Type": "application/json",
        },
    )
    if idempotence_key:
        req.add_header("Idempotence-Key", idempotence_key)

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        logger.error("YooKassa HTTP error %s: %s", e.code, detail)
        raise


def create_yookassa_payment(order: dict, method_type: str) -> dict:
    product = order["product"]
    amount = product_price(product)
    body = {
        "amount": {"value": f"{amount:.2f}", "currency": "RUB"},
        "capture": True,
        "payment_method_data": {"type": method_type},
        "confirmation": {"type": "redirect", "return_url": YOOKASSA_RETURN_URL},
        "description": f"{BOT_NAME}: {product_title(product)}",
        "metadata": {
            "order_id": order["order_id"],
            "user_id": order["user_id"],
            "product": product,
        },
    }
    return yookassa_request("POST", "/payments", body, idempotence_key=order["order_id"])


def get_yookassa_payment(payment_id: str) -> dict:
    return yookassa_request("GET", f"/payments/{payment_id}")


def verify_yookassa_payment(payment_id: str | None, order_id: str) -> bool:
    if not payment_id:
        return False
    try:
        payment = get_yookassa_payment(payment_id)
    except Exception:
        logger.exception("Failed to verify YooKassa payment %s", payment_id)
        return False

    metadata = payment.get("metadata") or {}
    if metadata.get("order_id") != order_id:
        logger.warning("YooKassa payment metadata mismatch: %s", payment)
        return False
    return payment.get("status") == "succeeded" and bool(payment.get("paid"))


def send_bot_message_sync(chat_id: int | str, text: str, reply_markup: dict | None = None):
    body = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }
    if reply_markup:
        body["reply_markup"] = reply_markup
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15):
            return
    except Exception:
        logger.exception("Failed to send Telegram payment notification")


def notify_paid_order_sync(order: dict):
    set_selected_product(int(order["user_id"]), order["product"])
    send_bot_message_sync(
        order["user_id"],
        paid_prompt(order["product"]),
        reply_markup=gender_reply_markup_dict(),
    )


# ================== HANDLERS ==================
@dp.message(F.text.in_({"/start", "/help", "âï¸ ÐÐ°Ð·Ð°Ð´ Ð² Ð¼ÐµÐ½Ñ"}))
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    active_order = get_active_paid_order(message.from_user.id)
    if active_order:
        set_selected_product(message.from_user.id, active_order["product"])
        await state.set_state(AnalysisStates.waiting_for_gender)
        await message.answer(
            paid_prompt(active_order["product"]),
            parse_mode="HTML", reply_markup=gender_keyboard,
        )
        return

    clear_selected_product(message.from_user.id)
    uc = get_user_count(message.from_user.id)
    tc = get_today_count()
    await message.answer(
        "<b>HEIM FACE</b>\n"
        + ("ð  <b>TEST MODE (ADMIN)</b>\n\n" if is_admin(message.from_user.id) else "")
        +
        "ÐÐµÐ¾Ð¼ÐµÑÑÐ¸Ñ Ð»Ð¸ÑÐ° Ð² ÑÐ¸ÑÑÐ°Ñ. Ð¡Ð¸Ð¼Ð¼ÐµÑÑÐ¸Ñ, Ð¿ÑÐ¾Ð¿Ð¾ÑÑÐ¸Ð¸ Ð¸ Ð¸ÑÐ¾Ð³Ð¾Ð²ÑÐ¹ tier â "
        "ÑÐ°ÑÑÑÐ¸ÑÐ°Ð½Ð¾ Ð¿Ð¾ Ð°Ð½ÑÑÐ¾Ð¿Ð¾Ð¼ÐµÑÑÐ¸ÑÐµÑÐºÐ¸Ð¼ Ð½Ð¾ÑÐ¼Ð°Ð¼, Ð±ÐµÐ· ÑÑÐ±ÑÐµÐºÑÐ¸Ð²Ð½ÑÑ Ð¾ÑÐµÐ½Ð¾Ðº.\n\n"
        "<b>Ð¢Ð°ÑÐ¸ÑÑ</b>\n"
        f"ð <b>Ð Ð°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ°</b> â {FACE_REPORT_PRICE_RUB} â½\n"
        "PDF Ð½Ð° 23 ÑÑÑÐ°Ð½Ð¸ÑÑ: 20 Ð¼ÐµÑÑÐ¸Ðº, ÑÐ¸Ð¼Ð¼ÐµÑÑÐ¸Ñ, Ð¸ÑÐ¾Ð³Ð¾Ð²ÑÐ¹ tier.\n\n"
        f"ð <b>Premium Plan</b> â {PREMIUM_PLAN_PRICE_RUB} â½\n"
        "ÐÑÑ Ð¸Ð· ÑÐ°Ð·Ð±Ð¾ÑÐ° + Ð¿ÐµÑÑÐ¾Ð½Ð°Ð»ÑÐ½ÑÐ¹ Ð¿Ð»Ð°Ð½ ÑÐ»ÑÑÑÐµÐ½Ð¸Ñ Ð¸ Ð±Ð¾Ð½ÑÑ.\n\n"
        f"<i>ÐÐ¾ÑÐ¾Ð²ÑÑ ÑÐ°Ð·Ð±Ð¾ÑÐ¾Ð²: {uc} Â· ÑÐµÐ³Ð¾Ð´Ð½Ñ: {tc}</i>\n\n"
        "ÐÑÐ±ÐµÑÐ¸ÑÐµ ÑÐ°ÑÐ¸Ñ Ð½Ð¸Ð¶Ðµ.",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


@dp.message(F.text == "Ð§ÑÐ¾ ÑÑÐ¾?")
async def what_is_this(message: Message):
    await message.answer(
        "<b>ÐÐ°ÑÐµÐ¼ Ð½ÑÐ¶ÐµÐ½ ÑÐ°Ð·Ð±Ð¾Ñ</b>\n\n"
        "Ð¡Ð²Ð¾Ñ Ð»Ð¸ÑÐ¾ Ð²Ñ Ð²Ð¸Ð´Ð¸ÑÐµ ÐºÐ°Ð¶Ð´ÑÐ¹ Ð´ÐµÐ½Ñ â Ð¸ Ð¿ÐµÑÐµÑÑÐ°ÑÑÐµ Ð·Ð°Ð¼ÐµÑÐ°ÑÑ Ð´ÐµÑÐ°Ð»Ð¸. "
        "Heim Face Ð¿Ð¾ÐºÐ°Ð·ÑÐ²Ð°ÐµÑ ÐµÐ³Ð¾ ÑÐ°Ðº, ÐºÐ°Ðº Ð²Ð¸Ð´ÑÑ Ð´ÑÑÐ³Ð¸Ðµ: ÑÐµÑÐµÐ· ÑÐ¾ÑÐ½ÑÑ Ð³ÐµÐ¾Ð¼ÐµÑÑÐ¸Ñ.\n\n"
        "ÐÐ»Ð³Ð¾ÑÐ¸ÑÐ¼ Ð½Ð°ÑÐ¾Ð´Ð¸Ñ ÐºÐ»ÑÑÐµÐ²ÑÐµ ÑÐ¾ÑÐºÐ¸ Ð»Ð¸ÑÐ° Ð¸ ÑÑÐ¸ÑÐ°ÐµÑ 20 Ð¿ÑÐ¾Ð¿Ð¾ÑÑÐ¸Ð¹ â "
        "ÑÐ¸Ð¼Ð¼ÐµÑÑÐ¸Ñ, Ð±Ð°Ð»Ð°Ð½Ñ ÑÐµÑÑ, Ð³Ð°ÑÐ¼Ð¾Ð½Ð¸Ñ. ÐÐ°Ð¶Ð´Ð°Ñ Ð¼ÐµÑÑÐ¸ÐºÐ° ÑÑÐ°Ð²Ð½Ð¸Ð²Ð°ÐµÑÑÑ Ñ "
        "Ð°Ð½ÑÑÐ¾Ð¿Ð¾Ð¼ÐµÑÑÐ¸ÑÐµÑÐºÐ¸Ð¼Ð¸ Ð½Ð¾ÑÐ¼Ð°Ð¼Ð¸ Ð´Ð»Ñ Ð²Ð°ÑÐµÐ³Ð¾ Ð¿Ð¾Ð»Ð°. ÐÐ° Ð²ÑÑÐ¾Ð´Ðµ â Ð¿Ð¾Ð½ÑÑÐ½ÑÐ¹ "
        "tier Ð¸ ÑÑÐ½Ð°Ñ ÐºÐ°ÑÑÐ¸Ð½Ð° ÑÐ¸Ð»ÑÐ½ÑÑ ÑÑÐ¾ÑÐ¾Ð½ Ð¸ Ð·Ð¾Ð½ ÑÐ¾ÑÑÐ°.\n\n"
        "ÐÐµÐ· ÑÑÐ±ÑÐµÐºÑÐ¸Ð²Ð½ÑÑ Ð¼Ð½ÐµÐ½Ð¸Ð¹. Ð¢Ð¾Ð»ÑÐºÐ¾ ÑÐ¸ÑÑÑ, ÐºÐ¾ÑÐ¾ÑÑÐµ Ð¼Ð¾Ð¶Ð½Ð¾ Ð¿ÑÐ¾Ð²ÐµÑÐ¸ÑÑ Ð¸ "
        "Ñ ÐºÐ¾ÑÐ¾ÑÑÐ¼Ð¸ Ð¼Ð¾Ð¶Ð½Ð¾ ÑÐ°Ð±Ð¾ÑÐ°ÑÑ.\n\n"
        f"ð <b>Ð Ð°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ°</b> â {FACE_REPORT_PRICE_RUB} â½\n"
        f"ð <b>Premium Plan</b> â {PREMIUM_PLAN_PRICE_RUB} â½ Â· ÑÐ°Ð·Ð±Ð¾Ñ + Ð¿Ð»Ð°Ð½ ÑÐ»ÑÑÑÐµÐ½Ð¸Ñ",
        parse_mode="HTML",
    )


@dp.message(F.text == "Ð¢ÐµÑÐ¿Ð¾Ð´Ð´ÐµÑÐ¶ÐºÐ°")
async def support(message: Message):
    await message.answer(
        "<b>ÐÐ¾Ð´Ð´ÐµÑÐ¶ÐºÐ°</b>\n\n"
        "ÐÐ¾Ð¿ÑÐ¾ÑÑ Ð¿Ð¾ Ð¾Ð¿Ð»Ð°ÑÐµ, Ð´Ð¾ÑÑÑÐ¿Ñ Ð¸Ð»Ð¸ Ð¾ÑÑÑÑÑ â Ð¿Ð¸ÑÐ¸ÑÐµ Ð½Ð°Ð¿ÑÑÐ¼ÑÑ: @aeonin\n\n"
        "ÐÑÐ²ÐµÑÐ°ÐµÐ¼ Ð»Ð¸ÑÐ½Ð¾ Ð¸ ÑÐµÑÐ°ÐµÐ¼ Ð±ÑÑÑÑÐ¾.",
        parse_mode="HTML",
    )


async def start_product_flow(message: Message, state: FSMContext, product: str):
    await state.clear()
    set_selected_product(message.from_user.id, product)
    await state.update_data(product=product)
    if is_admin(message.from_user.id):
        log_admin_bypass(message.from_user.id)
        await message.answer(
            product_description(product) + "\n\nð  <b>TEST MODE (ADMIN)</b>\nÐÐ¿Ð»Ð°ÑÐ° Ð±ÑÐ´ÐµÑ Ð¿ÑÐ¾Ð¿ÑÑÐµÐ½Ð° Ð´Ð»Ñ ÑÐµÑÑÐ¸ÑÐ¾Ð²Ð°Ð½Ð¸Ñ.",
            parse_mode="HTML",
            reply_markup=admin_test_keyboard(product),
        )
        return

    await message.answer(
        product_description(product),
        parse_mode="HTML",
        reply_markup=payment_keyboard(product),
    )


@dp.message(F.text == "ð Ð Ð°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ°")
async def get_face_report(message: Message, state: FSMContext):
    await start_product_flow(message, state, PRODUCT_FACE_REPORT)


@dp.message(F.text == "ð Premium Plan")
async def get_premium_plan(message: Message, state: FSMContext):
    await start_product_flow(message, state, PRODUCT_PREMIUM_PLAN)


@dp.message(F.text == "ð  Ð¥Ð¾ÑÑ Ð¿Ð¾Ð»ÑÑÐ¸ÑÑ ÑÐ²Ð¾Ð¹ ÑÐ°Ð·Ð±Ð¾Ñ")
async def get_legacy_report(message: Message, state: FSMContext):
    await start_product_flow(message, state, PRODUCT_FACE_REPORT)


@dp.callback_query(F.data.startswith("pay:"))
async def payment_callback(callback: CallbackQuery, state: FSMContext):
    parts = callback.data.split(":")
    provider = parts[1]

    if provider == "stars":
        product = parts[2]
        if product not in {PRODUCT_FACE_REPORT, PRODUCT_PREMIUM_PLAN}:
            await callback.answer("ÐÐµÐ¸Ð·Ð²ÐµÑÑÐ½ÑÐ¹ ÑÐ°ÑÐ¸Ñ.", show_alert=True)
            return
        order = create_order(
            callback.from_user.id,
            product,
            PROVIDER_STARS,
            product_stars_price(product),
            TELEGRAM_STARS_CURRENCY,
        )
        await callback.answer()
        await bot.send_invoice(
            chat_id=callback.message.chat.id,
            title=f"{BOT_NAME}: {product_title(product)}",
            description=f"ÐÐ¿Ð»Ð°ÑÐ° ÑÐ°ÑÐ¸ÑÐ° Â«{product_title(product)}Â». ÐÐ¾ÑÐ»Ðµ Ð¾Ð¿Ð»Ð°ÑÑ Ð±Ð¾Ñ Ð¾ÑÐºÑÐ¾ÐµÑ Ð·Ð°Ð³ÑÑÐ·ÐºÑ ÑÐ¾ÑÐ¾.",
            payload=f"stars:{order['order_id']}",
            provider_token="",
            currency=TELEGRAM_STARS_CURRENCY,
            prices=[LabeledPrice(label=product_title(product), amount=product_stars_price(product))],
        )
        return

    if provider == "yookassa":
        method_type = parts[2]
        product = parts[3]
        if method_type not in {"bank_card", "sbp"} or product not in {PRODUCT_FACE_REPORT, PRODUCT_PREMIUM_PLAN}:
            await callback.answer("ÐÐµÐºÐ¾ÑÑÐµÐºÑÐ½ÑÐ¹ ÑÐ¿Ð¾ÑÐ¾Ð± Ð¾Ð¿Ð»Ð°ÑÑ.", show_alert=True)
            return
        if not YOOKASSA_ENABLED:
            await callback.answer("Ð®Kassa ÐµÑÑ Ð½Ðµ Ð½Ð°ÑÑÑÐ¾ÐµÐ½Ð°. ÐÐ¾Ð¶Ð½Ð¾ Ð¾Ð¿Ð»Ð°ÑÐ¸ÑÑ ÑÐµÑÐµÐ· Telegram Stars.", show_alert=True)
            return

        order = create_order(
            callback.from_user.id,
            product,
            PROVIDER_YOOKASSA,
            product_price(product),
            "RUB",
            payment_method=method_type,
        )
        try:
            payment = create_yookassa_payment(order, method_type)
        except Exception:
            mark_order_failed(order["order_id"])
            await callback.answer("ÐÐµ ÑÐ´Ð°Ð»Ð¾ÑÑ ÑÐ¾Ð·Ð´Ð°ÑÑ Ð¿Ð»Ð°ÑÑÐ¶. ÐÐ¾Ð¿ÑÐ¾Ð±ÑÐ¹ÑÐµ Ð¿Ð¾Ð·Ð¶Ðµ Ð¸Ð»Ð¸ Ð²ÑÐ±ÐµÑÐ¸ÑÐµ Stars.", show_alert=True)
            return

        payment_id = payment.get("id")
        confirmation_url = (payment.get("confirmation") or {}).get("confirmation_url")
        update_order_payment(
            order["order_id"],
            provider_payment_id=payment_id,
            confirmation_url=confirmation_url,
            status=payment.get("status"),
        )

        if not confirmation_url:
            await callback.answer("Ð®Kassa Ð½Ðµ Ð²ÐµÑÐ½ÑÐ»Ð° ÑÑÑÐ»ÐºÑ Ð¾Ð¿Ð»Ð°ÑÑ. ÐÐ¾Ð¿ÑÐ¾Ð±ÑÐ¹ÑÐµ Ð¿Ð¾Ð·Ð¶Ðµ.", show_alert=True)
            return

        method_title = "ÐºÐ°ÑÑÐ¾Ð¹" if method_type == "bank_card" else "ÑÐµÑÐµÐ· Ð¡ÐÐ"
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text=f"ÐÐµÑÐµÐ¹ÑÐ¸ Ðº Ð¾Ð¿Ð»Ð°ÑÐµ {method_title}", url=confirmation_url)],
            [InlineKeyboardButton(text="ÐÑÐ¾Ð²ÐµÑÐ¸ÑÑ Ð¾Ð¿Ð»Ð°ÑÑ", callback_data=f"check:{order['order_id']}")],
        ])
        await callback.answer()
        await callback.message.answer(
            f"<b>ÐÐ°ÐºÐ°Ð· ÑÐ¾Ð·Ð´Ð°Ð½.</b>\n\n"
            f"Ð¢Ð°ÑÐ¸Ñ: <b>{product_title(product)}</b>\n"
            f"Ð¡ÑÐ¼Ð¼Ð°: <b>{product_price(product)} â½</b>\n\n"
            "ÐÐ¾ÑÐ»Ðµ ÑÑÐ¿ÐµÑÐ½Ð¾Ð¹ Ð¾Ð¿Ð»Ð°ÑÑ Ð±Ð¾Ñ Ð°Ð²ÑÐ¾Ð¼Ð°ÑÐ¸ÑÐµÑÐºÐ¸ Ð¾ÑÐºÑÐ¾ÐµÑ Ð²ÑÐ±Ð¾Ñ Ð¿Ð¾Ð»Ð°. ÐÑÐ»Ð¸ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ðµ Ð½Ðµ Ð¿ÑÐ¸ÑÐ»Ð¾ ÑÑÐ°Ð·Ñ, Ð½Ð°Ð¶Ð¼Ð¸ÑÐµ Â«ÐÑÐ¾Ð²ÐµÑÐ¸ÑÑ Ð¾Ð¿Ð»Ð°ÑÑÂ».",
            parse_mode="HTML",
            reply_markup=keyboard,
        )


@dp.callback_query(F.data.startswith("admin_test:"))
async def admin_test_callback(callback: CallbackQuery, state: FSMContext):
    if not is_admin(callback.from_user.id):
        await callback.answer("ÐÐµÐ´Ð¾ÑÑÑÐ¿Ð½Ð¾.", show_alert=True)
        return

    product = callback.data.split(":", 1)[1]
    if product not in {PRODUCT_FACE_REPORT, PRODUCT_PREMIUM_PLAN}:
        await callback.answer("ÐÐµÐ¸Ð·Ð²ÐµÑÑÐ½ÑÐ¹ ÑÐ°ÑÐ¸Ñ.", show_alert=True)
        return

    log_admin_bypass(callback.from_user.id)
    set_selected_product(callback.from_user.id, product)
    await state.set_state(AnalysisStates.waiting_for_gender)
    await callback.answer("TEST MODE Ð²ÐºÐ»ÑÑÑÐ½.")
    await callback.message.answer(
        "ð  <b>TEST MODE (ADMIN)</b>\n\n"
        f"Ð¢Ð°ÑÐ¸Ñ: <b>{product_title(product)}</b>\n"
        "ÐÐ¿Ð»Ð°ÑÐ° Ð¿ÑÐ¾Ð¿ÑÑÐµÐ½Ð°. ÐÑÐ±ÐµÑÐ¸ÑÐµ Ð¿Ð¾Ð» Ð´Ð»Ñ ÑÐµÑÑÐ¾Ð²Ð¾Ð³Ð¾ ÑÐ°Ð·Ð±Ð¾ÑÐ°.",
        parse_mode="HTML",
        reply_markup=gender_keyboard,
    )


@dp.callback_query(F.data.startswith("check:"))
async def check_payment_callback(callback: CallbackQuery, state: FSMContext):
    order_id = callback.data.split(":", 1)[1]
    order = get_order(order_id)
    if not order or order.get("user_id") != str(callback.from_user.id):
        await callback.answer("ÐÐ°ÐºÐ°Ð· Ð½Ðµ Ð½Ð°Ð¹Ð´ÐµÐ½.", show_alert=True)
        return

    if order.get("status") == "paid":
        await state.set_state(AnalysisStates.waiting_for_gender)
        set_selected_product(callback.from_user.id, order["product"])
        await callback.answer("ÐÐ¿Ð»Ð°ÑÐ° ÑÐ¶Ðµ Ð¿Ð¾Ð´ÑÐ²ÐµÑÐ¶Ð´ÐµÐ½Ð°.")
        await callback.message.answer(paid_prompt(order["product"]), parse_mode="HTML", reply_markup=gender_keyboard)
        return

    payment_id = order.get("provider_payment_id")
    if order.get("provider") == PROVIDER_YOOKASSA and verify_yookassa_payment(payment_id, order_id):
        order = mark_order_paid(order_id, payment_id)
        await state.set_state(AnalysisStates.waiting_for_gender)
        await callback.answer("ÐÐ¿Ð»Ð°ÑÐ° Ð¿Ð¾Ð´ÑÐ²ÐµÑÐ¶Ð´ÐµÐ½Ð°.")
        await callback.message.answer(paid_prompt(order["product"]), parse_mode="HTML", reply_markup=gender_keyboard)
        return

    await callback.answer("ÐÐ¿Ð»Ð°ÑÐ° Ð¿Ð¾ÐºÐ° Ð½Ðµ Ð¿Ð¾Ð´ÑÐ²ÐµÑÐ¶Ð´ÐµÐ½Ð°.", show_alert=True)


@dp.pre_checkout_query()
async def pre_checkout(pre_checkout_query: PreCheckoutQuery):
    payload = pre_checkout_query.invoice_payload or ""
    if not payload.startswith("stars:"):
        await pre_checkout_query.answer(ok=False, error_message="ÐÐµÐºÐ¾ÑÑÐµÐºÑÐ½ÑÐ¹ Ð¿Ð»Ð°ÑÑÐ¶.")
        return

    order = get_order(payload.split(":", 1)[1])
    if not order or order.get("provider") != PROVIDER_STARS or order.get("status") != "pending":
        await pre_checkout_query.answer(ok=False, error_message="ÐÐ°ÐºÐ°Ð· Ð½Ðµ Ð½Ð°Ð¹Ð´ÐµÐ½ Ð¸Ð»Ð¸ ÑÐ¶Ðµ Ð¾Ð±ÑÐ°Ð±Ð¾ÑÐ°Ð½.")
        return

    expected_amount = product_stars_price(order["product"])
    if pre_checkout_query.currency != TELEGRAM_STARS_CURRENCY or pre_checkout_query.total_amount != expected_amount:
        await pre_checkout_query.answer(ok=False, error_message="Ð¡ÑÐ¼Ð¼Ð° Ð¿Ð»Ð°ÑÐµÐ¶Ð° Ð½Ðµ ÑÐ¾Ð²Ð¿Ð°Ð´Ð°ÐµÑ Ñ Ð·Ð°ÐºÐ°Ð·Ð¾Ð¼.")
        return

    await pre_checkout_query.answer(ok=True)


@dp.message(F.successful_payment)
async def successful_payment(message: Message, state: FSMContext):
    payment = message.successful_payment
    payload = payment.invoice_payload or ""
    if not payload.startswith("stars:"):
        return

    order_id = payload.split(":", 1)[1]
    order = mark_order_paid(order_id, payment.telegram_payment_charge_id)
    if not order:
        await message.answer("ÐÐ¿Ð»Ð°ÑÐ° Ð¿Ð¾Ð»ÑÑÐµÐ½Ð°, Ð½Ð¾ Ð·Ð°ÐºÐ°Ð· Ð½Ðµ Ð½Ð°Ð¹Ð´ÐµÐ½. ÐÐ°Ð¿Ð¸ÑÐ¸ÑÐµ Ð² ÑÐµÑÐ¿Ð¾Ð´Ð´ÐµÑÐ¶ÐºÑ.")
        return

    set_selected_product(message.from_user.id, order["product"])
    await state.set_state(AnalysisStates.waiting_for_gender)
    await message.answer(paid_prompt(order["product"]), parse_mode="HTML", reply_markup=gender_keyboard)


@dp.message(AnalysisStates.waiting_for_gender, F.text.in_({"ð¨ ÐÑÐ¶ÑÐºÐ¾Ð¹", "ð© ÐÐµÐ½ÑÐºÐ¸Ð¹"}))
async def choose_gender(message: Message, state: FSMContext):
    active_order = get_active_paid_order(message.from_user.id)
    product = None
    order_id = None
    if active_order:
        product = active_order["product"]
        order_id = active_order["order_id"]
    elif is_admin(message.from_user.id):
        data = await state.get_data()
        product = data.get("product") or get_selected_product(message.from_user.id) or PRODUCT_FACE_REPORT
        log_admin_bypass(message.from_user.id)
    else:
        await state.clear()
        await message.answer(
            "Ð§ÑÐ¾Ð±Ñ Ð½Ð°ÑÐ°ÑÑ, Ð²ÑÐ±ÐµÑÐ¸ÑÐµ ÑÐ°ÑÐ¸Ñ Ð² Ð¼ÐµÐ½Ñ: <b>ð Ð Ð°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ°</b> Ð¸Ð»Ð¸ <b>ð Premium Plan</b>.",
            parse_mode="HTML", reply_markup=main_keyboard,
        )
        return

    set_selected_product(message.from_user.id, product)
    gender = "male" if "ÐÑÐ¶ÑÐºÐ¾Ð¹" in message.text else "female"
    await state.update_data(gender=gender, product=product, order_id=order_id)
    await state.set_state(AnalysisStates.waiting_for_photo)
    await message.answer(
        photo_prompt(product, gender),
        parse_mode="HTML", reply_markup=cancel_keyboard,
    )


@dp.message(AnalysisStates.waiting_for_gender)
async def wrong_gender(message: Message):
    await message.answer("ÐÑÐ±ÐµÑÐ¸ÑÐµ Ð¿Ð¾Ð» ÐºÐ½Ð¾Ð¿ÐºÐ¾Ð¹ Ð½Ð¸Ð¶Ðµ.",
                         reply_markup=gender_keyboard)


@dp.message(StateFilter(None), F.text.in_({"ð¨ ÐÑÐ¶ÑÐºÐ¾Ð¹", "ð© ÐÐµÐ½ÑÐºÐ¸Ð¹"}))
async def choose_gender_after_webhook(message: Message, state: FSMContext):
    active_order = get_active_paid_order(message.from_user.id)
    product = None
    order_id = None
    if active_order:
        product = active_order["product"]
        order_id = active_order["order_id"]
    elif is_admin(message.from_user.id):
        product = get_selected_product(message.from_user.id) or PRODUCT_FACE_REPORT
        log_admin_bypass(message.from_user.id)
    else:
        await message.answer(
            "Ð§ÑÐ¾Ð±Ñ Ð½Ð°ÑÐ°ÑÑ, Ð²ÑÐ±ÐµÑÐ¸ÑÐµ ÑÐ°ÑÐ¸Ñ Ð² Ð¼ÐµÐ½Ñ: <b>ð Ð Ð°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ°</b> Ð¸Ð»Ð¸ <b>ð Premium Plan</b>.",
            parse_mode="HTML", reply_markup=main_keyboard,
        )
        return

    set_selected_product(message.from_user.id, product)
    gender = "male" if "ÐÑÐ¶ÑÐºÐ¾Ð¹" in message.text else "female"
    await state.update_data(gender=gender, product=product, order_id=order_id)
    await state.set_state(AnalysisStates.waiting_for_photo)
    await message.answer(
        photo_prompt(product, gender),
        parse_mode="HTML", reply_markup=cancel_keyboard,
    )


async def process_image(message: Message, image_bytes: bytes, state: FSMContext):
    data   = await state.get_data()
    gender = data.get("gender", "male")
    active_order = get_active_paid_order(message.from_user.id)
    product = None
    order_id = None

    if active_order:
        product = active_order["product"]
        order_id = active_order["order_id"]
    elif is_admin(message.from_user.id):
        product = data.get("product") or get_selected_product(message.from_user.id) or PRODUCT_FACE_REPORT
        log_admin_bypass(message.from_user.id)
    else:
        await state.clear()
        clear_selected_product(message.from_user.id)
        await message.answer(
            "Ð¡Ð½Ð°ÑÐ°Ð»Ð° Ð²ÑÐ±ÐµÑÐ¸ÑÐµ ÑÐ°ÑÐ¸Ñ Ð² Ð¼ÐµÐ½Ñ â Ð¿Ð¾ÑÐ»Ðµ Ð¾Ð¿Ð»Ð°ÑÑ Ñ Ð¿ÑÐ¸Ð¼Ñ ÑÐ¾ÑÐ¾ Ð¸ ÑÐ¾Ð±ÐµÑÑ PDF.",
            reply_markup=main_keyboard,
        )
        return

    await message.answer(
        f"<b>ÐÐ½Ð°Ð»Ð¸Ð·Ð¸ÑÑÑ Ð»Ð¸ÑÐ¾</b> Â· ÑÐ°ÑÐ¸Ñ Â«{product_title(product)}Â»\n\n"
        "ÐÐ±ÑÑÐ½Ð¾ Ð·Ð°Ð½Ð¸Ð¼Ð°ÐµÑ Ð´Ð¾ 30 ÑÐµÐºÑÐ½Ð´.",
        parse_mode="HTML",
    )

    analysis, error = analyze_face(image_bytes, gender)
    if error:
        await message.answer(f"{error}\n\nÐÐ¾Ð¿ÑÐ¾Ð±ÑÐ¹ÑÐµ Ð´ÑÑÐ³Ð¾Ðµ ÑÐ¾ÑÐ¾.",
                             reply_markup=cancel_keyboard)
        return

    temp_paths = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            face_pdf_path = tmp.name
        temp_paths.append(face_pdf_path)
        create_pdf_report(image_bytes, analysis, gender, face_pdf_path)

        looks_pdf_path = None
        if product == PRODUCT_PREMIUM_PLAN:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                looks_pdf_path = tmp.name
            temp_paths.append(looks_pdf_path)
            create_looksmaxxing_pdf(image_bytes, analysis, gender, looks_pdf_path)

        tier = analysis["tier"]
        await message.answer(
            "<b>Ð Ð°Ð·Ð±Ð¾Ñ Ð³Ð¾ÑÐ¾Ð².</b>\n\n"
            f"ÐÑÐµÐ½ÐºÐ° Ð³Ð°ÑÐ¼Ð¾Ð½Ð¸Ð¸: <b>{analysis['score']:.2f} / 10</b>\n"
            f"Tier: <b>{tier['abbr']} Â· {tier['name']}</b>\n"
            f"Ð£ÑÐ¾Ð²ÐµÐ½Ñ: {analysis['level']}\n\n"
            f"Ð¢Ð°ÑÐ¸Ñ: {product_title(product)}",
            parse_mode="HTML",
        )

        if product == PRODUCT_PREMIUM_PLAN:
            await message.answer_document(
                FSInputFile(face_pdf_path, filename=f"Ð Ð°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ° {BOT_NAME}.pdf"),
            )
            await message.answer_document(
                FSInputFile(looks_pdf_path, filename="ÐÑÐºÑÐ¼Ð°ÐºÑÐ¸Ð½Ð³-Ð¿Ð»Ð°Ð½ Heim Face.pdf"),
            )
            if os.path.exists(DEPUFF_GUIDE_PATH):
                await message.answer_document(
                    FSInputFile(DEPUFF_GUIDE_PATH, filename="ÐÐ¾Ð½ÑÑ Ð¿Ð¾ Ð¾ÑÑÑÐ½Ð¾ÑÑÐ¸ Heim Face.pdf"),
                    reply_markup=main_keyboard,
                )
            else:
                logger.error("Premium depuff guide is missing: %s", DEPUFF_GUIDE_PATH)
                await message.answer(
                    "ÐÑÐ½Ð¾Ð²Ð½ÑÐµ PDF Ð¾ÑÐ¿ÑÐ°Ð²Ð»ÐµÐ½Ñ. ÐÐ¾Ð½ÑÑÐ½ÑÐ¹ ÑÐ°Ð¹Ð» Ð¿Ð¾ Ð¾ÑÑÑÐ½Ð¾ÑÑÐ¸ Ð½Ðµ Ð½Ð°Ð¹Ð´ÐµÐ½ Ð½Ð° ÑÐµÑÐ²ÐµÑÐµ.",
                    reply_markup=main_keyboard,
                )
        else:
            await message.answer_document(
                FSInputFile(face_pdf_path, filename=f"Ð Ð°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ° {BOT_NAME}.pdf"),
                reply_markup=main_keyboard,
            )

        increment_counter(message.from_user.id, product)
        if order_id:
            consume_paid_order(order_id)
        clear_selected_product(message.from_user.id)
        await state.clear()

    except Exception as e:
        logger.exception("PDF generation failed")
        await message.answer(
            f"ÐÑÐ¸Ð±ÐºÐ° Ð¿ÑÐ¸ ÑÐ¾Ð·Ð´Ð°Ð½Ð¸Ð¸ Ð¾ÑÑÑÑÐ°: {str(e)[:200]}\n\nÐÐ¾Ð¿ÑÐ¾Ð±ÑÐ¹ÑÐµ ÐµÑÑ ÑÐ°Ð·.",
            reply_markup=cancel_keyboard,
        )
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass


@dp.message(AnalysisStates.waiting_for_photo, F.photo)
async def handle_photo_state(message: Message, state: FSMContext):
    photo = message.photo[-1]
    file  = await bot.get_file(photo.file_id)
    buf   = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    await process_image(message, buf.getvalue(), state)


@dp.message(AnalysisStates.waiting_for_photo, F.document)
async def handle_doc_state(message: Message, state: FSMContext):
    doc = message.document
    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await message.answer("ÐÑÐ¶Ð½Ð¾ ÑÐ¾ÑÐ¾ Ð»Ð¸ÑÐ°. ÐÑÐ¸ÑÐ»Ð¸ÑÐµ Ð¸Ð·Ð¾Ð±ÑÐ°Ð¶ÐµÐ½Ð¸Ðµ.")
        return
    file = await bot.get_file(doc.file_id)
    buf  = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    await process_image(message, buf.getvalue(), state)


@dp.message(AnalysisStates.waiting_for_photo)
async def wrong_state_photo(message: Message):
    await message.answer("ÐÐ´Ñ ÑÐ¾ÑÐ¾ Ð»Ð¸ÑÐ° Ð´Ð»Ñ Ð²ÑÐ±ÑÐ°Ð½Ð½Ð¾Ð³Ð¾ ÑÐ°ÑÐ¸ÑÐ°.", reply_markup=cancel_keyboard)


@dp.message(F.photo)
async def photo_no_state(message: Message):
    active_order = get_active_paid_order(message.from_user.id)
    if active_order:
        set_selected_product(message.from_user.id, active_order["product"])
        await message.answer(paid_prompt(active_order["product"]), parse_mode="HTML", reply_markup=gender_keyboard)
        return
    if is_admin(message.from_user.id):
        log_admin_bypass(message.from_user.id)
        await message.answer(
            "ð  <b>TEST MODE (ADMIN)</b>\n\nÐ¡Ð½Ð°ÑÐ°Ð»Ð° Ð²ÑÐ±ÐµÑÐ¸ÑÐµ ÑÐµÑÑÐ¸ÑÑÐµÐ¼ÑÐ¹ ÑÐ°ÑÐ¸Ñ: <b>ð Ð Ð°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ°</b> Ð¸Ð»Ð¸ <b>ð Premium Plan</b>.",
            parse_mode="HTML", reply_markup=main_keyboard,
        )
        return
    await message.answer(
        "Ð§ÑÐ¾Ð±Ñ Ð½Ð°ÑÐ°ÑÑ, Ð²ÑÐ±ÐµÑÐ¸ÑÐµ ÑÐ°ÑÐ¸Ñ Ð² Ð¼ÐµÐ½Ñ: <b>ð Ð Ð°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ°</b> Ð¸Ð»Ð¸ <b>ð Premium Plan</b>.",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


@dp.message(F.document)
async def doc_no_state(message: Message):
    active_order = get_active_paid_order(message.from_user.id)
    if active_order:
        set_selected_product(message.from_user.id, active_order["product"])
        await message.answer(paid_prompt(active_order["product"]), parse_mode="HTML", reply_markup=gender_keyboard)
        return
    if is_admin(message.from_user.id):
        log_admin_bypass(message.from_user.id)
        await message.answer(
            "ð  <b>TEST MODE (ADMIN)</b>\n\nÐ¡Ð½Ð°ÑÐ°Ð»Ð° Ð²ÑÐ±ÐµÑÐ¸ÑÐµ ÑÐµÑÑÐ¸ÑÑÐµÐ¼ÑÐ¹ ÑÐ°ÑÐ¸Ñ: <b>ð Ð Ð°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ°</b> Ð¸Ð»Ð¸ <b>ð Premium Plan</b>.",
            parse_mode="HTML", reply_markup=main_keyboard,
        )
        return
    await message.answer(
        "Ð§ÑÐ¾Ð±Ñ Ð½Ð°ÑÐ°ÑÑ, Ð²ÑÐ±ÐµÑÐ¸ÑÐµ ÑÐ°ÑÐ¸Ñ Ð² Ð¼ÐµÐ½Ñ: <b>ð Ð Ð°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ°</b> Ð¸Ð»Ð¸ <b>ð Premium Plan</b>.",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


@dp.message()
async def fallback(message: Message):
    await message.answer(
        "ÐÑÐ±ÐµÑÐ¸ÑÐµ ÑÐ°ÑÐ¸Ñ Ð² Ð¼ÐµÐ½Ñ, ÑÑÐ¾Ð±Ñ Ð½Ð°ÑÐ°ÑÑ: <b>ð Ð Ð°Ð·Ð±Ð¾Ñ Ð»Ð¸ÑÐ°</b> Ð¸Ð»Ð¸ <b>ð Premium Plan</b>.",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


# ================== ENTRY POINT ==================
async def main():
    logger.info(f"{BOT_NAME} starting...")
    threading.Thread(target=run_web, daemon=True).start()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

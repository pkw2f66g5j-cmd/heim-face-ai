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
    raise ValueError("BOT_TOKEN is not set. Создай .env с BOT_TOKEN=...")

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
        [KeyboardButton(text="💎 Разбор лица")],
        [KeyboardButton(text="👑 Premium Plan")],
        [KeyboardButton(text="Что это?"), KeyboardButton(text="Техподдержка")],
    ],
    resize_keyboard=True,
)
gender_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="👨 Мужской"), KeyboardButton(text="👩 Женский")],
        [KeyboardButton(text="◀️ Назад в меню")],
    ],
    resize_keyboard=True,
)
cancel_keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="◀️ Назад в меню")]],
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
        return "Оплата картой и СБП появится после добавления ключей ЮKassa. Telegram Stars доступны внутри Telegram."
    return "Доступна оплата картой, СБП через ЮKassa или Telegram Stars внутри Telegram."


def product_title(product: str) -> str:
    if product == PRODUCT_PREMIUM_PLAN:
        return "Premium Plan"
    return "Разбор лица"


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
            f"<b>👑 Premium Plan</b>\n"
            f"Карта / СБП — {PREMIUM_PLAN_PRICE_RUB} ₽  ·  Stars — {PREMIUM_PLAN_PRICE_STARS} ⭐\n\n"
            "Полный разбор и личный план: не только цифры, но и понятные шаги.\n\n"
            "В пакете:\n"
            "— разбор лица на 23 страницы (20 метрик, симметрия, tier)\n"
            "— персональный план улучшения образа\n"
            "— рекомендации: причёска, кожа, брови, нижняя треть\n"
            "— планы на 7 и 30 дней\n"
            "— бонус: гайд по снижению отёчности\n\n"
            f"<i>{payment_mode_text()}</i>\n\n"
            "После оплаты бот откроет выбор пола и приём фото."
        )

    return (
        f"<b>💎 Разбор лица</b>\n"
        f"Карта / СБП — {FACE_REPORT_PRICE_RUB} ₽  ·  Stars — {FACE_REPORT_PRICE_STARS} ⭐\n\n"
        "Точная геометрия вашего лица в premium-формате.\n\n"
        "В отчёте:\n"
        "— 23 страницы, 20 ключевых метрик\n"
        "— симметрия и пропорции с разметкой на фото\n"
        "— итоговая оценка гармонии и tier\n\n"
        f"<i>{payment_mode_text()}</i>\n\n"
        "После оплаты бот откроет выбор пола и приём фото."
    )


def payment_keyboard(product: str) -> InlineKeyboardMarkup:
    rows = []
    if YOOKASSA_ENABLED:
        rows.extend([
            [InlineKeyboardButton(text="Оплатить картой", callback_data=f"pay:yookassa:bank_card:{product}")],
            [InlineKeyboardButton(text="Оплатить через СБП", callback_data=f"pay:yookassa:sbp:{product}")],
        ])
    rows.append([
        InlineKeyboardButton(
            text=f"Оплатить Stars · {product_stars_price(product)} ⭐",
            callback_data=f"pay:stars:{product}",
        )
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def admin_test_keyboard(product: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🛠 TEST MODE (ADMIN)", callback_data=f"admin_test:{product}")]
    ])


def paid_prompt(product: str) -> str:
    return (
        "<b>Оплата получена.</b>\n\n"
        f"Тариф: <b>{product_title(product)}</b>\n\n"
        "Выберите пол — нормы анализа различаются для мужчин и женщин."
    )


def photo_prompt(product: str, gender: str) -> str:
    word = "мужской" if gender == "male" else "женский"
    return (
        f"Пол: <b>{word}</b>  ·  Тариф: <b>{product_title(product)}</b>\n\n"
        "<b>Пришлите фото лица.</b>\n\n"
        "Для точного результата:\n"
        "— строго анфас, прямо в камеру\n"
        "— нейтральное выражение, рот закрыт\n"
        "— ровный свет, без теней\n"
        "— без очков, маски и головного убора\n"
        "— лоб и брови открыты"
    )


def gender_reply_markup_dict() -> dict:
    return {
        "keyboard": [
            [{"text": "👨 Мужской"}, {"text": "👩 Женский"}],
            [{"text": "◀️ Назад в меню"}],
        ],
        "resize_keyboard": True,
    }


def yookassa_auth_header() -> str:
    raw = f"{YOOKASSA_SHOP_ID}:{YOOKASSA_SECRET_KEY}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def yookassa_request(method: str, path: str, body: dict | None = None,
                     idempotence_key: str | None = None) -> dict:
    if not YOOKASSA_ENABLED:
        raise RuntimeError("ЮKassa не настроена: нет YOOKASSA_SHOP_ID или YOOKASSA_SECRET_KEY.")

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
@dp.message(F.text.in_({"/start", "/help", "◀️ Назад в меню"}))
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
        + ("🛠 <b>TEST MODE (ADMIN)</b>\n\n" if is_admin(message.from_user.id) else "")
        +
        "Геометрия лица в цифрах. Симметрия, пропорции и итоговый tier — "
        "рассчитано по антропометрическим нормам, без субъективных оценок.\n\n"
        "<b>Тарифы</b>\n"
        f"💎 <b>Разбор лица</b> — {FACE_REPORT_PRICE_RUB} ₽\n"
        "PDF на 23 страницы: 20 метрик, симметрия, итоговый tier.\n\n"
        f"👑 <b>Premium Plan</b> — {PREMIUM_PLAN_PRICE_RUB} ₽\n"
        "Всё из разбора + персональный план улучшения и бонус.\n\n"
        f"<i>Готовых разборов: {uc} · сегодня: {tc}</i>\n\n"
        "Выберите тариф ниже.",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


@dp.message(F.text == "Что это?")
async def what_is_this(message: Message):
    await message.answer(
        "<b>Зачем нужен разбор</b>\n\n"
        "Своё лицо вы видите каждый день — и перестаёте замечать детали. "
        "Heim Face показывает его так, как видят другие: через точную геометрию.\n\n"
        "Алгоритм находит ключевые точки лица и считает 20 пропорций — "
        "симметрию, баланс черт, гармонию. Каждая метрика сравнивается с "
        "антропометрическими нормами для вашего пола. На выходе — понятный "
        "tier и ясная картина сильных сторон и зон роста.\n\n"
        "Без субъективных мнений. Только цифры, которые можно проверить и "
        "с которыми можно работать.\n\n"
        f"💎 <b>Разбор лица</b> — {FACE_REPORT_PRICE_RUB} ₽\n"
        f"👑 <b>Premium Plan</b> — {PREMIUM_PLAN_PRICE_RUB} ₽ · разбор + план улучшения",
        parse_mode="HTML",
    )


@dp.message(F.text == "Техподдержка")
async def support(message: Message):
    await message.answer(
        "<b>Поддержка</b>\n\n"
        "Вопросы по оплате, доступу или отчёту — пишите напрямую: @aeonin\n\n"
        "Отвечаем лично и решаем быстро.",
        parse_mode="HTML",
    )


async def start_product_flow(message: Message, state: FSMContext, product: str):
    await state.clear()
    set_selected_product(message.from_user.id, product)
    await state.update_data(product=product)
    if is_admin(message.from_user.id):
        log_admin_bypass(message.from_user.id)
        await message.answer(
            product_description(product) + "\n\n🛠 <b>TEST MODE (ADMIN)</b>\nОплата будет пропущена для тестирования.",
            parse_mode="HTML",
            reply_markup=admin_test_keyboard(product),
        )
        return

    await message.answer(
        product_description(product),
        parse_mode="HTML",
        reply_markup=payment_keyboard(product),
    )


@dp.message(F.text == "💎 Разбор лица")
async def get_face_report(message: Message, state: FSMContext):
    await start_product_flow(message, state, PRODUCT_FACE_REPORT)


@dp.message(F.text == "👑 Premium Plan")
async def get_premium_plan(message: Message, state: FSMContext):
    await start_product_flow(message, state, PRODUCT_PREMIUM_PLAN)


@dp.message(F.text == "💠 Хочу получить свой разбор")
async def get_legacy_report(message: Message, state: FSMContext):
    await start_product_flow(message, state, PRODUCT_FACE_REPORT)


@dp.callback_query(F.data.startswith("pay:"))
async def payment_callback(callback: CallbackQuery, state: FSMContext):
    parts = callback.data.split(":")
    provider = parts[1]

    if provider == "stars":
        product = parts[2]
        if product not in {PRODUCT_FACE_REPORT, PRODUCT_PREMIUM_PLAN}:
            await callback.answer("Неизвестный тариф.", show_alert=True)
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
            description=f"Оплата тарифа «{product_title(product)}». После оплаты бот откроет загрузку фото.",
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
            await callback.answer("Некорректный способ оплаты.", show_alert=True)
            return
        if not YOOKASSA_ENABLED:
            await callback.answer("ЮKassa ещё не настроена. Можно оплатить через Telegram Stars.", show_alert=True)
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
            await callback.answer("Не удалось создать платёж. Попробуйте позже или выберите Stars.", show_alert=True)
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
            await callback.answer("ЮKassa не вернула ссылку оплаты. Попробуйте позже.", show_alert=True)
            return

        method_title = "картой" if method_type == "bank_card" else "через СБП"
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text=f"Перейти к оплате {method_title}", url=confirmation_url)],
            [InlineKeyboardButton(text="Проверить оплату", callback_data=f"check:{order['order_id']}")],
        ])
        await callback.answer()
        await callback.message.answer(
            f"<b>Заказ создан.</b>\n\n"
            f"Тариф: <b>{product_title(product)}</b>\n"
            f"Сумма: <b>{product_price(product)} ₽</b>\n\n"
            "После успешной оплаты бот автоматически откроет выбор пола. Если сообщение не пришло сразу, нажмите «Проверить оплату».",
            parse_mode="HTML",
            reply_markup=keyboard,
        )


@dp.callback_query(F.data.startswith("admin_test:"))
async def admin_test_callback(callback: CallbackQuery, state: FSMContext):
    if not is_admin(callback.from_user.id):
        await callback.answer("Недоступно.", show_alert=True)
        return

    product = callback.data.split(":", 1)[1]
    if product not in {PRODUCT_FACE_REPORT, PRODUCT_PREMIUM_PLAN}:
        await callback.answer("Неизвестный тариф.", show_alert=True)
        return

    log_admin_bypass(callback.from_user.id)
    set_selected_product(callback.from_user.id, product)
    await state.set_state(AnalysisStates.waiting_for_gender)
    await callback.answer("TEST MODE включён.")
    await callback.message.answer(
        "🛠 <b>TEST MODE (ADMIN)</b>\n\n"
        f"Тариф: <b>{product_title(product)}</b>\n"
        "Оплата пропущена. Выберите пол для тестового разбора.",
        parse_mode="HTML",
        reply_markup=gender_keyboard,
    )


@dp.callback_query(F.data.startswith("check:"))
async def check_payment_callback(callback: CallbackQuery, state: FSMContext):
    order_id = callback.data.split(":", 1)[1]
    order = get_order(order_id)
    if not order or order.get("user_id") != str(callback.from_user.id):
        await callback.answer("Заказ не найден.", show_alert=True)
        return

    if order.get("status") == "paid":
        await state.set_state(AnalysisStates.waiting_for_gender)
        set_selected_product(callback.from_user.id, order["product"])
        await callback.answer("Оплата уже подтверждена.")
        await callback.message.answer(paid_prompt(order["product"]), parse_mode="HTML", reply_markup=gender_keyboard)
        return

    payment_id = order.get("provider_payment_id")
    if order.get("provider") == PROVIDER_YOOKASSA and verify_yookassa_payment(payment_id, order_id):
        order = mark_order_paid(order_id, payment_id)
        await state.set_state(AnalysisStates.waiting_for_gender)
        await callback.answer("Оплата подтверждена.")
        await callback.message.answer(paid_prompt(order["product"]), parse_mode="HTML", reply_markup=gender_keyboard)
        return

    await callback.answer("Оплата пока не подтверждена.", show_alert=True)


@dp.pre_checkout_query()
async def pre_checkout(pre_checkout_query: PreCheckoutQuery):
    payload = pre_checkout_query.invoice_payload or ""
    if not payload.startswith("stars:"):
        await pre_checkout_query.answer(ok=False, error_message="Некорректный платёж.")
        return

    order = get_order(payload.split(":", 1)[1])
    if not order or order.get("provider") != PROVIDER_STARS or order.get("status") != "pending":
        await pre_checkout_query.answer(ok=False, error_message="Заказ не найден или уже обработан.")
        return

    expected_amount = product_stars_price(order["product"])
    if pre_checkout_query.currency != TELEGRAM_STARS_CURRENCY or pre_checkout_query.total_amount != expected_amount:
        await pre_checkout_query.answer(ok=False, error_message="Сумма платежа не совпадает с заказом.")
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
        await message.answer("Оплата получена, но заказ не найден. Напишите в техподдержку.")
        return

    set_selected_product(message.from_user.id, order["product"])
    await state.set_state(AnalysisStates.waiting_for_gender)
    await message.answer(paid_prompt(order["product"]), parse_mode="HTML", reply_markup=gender_keyboard)


@dp.message(AnalysisStates.waiting_for_gender, F.text.in_({"👨 Мужской", "👩 Женский"}))
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
            "Чтобы начать, выберите тариф в меню: <b>💎 Разбор лица</b> или <b>👑 Premium Plan</b>.",
            parse_mode="HTML", reply_markup=main_keyboard,
        )
        return

    set_selected_product(message.from_user.id, product)
    gender = "male" if "Мужской" in message.text else "female"
    await state.update_data(gender=gender, product=product, order_id=order_id)
    await state.set_state(AnalysisStates.waiting_for_photo)
    await message.answer(
        photo_prompt(product, gender),
        parse_mode="HTML", reply_markup=cancel_keyboard,
    )


@dp.message(AnalysisStates.waiting_for_gender)
async def wrong_gender(message: Message):
    await message.answer("Выберите пол кнопкой ниже.",
                         reply_markup=gender_keyboard)


@dp.message(StateFilter(None), F.text.in_({"👨 Мужской", "👩 Женский"}))
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
            "Чтобы начать, выберите тариф в меню: <b>💎 Разбор лица</b> или <b>👑 Premium Plan</b>.",
            parse_mode="HTML", reply_markup=main_keyboard,
        )
        return

    set_selected_product(message.from_user.id, product)
    gender = "male" if "Мужской" in message.text else "female"
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
            "Сначала выберите тариф в меню — после оплаты я приму фото и соберу PDF.",
            reply_markup=main_keyboard,
        )
        return

    await message.answer(
        f"<b>Анализирую лицо</b> · тариф «{product_title(product)}»\n\n"
        "Обычно занимает до 30 секунд.",
        parse_mode="HTML",
    )

    analysis, error = analyze_face(image_bytes, gender)
    if error:
        await message.answer(f"{error}\n\nПопробуйте другое фото.",
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
            "<b>Разбор готов.</b>\n\n"
            f"Оценка гармонии: <b>{analysis['score']:.2f} / 10</b>\n"
            f"Tier: <b>{tier['abbr']} · {tier['name']}</b>\n"
            f"Уровень: {analysis['level']}\n\n"
            f"Тариф: {product_title(product)}",
            parse_mode="HTML",
        )

        if product == PRODUCT_PREMIUM_PLAN:
            await message.answer_document(
                FSInputFile(face_pdf_path, filename=f"Разбор лица {BOT_NAME}.pdf"),
            )
            await message.answer_document(
                FSInputFile(looks_pdf_path, filename="Луксмаксинг-план Heim Face.pdf"),
            )
            if os.path.exists(DEPUFF_GUIDE_PATH):
                await message.answer_document(
                    FSInputFile(DEPUFF_GUIDE_PATH, filename="Бонус по отёчности Heim Face.pdf"),
                    reply_markup=main_keyboard,
                )
            else:
                logger.error("Premium depuff guide is missing: %s", DEPUFF_GUIDE_PATH)
                await message.answer(
                    "Основные PDF отправлены. Бонусный файл по отёчности не найден на сервере.",
                    reply_markup=main_keyboard,
                )
        else:
            await message.answer_document(
                FSInputFile(face_pdf_path, filename=f"Разбор лица {BOT_NAME}.pdf"),
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
            f"Ошибка при создании отчёта: {str(e)[:200]}\n\nПопробуйте ещё раз.",
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
        await message.answer("Нужно фото лица. Пришлите изображение.")
        return
    file = await bot.get_file(doc.file_id)
    buf  = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    await process_image(message, buf.getvalue(), state)


@dp.message(AnalysisStates.waiting_for_photo)
async def wrong_state_photo(message: Message):
    await message.answer("Жду фото лица для выбранного тарифа.", reply_markup=cancel_keyboard)


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
            "🛠 <b>TEST MODE (ADMIN)</b>\n\nСначала выберите тестируемый тариф: <b>💎 Разбор лица</b> или <b>👑 Premium Plan</b>.",
            parse_mode="HTML", reply_markup=main_keyboard,
        )
        return
    await message.answer(
        "Чтобы начать, выберите тариф в меню: <b>💎 Разбор лица</b> или <b>👑 Premium Plan</b>.",
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
            "🛠 <b>TEST MODE (ADMIN)</b>\n\nСначала выберите тестируемый тариф: <b>💎 Разбор лица</b> или <b>👑 Premium Plan</b>.",
            parse_mode="HTML", reply_markup=main_keyboard,
        )
        return
    await message.answer(
        "Чтобы начать, выберите тариф в меню: <b>💎 Разбор лица</b> или <b>👑 Premium Plan</b>.",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


@dp.message()
async def fallback(message: Message):
    await message.answer(
        "Выберите тариф в меню, чтобы начать: <b>💎 Разбор лица</b> или <b>👑 Premium Plan</b>.",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


# ================== ENTRY POINT ==================
async def main():
    logger.info(f"{BOT_NAME} starting...")
    threading.Thread(target=run_web, daemon=True).start()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

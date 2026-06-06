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
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import StateFilter, CommandStart, CommandObject
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
from share_card import create_share_card
from counters import (
    increment_counter, get_user_count, get_today_count, get_total_count,
    set_selected_product, get_selected_product, clear_selected_product,
    create_order, get_order, update_order_payment, mark_order_paid,
    mark_order_failed, get_active_paid_order, consume_paid_order,
    find_order_by_provider_payment_id,
    ensure_user, set_referrer, get_referrer, get_referral_stats,
    credit_referral, has_free_premium, consume_free_premium,
    mark_order_referral_credited,
    add_rating, get_ratings_summary,
    save_result_snapshot, get_last_snapshot, get_previous_snapshot,
)


# ================== LOGGING ==================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ================== BOT INIT ==================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN is not set. Создай .env с BOT_TOKEN=...")

# Увеличенный таймаут сессии: Premium PDF тяжёлый, дефолтных ~60с не хватает.
try:
    _bot_session = AiohttpSession(timeout=180)
    bot = Bot(token=BOT_TOKEN, session=_bot_session)
except TypeError:
    # На случай иной сигнатуры в версии aiogram — не валим запуск.
    logger.warning("AiohttpSession(timeout=...) не поддержан, использую дефолтную сессию")
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
                credit_referral_for_order(order)
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
        [KeyboardButton(text="👥 Бесплатно через друзей")],
        [KeyboardButton(text="👁 Пример разбора")],
        [KeyboardButton(text="❓ Что это?"), KeyboardButton(text="💬 Поддержка")],
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
DEMO_FACE_PATH = os.path.join(BASE_DIR, "assets", "Разбор лица Heim Face.pdf")
DEMO_PREMIUM_PATH = os.path.join(BASE_DIR, "assets", "Луксмаксинг-план Heim Face.pdf")
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
            f"<b>👑 Premium Plan</b>\n\n"
            f"Карта / СБП — {PREMIUM_PLAN_PRICE_RUB} ₽\n"
            f"Stars — {PREMIUM_PLAN_PRICE_STARS} ⭐\n\n"
            "Полный пакет для максимального раскрытия внешнего потенциала.\n\n"
            "<b>💎 Разбор лица</b>\n"
            "✅ Оценка гармонии\n"
            "✅ Tier\n"
            "✅ 20 метрик\n"
            "✅ Симметрия и пропорции\n\n"
            "<b>👑 Premium Plan</b>\n"
            "✅ Всё из разбора\n"
            "✅ Персональный план улучшения\n"
            "✅ Приоритеты по силе влияния\n"
            "✅ План на 7 дней\n"
            "✅ План на 30 дней\n"
            "✅ Рекомендации по причёске\n"
            "✅ Рекомендации по бровям\n"
            "✅ Рекомендации по коже\n"
            "✅ Рекомендации по нижней трети\n"
            "✅ Бонус по снижению отёчности\n\n"
            "<b>Доплата всего 500 ₽ позволяет не только узнать свой уровень, "
            "но и получить конкретный план улучшения.</b>\n\n"
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
    rows = [[InlineKeyboardButton(text="👁 Посмотреть пример отчёта", callback_data=f"demo:{product}")]]
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
        [InlineKeyboardButton(text="👁 Посмотреть пример отчёта", callback_data=f"demo:{product}")],
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


def credit_referral_for_order(order: dict):
    """Засчитывает реферала после оплаты заказа и уведомляет реферера.
    Идемпотентно (защита внутри credit_referral). Любые сбои не должны
    ломать основной поток выдачи разбора."""
    try:
        if not order or order.get("referral_credited"):
            return
        result = credit_referral(int(order["user_id"]))
        try:
            mark_order_referral_credited(order["order_id"])
        except Exception:
            pass
        if not result:
            return
        text = (
            "🎉 По вашей ссылке оплатили разбор.\n\n"
            f"Прогресс: <b>{result['paid_referrals_count']} / {result['goal']}</b>"
        )
        if result["granted_premium"]:
            text += "\n\n🎁 Вы получили бесплатный Premium Plan."
        send_bot_message_sync(result["referrer_id"], text)
    except Exception:
        logger.exception("Referral crediting failed for order %s", order.get("order_id"))


# ================== HANDLERS ==================
def _parse_ref_id(arg: str | None) -> int | None:
    """Извлекает user_id реферера из payload вида REF_123 или 123."""
    if not arg:
        return None
    token = arg.strip()
    if token.upper().startswith("REF_"):
        token = token[4:]
    if token.isdigit():
        return int(token)
    return None


@dp.message(CommandStart(deep_link=True))
@dp.message(CommandStart())
async def cmd_start_entry(message: Message, command: CommandObject, state: FSMContext):
    # Регистрируем пользователя и обрабатываем реферальную ссылку.
    ensure_user(message.from_user.id)
    ref_id = _parse_ref_id(command.args)
    if ref_id is not None:
        set_referrer(message.from_user.id, ref_id)  # не перезапишет и отсечёт самореферал
    await _show_start(message, state)


@dp.message(F.text.in_({"/help", "◀️ Назад в меню"}))
async def cmd_help_or_back(message: Message, state: FSMContext):
    await _show_start(message, state)


async def _show_start(message: Message, state: FSMContext):
    await state.clear()
    ensure_user(message.from_user.id)
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
    total = get_total_count()
    social = max(total, 300)  # социальное доказательство, не ниже базовой планки
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
        f"📊 Уже выполнено более {social} разборов.\n\n"
        "Выберите тариф ниже.",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


@dp.message(F.text == "❓ Что это?")
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


@dp.message(F.text == "💬 Поддержка")
async def support(message: Message):
    await message.answer(
        "<b>Поддержка</b>\n\n"
        "Вопросы по оплате, доступу или отчёту — пишите напрямую: @aeonin\n\n"
        "Отвечаем лично и решаем быстро.",
        parse_mode="HTML",
    )


@dp.message(F.text == "👥 Бесплатно через друзей")
async def referral_menu(message: Message):
    ensure_user(message.from_user.id)
    stats = get_referral_stats(message.from_user.id)
    username = BOT_USERNAME.lstrip("@")
    link = f"https://t.me/{username}?start=REF_{message.from_user.id}"
    paid = stats["paid_referrals_count"]
    goal = stats["goal"]
    free = stats["free_premium_count"]

    text = (
        "<b>👥 Получите Premium Plan бесплатно</b>\n\n"
        f"Пригласите {goal} друзей, которые оплатят любой тариф, "
        "и получите полный Premium Plan без оплаты.\n\n"
        "Ваш прогресс:\n"
        f"<b>{paid} / {goal}</b> оплаченных приглашений\n\n"
        "Ваша ссылка:\n"
        f"{link}"
    )
    if free > 0:
        text += f"\n\n🎁 У вас есть бесплатный Premium Plan: <b>{free}</b>. Нажмите «👑 Premium Plan», чтобы использовать."

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="👁 Посмотреть пример Premium", callback_data=f"demo:{PRODUCT_PREMIUM_PLAN}")]
    ])
    await message.answer(text, parse_mode="HTML", disable_web_page_preview=True, reply_markup=keyboard)


async def start_product_flow(message: Message, state: FSMContext, product: str, user_id: int | None = None):
    if user_id is None:
        user_id = message.from_user.id
    await state.clear()
    set_selected_product(user_id, product)
    await state.update_data(product=product)

    # Бесплатный Premium Plan, начисленный за рефералов — без оплаты.
    if product == PRODUCT_PREMIUM_PLAN and has_free_premium(user_id):
        if consume_free_premium(user_id):
            await state.update_data(free_premium=True)
            await state.set_state(AnalysisStates.waiting_for_gender)
            await message.answer(
                "🎁 <b>Бесплатный Premium Plan активирован.</b>\n\n"
                "Оплата не требуется — он начислен за приглашённых друзей.\n\n"
                "Выберите пол — нормы анализа различаются для мужчин и женщин.",
                parse_mode="HTML", reply_markup=gender_keyboard,
            )
            return

    if is_admin(user_id):
        log_admin_bypass(user_id)
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


def _demo_files_for(product: str):
    """Список (path, filename) демо-файлов для тарифа."""
    if product == PRODUCT_PREMIUM_PLAN:
        return [
            (DEMO_FACE_PATH, "Пример · Разбор лица.pdf"),
            (DEMO_PREMIUM_PATH, "Пример · Луксмаксинг-план.pdf"),
        ]
    return [(DEMO_FACE_PATH, "Пример · Разбор лица.pdf")]


async def _send_demo(message: Message, product: str, closing: str | None = None,
                     closing_markup=None) -> bool:
    """Отправляет демо-PDF тарифа. Возвращает False, если файлов нет.
    Не создаёт заказ и не меняет FSM-состояние."""
    logger.info("DEMO CALLBACK START | product=%s", product)
    sent_any = False
    for path, name in _demo_files_for(product):
        exists = os.path.exists(path)
        logger.info("FILE EXISTS: %s -> %s", path, exists)
        if not exists:
            continue
        try:
            logger.info("SENDING FILE: %s", path)
            await message.answer_document(FSInputFile(path, filename=name))
            sent_any = True
        except Exception:
            logger.exception("Failed to send demo file: %s", path)
            await message.answer("Пример отчёта временно недоступен. Попробуйте позже.")
            return True
    if not sent_any:
        logger.warning("DEMO: no files found for product=%s (checked assets/)", product)
        return False
    if closing:
        await message.answer(closing, parse_mode="HTML", reply_markup=closing_markup)
    return True


# --- Подменю «Пример разбора» из главного меню ---
def demo_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📄 Пример обычного разбора", callback_data=f"demo:{PRODUCT_FACE_REPORT}")],
        [InlineKeyboardButton(text="👑 Пример Premium Plan", callback_data=f"demo:{PRODUCT_PREMIUM_PLAN}")],
    ])


@dp.message(F.text == "👁 Пример разбора")
async def example_menu(message: Message):
    await message.answer(
        "<b>Примеры отчётов</b>\n\n"
        "Выберите, что показать — это просто пример формата, оплата не требуется.",
        parse_mode="HTML",
        reply_markup=demo_menu_keyboard(),
    )


FEEDBACK_DELAY_SECONDS = 5 * 60


def _build_dynamics_message(prev: dict, new_score, new_tier) -> str | None:
    """Строит сообщение динамики «было → стало». Возвращает None, если
    данных недостаточно для корректного сравнения (тогда ничего не показываем)."""
    try:
        if not isinstance(prev, dict):
            return None
        old_score = prev.get("score")
        old_tier = prev.get("tier")
        if old_score is None or new_score is None or old_tier is None or new_tier is None:
            return None
        old_s = float(old_score)
        new_s = float(new_score)
    except (TypeError, ValueError):
        return None

    delta = new_s - old_s
    if delta > 0.15:
        sign = f"+{delta:.2f}"
        verdict = "Рост есть. Продолжайте отслеживать прогресс раз в 30 дней."
    elif delta < -0.15:
        sign = f"{delta:.2f}"
        verdict = ("Разница может быть связана со светом, ракурсом, отёчностью "
                   "или качеством фото.")
    else:
        sign = f"{'+' if delta >= 0 else ''}{delta:.2f}"
        verdict = ("Результат стабилен. Для заметного изменения обычно нужен "
                   "цикл 30 дней.")

    return (
        "📊 <b>Динамика результата</b>\n\n"
        "Прошлый разбор:\n"
        f"{old_s:.2f} / 10 · {old_tier}\n\n"
        "Новый разбор:\n"
        f"{new_s:.2f} / 10 · {new_tier}\n\n"
        f"Изменение: <b>{sign} балла</b>\n\n"
        f"{verdict}"
    )


async def _schedule_feedback_request(chat_id: int):
    """Через 5 минут после разбора просит оценить качество. Без БД/планировщиков."""
    try:
        await asyncio.sleep(FEEDBACK_DELAY_SECONDS)
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⭐", callback_data="rate:1")],
            [InlineKeyboardButton(text="⭐⭐", callback_data="rate:2")],
            [InlineKeyboardButton(text="⭐⭐⭐", callback_data="rate:3")],
            [InlineKeyboardButton(text="⭐⭐⭐⭐", callback_data="rate:4")],
            [InlineKeyboardButton(text="⭐⭐⭐⭐⭐", callback_data="rate:5")],
        ])
        await bot.send_message(
            chat_id,
            "⭐ <b>Как вам качество разбора?</b>\n\nВыберите оценку:",
            parse_mode="HTML", reply_markup=kb,
        )
    except Exception:
        logger.exception("Failed to send feedback request")


@dp.callback_query(F.data.startswith("rate:"))
async def rating_callback(callback: CallbackQuery):
    try:
        rating = int(callback.data.split(":", 1)[1])
    except (ValueError, IndexError):
        await callback.answer()
        return
    add_rating(callback.from_user.id, rating)
    await callback.answer("Спасибо!")
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass
    if rating >= 5:
        await callback.message.answer(
            "Спасибо за высокую оценку ❤️\n\n"
            "Если разбор оказался полезным, буду благодарен за отзыв "
            "или рекомендацию друзьям.",
        )
    else:
        await callback.message.answer(
            "Спасибо за обратную связь.\nМы будем улучшать качество разбора.",
        )


@dp.callback_query(F.data == "upsell_premium")
async def upsell_premium_callback(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await start_product_flow(callback.message, state, PRODUCT_PREMIUM_PLAN, user_id=callback.from_user.id)


@dp.callback_query(F.data.startswith("demo:"))
async def demo_callback(callback: CallbackQuery):
    """Отправляет пример(ы) отчёта. Не создаёт заказ и не меняет состояние."""
    product = callback.data.split(":", 1)[1]
    await callback.answer()
    buy_kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💎 Заказать разбор", callback_data="buy:face_report")],
        [InlineKeyboardButton(text="👑 Premium Plan", callback_data="buy:premium_plan")],
    ])
    ok = await _send_demo(
        callback.message, product,
        closing="Так выглядит формат отчёта. Хотите узнать свои реальные показатели?",
        closing_markup=buy_kb,
    )
    if not ok:
        # callback.answer уже вызван выше — fallback шлём сообщением, иначе его не видно.
        await callback.message.answer("Пример отчёта временно недоступен. Попробуйте позже.")


@dp.callback_query(F.data.startswith("buy:"))
async def buy_callback(callback: CallbackQuery, state: FSMContext):
    """Открывает существующий экран оплаты выбранного тарифа."""
    product = callback.data.split(":", 1)[1]
    if product not in {PRODUCT_FACE_REPORT, PRODUCT_PREMIUM_PLAN}:
        await callback.answer("Неизвестный тариф.", show_alert=True)
        return
    await callback.answer()
    await start_product_flow(callback.message, state, product, user_id=callback.from_user.id)


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
        credit_referral_for_order(order)
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

    credit_referral_for_order(order)
    set_selected_product(message.from_user.id, order["product"])
    await state.set_state(AnalysisStates.waiting_for_gender)
    await message.answer(paid_prompt(order["product"]), parse_mode="HTML", reply_markup=gender_keyboard)


@dp.message(AnalysisStates.waiting_for_gender, F.text.in_({"👨 Мужской", "👩 Женский"}))
async def choose_gender(message: Message, state: FSMContext):
    data = await state.get_data()
    active_order = get_active_paid_order(message.from_user.id)
    product = None
    order_id = None
    if active_order:
        product = active_order["product"]
        order_id = active_order["order_id"]
    elif data.get("free_premium"):
        # Бесплатный Premium за рефералов — без заказа.
        product = PRODUCT_PREMIUM_PLAN
        order_id = None
    elif is_admin(message.from_user.id):
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
    await state.update_data(gender=gender, product=product, order_id=order_id,
                            free_premium=data.get("free_premium", False))
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


TELEGRAM_TIMEOUT_MSG = (
    "Файл готов, но Telegram не успел его отправить. "
    "Попробуйте ещё раз или напишите в поддержку."
)


async def _send_pdf_logged(message: Message, path: str, filename: str, reply_markup=None) -> bool:
    """Отправляет один документ с логированием и обработкой таймаута.
    Возвращает True при успехе, False при ошибке (пользователю показано сообщение)."""
    try:
        size = os.path.getsize(path)
        logger.info("PDF создан: %s | размер %.1f KB", filename, size / 1024)
        logger.info("Начинаю отправку: %s", filename)
        await message.answer_document(
            FSInputFile(path, filename=filename), reply_markup=reply_markup,
        )
        logger.info("Файл отправлен: %s", filename)
        return True
    except asyncio.TimeoutError:
        logger.error("Таймаут отправки файла: %s", filename)
        await message.answer(TELEGRAM_TIMEOUT_MSG, reply_markup=main_keyboard)
        return False
    except Exception:
        logger.exception("Ошибка отправки файла: %s", filename)
        await message.answer(TELEGRAM_TIMEOUT_MSG, reply_markup=main_keyboard)
        return False


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
        "🔍 <b>Анализирую геометрию лица</b>\n\n"
        "• рассчитываю пропорции\n"
        "• определяю tier\n"
        "• ищу сильные стороны\n"
        "• формирую персональный отчёт\n\n"
        "Обычно занимает 10–30 секунд.",
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
            await _send_pdf_logged(message, face_pdf_path, f"Разбор лица {BOT_NAME}.pdf")
            await asyncio.sleep(1.5)
            await _send_pdf_logged(message, looks_pdf_path, "Луксмаксинг-план Heim Face.pdf")
            await asyncio.sleep(1.5)
            if os.path.exists(DEPUFF_GUIDE_PATH):
                await _send_pdf_logged(
                    message, DEPUFF_GUIDE_PATH, "Бонус по отёчности Heim Face.pdf",
                    reply_markup=main_keyboard,
                )
            else:
                logger.error("Premium depuff guide is missing: %s", DEPUFF_GUIDE_PATH)
                await message.answer(
                    "Основные PDF отправлены. Бонусный файл по отёчности не найден на сервере.",
                    reply_markup=main_keyboard,
                )
        else:
            await _send_pdf_logged(
                message, face_pdf_path, f"Разбор лица {BOT_NAME}.pdf",
                reply_markup=main_keyboard,
            )

        increment_counter(message.from_user.id, product)
        if order_id:
            consume_paid_order(order_id)
        clear_selected_product(message.from_user.id)

        # Снимок результата + динамика при повторном разборе (не влияет на выдачу).
        try:
            _prev = get_last_snapshot(message.from_user.id)  # прошлый результат (до сохранения нового)
            _metrics = analysis.get("metrics") or []
            _weakest = min(_metrics, key=lambda m: m["score"])["name"] if _metrics else None
            _new_tier = (analysis.get("tier") or {}).get("abbr")
            _new_score = analysis.get("score")
            _saved = save_result_snapshot(
                message.from_user.id,
                score=_new_score,
                tier=_new_tier,
                top_percent=analysis.get("top_percent"),
                weakest_metric=_weakest,
                product=product,
            )
            if _saved and _prev:
                _msg = _build_dynamics_message(_prev, _new_score, _new_tier)
                if _msg:
                    await message.answer(_msg, parse_mode="HTML")
        except Exception:
            logger.exception("result snapshot/dynamics failed (non-blocking)")

        # Карточка результата для шаринга (PNG)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                card_path = tmp.name
            temp_paths.append(card_path)
            create_share_card(analysis, card_path)
            await message.answer_photo(
                FSInputFile(card_path, filename="Heim Face.png"),
                caption="📤 Сохраните карточку или отправьте другу.\n\nСравните результаты и узнайте, кто находится выше по tier.",
            )
        except Exception:
            logger.exception("Share card generation failed")

        # Блок потенциала — только для Premium (после всех файлов и карточки)
        if product == PRODUCT_PREMIUM_PLAN:
            tier = analysis["tier"]
            await message.answer(
                "📈 <b>Потенциал после внедрения плана</b>\n\n"
                f"Текущий уровень:\n<b>{tier['abbr']} · {tier['name']}</b>\n\n"
                "При соблюдении рекомендаций ваш визуальный потенциал может "
                "заметно вырасти.\n\n"
                "<i>Результат зависит от дисциплины, качества внедрения и исходных данных.</i>",
                parse_mode="HTML",
            )

        # Апселл Premium после обычного разбора
        if product == PRODUCT_FACE_REPORT:
            # Самая слабая метрика пользователя — главный лимитер.
            weakest = None
            try:
                metrics = analysis.get("metrics") or []
                if metrics:
                    weakest = min(metrics, key=lambda m: m["score"])["name"]
            except Exception:
                weakest = None

            if weakest:
                upsell_text = (
                    f"🔥 <b>Ваш главный лимитер сейчас: {weakest}</b>\n\n"
                    "Premium Plan покажет:\n"
                    "• что даёт наибольший визуальный эффект;\n"
                    "• что улучшать первым;\n"
                    "• какие изменения дадут максимальный прирост восприятия."
                )
            else:
                upsell_text = (
                    "🔥 <b>Хотите получить персональный план улучшения?</b>\n\n"
                    "Мы уже рассчитали ваши сильные зоны и зоны роста. "
                    "Premium Plan покажет, что именно улучшать первым: брови, кожу, "
                    "причёску, нижнюю треть и фото-подачу.\n\n"
                    "Внутри:\n"
                    "✅ персональный луксмаксинг-план\n"
                    "✅ приоритеты улучшений\n"
                    "✅ план на 7 дней\n"
                    "✅ план на 30 дней\n"
                    "✅ бонус по снижению отёчности"
                )
            await message.answer(
                upsell_text,
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="👑 Получить Premium Plan", callback_data="upsell_premium")]
                ]),
            )

        # Запрос оценки качества через 5 минут (без БД/планировщиков)
        asyncio.create_task(_schedule_feedback_request(message.chat.id))

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

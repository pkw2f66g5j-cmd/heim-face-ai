import os
import io
import asyncio
import logging
import tempfile
import threading

from dotenv import load_dotenv
load_dotenv()

from flask import Flask

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile, KeyboardButton, ReplyKeyboardMarkup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

from config import (
    BOT_NAME, BOT_USERNAME,
    FACE_REPORT_PRICE_RUB, PREMIUM_PLAN_PRICE_RUB,
    PRODUCT_FACE_REPORT, PRODUCT_PREMIUM_PLAN,
    YOOKASSA_ENABLED,
)
from analysis import analyze_face
from pdf_builder import create_pdf_report, create_looksmaxxing_pdf
from counters import (
    increment_counter, get_user_count, get_today_count,
    set_selected_product, get_selected_product, clear_selected_product,
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


def payment_mode_text() -> str:
    if YOOKASSA_ENABLED:
        return "Оплата: ЮKassa подготовлена. После включения платежного обработчика здесь будет платёжная ссылка."
    return (
        "Оплата: тестовый режим. Ключи ЮKassa не заданы, поэтому бот не списывает деньги "
        "и сразу открывает получение отчёта."
    )


def product_title(product: str) -> str:
    if product == PRODUCT_PREMIUM_PLAN:
        return "Premium Plan"
    return "Разбор лица"


def product_price(product: str) -> int:
    if product == PRODUCT_PREMIUM_PLAN:
        return PREMIUM_PLAN_PRICE_RUB
    return FACE_REPORT_PRICE_RUB


def product_description(product: str) -> str:
    if product == PRODUCT_PREMIUM_PLAN:
        return (
            f"<b>👑 Premium Plan</b>\n"
            f"Цена: <b>{PREMIUM_PLAN_PRICE_RUB} ₽</b>\n\n"
            "Премиальный пакет для тех, кто хочет не только цифры, но и понятный план улучшения образа.\n\n"
            "<b>Внутри:</b>\n"
            "• 23-страничный PDF-разбор лица\n"
            "• 20 метрик, симметрия, пропорции, точки и линии на лице\n"
            "• итоговая оценка гармонии и tier\n"
            "• отдельный PDF «Луксмаксинг-план»\n"
            "• рекомендации по причёске, коже, бровям и нижней трети\n"
            "• план по снижению отёчности\n"
            "• советы по фото и позированию\n"
            "• план на 7 дней и план на 30 дней\n"
            "• бонусный PDF по отёчности\n\n"
            f"{payment_mode_text()}\n\n"
            "<b>Выберите пол</b> — нормы анализа отличаются для мужчин и женщин."
        )

    return (
        f"<b>💎 Разбор лица</b>\n"
        f"Цена: <b>{FACE_REPORT_PRICE_RUB} ₽</b>\n\n"
        "Точный математический разбор геометрии лица в стиле Heim Face: чёрный premium, золото, графит и чистые выводы без шума.\n\n"
        "<b>Внутри:</b>\n"
        "• 23-страничный PDF-разбор лица\n"
        "• 20 ключевых метрик\n"
        "• симметрия и пропорции\n"
        "• точки и линии прямо на фото\n"
        "• итоговая оценка гармонии и tier\n\n"
        f"{payment_mode_text()}\n\n"
        "<b>Выберите пол</b> — нормы анализа отличаются для мужчин и женщин."
    )


# ================== HANDLERS ==================
@dp.message(F.text.in_({"/start", "/help", "◀️ Назад в меню"}))
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    clear_selected_product(message.from_user.id)
    uc = get_user_count(message.from_user.id)
    tc = get_today_count()
    await message.answer(
        "<b>Heim Face</b>\n"
        "Математический разбор лица в премиальном формате: геометрия, симметрия, пропорции и понятный итоговый tier.\n\n"
        f"Ваши готовые разборы: <b>{uc}</b>\n"
        f"Сегодня проведено разборов: <b>{tc}</b>\n\n"
        "Выберите тариф:",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


@dp.message(F.text == "Что это?")
async def what_is_this(message: Message):
    await message.answer(
        "<b>Что такое Heim Face</b>\n\n"
        "Это сервис математической оценки гармонии лица. Алгоритм находит ключевые точки, "
        "считает 20 антропометрических метрик и сравнивает их с нормами отдельно для мужчин и женщин.\n\n"
        "<b>Тарифы:</b>\n"
        f"💎 Разбор лица — <b>{FACE_REPORT_PRICE_RUB} ₽</b>: основной PDF на 23 страницы.\n"
        f"👑 Premium Plan — <b>{PREMIUM_PLAN_PRICE_RUB} ₽</b>: основной PDF, луксмаксинг-план и бонус по отёчности.\n\n"
        "Формат спокойный и премиальный: чёрный, золото, графит, без агрессивных обещаний.",
        parse_mode="HTML",
    )


@dp.message(F.text == "Техподдержка")
async def support(message: Message):
    await message.answer(
        "<b>Техподдержка</b>\n\n"
        f"По вопросам оплаты, отчётов и доступа напишите администратору проекта или проверьте бота {BOT_USERNAME}.",
        parse_mode="HTML",
    )


async def start_product_flow(message: Message, state: FSMContext, product: str):
    set_selected_product(message.from_user.id, product)
    await state.set_state(AnalysisStates.waiting_for_gender)
    await state.update_data(product=product)
    await message.answer(product_description(product), parse_mode="HTML", reply_markup=gender_keyboard)


@dp.message(F.text == "💎 Разбор лица")
async def get_face_report(message: Message, state: FSMContext):
    await start_product_flow(message, state, PRODUCT_FACE_REPORT)


@dp.message(F.text == "👑 Premium Plan")
async def get_premium_plan(message: Message, state: FSMContext):
    await start_product_flow(message, state, PRODUCT_PREMIUM_PLAN)


@dp.message(F.text == "💠 Хочу получить свой разбор")
async def get_legacy_report(message: Message, state: FSMContext):
    await start_product_flow(message, state, PRODUCT_FACE_REPORT)


@dp.message(AnalysisStates.waiting_for_gender, F.text.in_({"👨 Мужской", "👩 Женский"}))
async def choose_gender(message: Message, state: FSMContext):
    product = get_selected_product(message.from_user.id)
    if not product:
        await state.clear()
        await message.answer(
            "Сначала выберите тариф в главном меню: <b>💎 Разбор лица</b> или <b>👑 Premium Plan</b>.",
            parse_mode="HTML", reply_markup=main_keyboard,
        )
        return

    gender = "male" if "Мужской" in message.text else "female"
    await state.update_data(gender=gender, product=product)
    await state.set_state(AnalysisStates.waiting_for_photo)
    word = "мужской" if gender == "male" else "женский"
    await message.answer(
        f"Пол выбран: <b>{word}</b>\n"
        f"Тариф: <b>{product_title(product)}</b> · {product_price(product)} ₽\n\n"
        "<b>Отправьте фото лица.</b>\n\n"
        "Требования:\n"
        "• Строго анфас (прямо в камеру)\n"
        "• Нейтральное выражение, рот закрыт\n"
        "• Хорошее равномерное освещение\n"
        "• Без очков, маски, головного убора\n"
        "• Волосы не закрывают лоб и брови",
        parse_mode="HTML", reply_markup=cancel_keyboard,
    )


@dp.message(AnalysisStates.waiting_for_gender)
async def wrong_gender(message: Message):
    await message.answer("Пожалуйста, выберите пол кнопкой ниже.",
                         reply_markup=gender_keyboard)


async def process_image(message: Message, image_bytes: bytes, state: FSMContext):
    data   = await state.get_data()
    gender = data.get("gender", "male")
    product = get_selected_product(message.from_user.id)

    if not product:
        await state.clear()
        await message.answer(
            "Для начала выберите тариф в главном меню. После этого я приму фото и соберу PDF.",
            reply_markup=main_keyboard,
        )
        return

    await message.answer(
        f"<b>Анализирую лицо для тарифа «{product_title(product)}»...</b>\n\n"
        "Обычно это занимает до 30 секунд.",
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

        increment_counter(message.from_user.id, product)

        tier = analysis["tier"]
        await message.answer(
            "<b>Разбор готов.</b>\n\n"
            f"Итоговая оценка: <b>{analysis['score']:.2f} / 10</b>\n"
            f"Уровень: <b>{analysis['level']}</b>\n"
            f"Tier: <b>{tier['abbr']} · {tier['name']}</b>\n\n"
            f"Тариф: <b>{product_title(product)}</b>",
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
                await message.answer(
                    "Основные PDF отправлены. Бонусный файл по отёчности не найден на сервере.",
                    reply_markup=main_keyboard,
                )
        else:
            await message.answer_document(
                FSInputFile(face_pdf_path, filename=f"Разбор лица {BOT_NAME}.pdf"),
                reply_markup=main_keyboard,
            )

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
        await message.answer("Это не изображение. Отправьте фото лица.")
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
    await message.answer(
        "Сначала выберите тариф в главном меню: <b>💎 Разбор лица</b> или <b>👑 Premium Plan</b>.",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


@dp.message(F.document)
async def doc_no_state(message: Message):
    await message.answer(
        "Сначала выберите тариф в главном меню: <b>💎 Разбор лица</b> или <b>👑 Premium Plan</b>.",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


@dp.message()
async def fallback(message: Message):
    await message.answer(
        "Чтобы начать, выберите тариф: <b>💎 Разбор лица</b> или <b>👑 Premium Plan</b>.",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


# ================== ENTRY POINT ==================
async def main():
    logger.info(f"{BOT_NAME} starting...")
    threading.Thread(target=run_web, daemon=True).start()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

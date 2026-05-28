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

from config import BOT_NAME, BOT_USERNAME
from analysis import analyze_face
from pdf_builder import create_pdf_report
from counters import increment_counter, get_user_count, get_today_count


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
        [KeyboardButton(text="💠 Хочу получить свой разбор")],
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


# ================== HANDLERS ==================
@dp.message(F.text.in_({"/start", "/help", "◀️ Назад в меню"}))
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    uc = get_user_count(message.from_user.id)
    tc = get_today_count()
    await message.answer(
        "📍 <b>Главное меню</b>\n\n"
        f"<b>{BOT_NAME}</b> математически измеряет, насколько гармонично "
        "черты твоего лица сочетаются друг с другом.\n\n"
        f"Твой баланс: <b>{uc} разборов</b>\n"
        f"Сегодня пользователи провели: <b>{tc} разборов</b>",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


@dp.message(F.text == "Что это?")
async def what_is_this(message: Message):
    await message.answer(
        "📍 <b>Главное меню › Что это</b>\n\n"
        f"<b>{BOT_NAME}</b> — сервис математической оценки гармонии пропорций лица.\n\n"
        "🔬 Алгоритм определяет ключевые точки лица, рассчитывает 20 "
        "антропометрических метрик и сравнивает их с нормами отдельно для мужчин и женщин.\n\n"
        "📄 В PDF-отчёте на 23 страницы:\n"
        "• Итоговая оценка, уровень и tier\n"
        "• Профиль метрик с радар-чартом\n"
        "• 20 страниц с разбором каждой метрики\n"
        "• Визуализация измерений прямо на фото\n"
        "• Сильные стороны и зоны потенциала\n"
        "• Персональные рекомендации",
        parse_mode="HTML",
    )


@dp.message(F.text == "Техподдержка")
async def support(message: Message):
    await message.answer(
        "📍 <b>Главное меню › Техподдержка</b>\n\n"
        "По всем вопросам напишите администратору проекта.",
        parse_mode="HTML",
    )


@dp.message(F.text == "💠 Хочу получить свой разбор")
async def get_report(message: Message, state: FSMContext):
    await state.set_state(AnalysisStates.waiting_for_gender)
    await message.answer(
        "📍 <b>Главное меню › Разбор</b>\n\n"
        "⚜️ <b>1 полный разбор лица</b>\n\n"
        "📚 <b>Что входит:</b>\n"
        "• Персональный PDF-отчёт на 23 страницы\n"
        "• Tier-уровень внешности\n"
        "• Разбор 20 ключевых метрик с нормами для твоего пола\n"
        "• Визуализация измерений на фото\n"
        "• Подробное объяснение каждой метрики\n"
        "• Анализ сильных сторон и зон потенциала\n"
        "• Персональные рекомендации\n\n"
        "👤 <b>Выберите ваш пол</b> — алгоритм использует разные нормы:",
        parse_mode="HTML", reply_markup=gender_keyboard,
    )


@dp.message(AnalysisStates.waiting_for_gender, F.text.in_({"👨 Мужской", "👩 Женский"}))
async def choose_gender(message: Message, state: FSMContext):
    gender = "male" if "Мужской" in message.text else "female"
    await state.update_data(gender=gender)
    await state.set_state(AnalysisStates.waiting_for_photo)
    word = "мужской" if gender == "male" else "женский"
    await message.answer(
        f"✅ Пол выбран: <b>{word}</b>\n\n"
        "📸 <b>Отправьте фото лица.</b>\n\n"
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
    await message.answer("Пожалуйста, выберите пол кнопкой ниже 👇",
                         reply_markup=gender_keyboard)


async def process_image(message: Message, image_bytes: bytes, state: FSMContext):
    data   = await state.get_data()
    gender = data.get("gender", "male")

    await message.answer("⏳ <b>Анализирую лицо...</b>\n\nЭто займёт до 30 секунд.",
                         parse_mode="HTML")

    analysis, error = analyze_face(image_bytes, gender)
    if error:
        await message.answer(f"❌ {error}\n\nПопробуйте другое фото.",
                             reply_markup=cancel_keyboard)
        return

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_path = tmp.name
        create_pdf_report(image_bytes, analysis, gender, pdf_path)
        increment_counter(message.from_user.id)

        tier = analysis["tier"]
        await message.answer(
            "✅ <b>Разбор завершён!</b>\n\n"
            f"Итоговая оценка: <b>{analysis['score']:.2f} / 10</b>\n"
            f"Уровень: <b>{analysis['level']}</b>\n"
            f"Tier: <b>{tier['abbr']} · {tier['name']}</b>\n\n"
            "Полный отчёт ниже ↓",
            parse_mode="HTML",
        )
        await message.answer_document(
            FSInputFile(pdf_path, filename=f"Отчёт {BOT_NAME}.pdf"),
            reply_markup=main_keyboard,
        )
        await state.clear()

        try:
            os.remove(pdf_path)
        except OSError:
            pass

    except Exception as e:
        logger.exception("PDF generation failed")
        await message.answer(
            f"❌ Ошибка при создании отчёта: {str(e)[:200]}\n\nПопробуйте ещё раз.",
            reply_markup=cancel_keyboard,
        )


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
        await message.answer("📎 Это не изображение. Отправьте фото.")
        return
    file = await bot.get_file(doc.file_id)
    buf  = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    await process_image(message, buf.getvalue(), state)


@dp.message(AnalysisStates.waiting_for_photo)
async def wrong_state_photo(message: Message):
    await message.answer("📸 Жду фото лица.", reply_markup=cancel_keyboard)


@dp.message(F.photo)
async def photo_no_state(message: Message):
    await message.answer(
        "👋 Чтобы получить разбор, нажмите <b>«💠 Хочу получить свой разбор»</b>.",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


@dp.message(F.document)
async def doc_no_state(message: Message):
    await message.answer(
        "👋 Чтобы получить разбор, нажмите <b>«💠 Хочу получить свой разбор»</b>.",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


@dp.message()
async def fallback(message: Message):
    await message.answer(
        "📸 Чтобы начать — нажмите <b>«💠 Хочу получить свой разбор»</b>.",
        parse_mode="HTML", reply_markup=main_keyboard,
    )


# ================== ENTRY POINT ==================
async def main():
    logger.info(f"{BOT_NAME} starting...")
    threading.Thread(target=run_web, daemon=True).start()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

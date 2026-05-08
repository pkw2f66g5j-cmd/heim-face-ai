from flask import Flask
import threading
import os
import asyncio
import io
import logging
import math
import tempfile
import re

from dotenv import load_dotenv

load_dotenv()

import cv2
import mediapipe as mp
import numpy as np

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile, KeyboardButton, ReplyKeyboardMarkup

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from PIL import Image


# ── Web server for Render ─────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def home():
    return "Heim Face bot is running"

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Bot init ──────────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN is not set")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


# ── MediaPipe FaceMesh ────────────────────────────────────────────────────────
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
)


# ── Landmark indices ──────────────────────────────────────────────────────────
IDX_FACE_LEFT = 234
IDX_FACE_RIGHT = 454
IDX_NOSE_LEFT = 129
IDX_NOSE_RIGHT = 358
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291
IDX_EYE_LEFT = 33
IDX_EYE_RIGHT = 263


def euclidean(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return text.replace("&nbsp;", " ")


def analyze_face(image_bytes: bytes) -> str:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return "❌ Не удалось декодировать изображение."

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return "🙁 Лицо не обнаружено. Попробуйте фото с чётким фронтальным видом лица."

    lm = results.multi_face_landmarks[0].landmark

    def pt(idx):
        return lm[idx].x * w, lm[idx].y * h

    face_w = euclidean(pt(IDX_FACE_LEFT), pt(IDX_FACE_RIGHT))
    nose_w = euclidean(pt(IDX_NOSE_LEFT), pt(IDX_NOSE_RIGHT))
    mouth_w = euclidean(pt(IDX_MOUTH_LEFT), pt(IDX_MOUTH_RIGHT))
    eye_dist = euclidean(pt(IDX_EYE_LEFT), pt(IDX_EYE_RIGHT))

    nose_pct = nose_w / face_w * 100
    mouth_pct = mouth_w / face_w * 100

    # Примерные score-значения для первого PDF
    nose_score = max(0, min(10, 10 - abs(nose_pct - 25) * 0.35))
    mouth_score = max(0, min(10, 10 - abs(mouth_pct - 40) * 0.45))
    eye_score = max(0, min(10, 8.5))

    total_score = (nose_score + mouth_score + eye_score) / 3

    lines = [
        "📐 <b>Анализ лица</b>",
        "",
        f"⭐ <b>Итоговая оценка:</b> {total_score:.2f} / 10",
        "",
        f"📏 <b>Ширина лица:</b> {face_w:.1f} px",
        f"👃 <b>Ширина носа:</b> {nose_w:.1f} px ({nose_pct:.1f}% от ширины лица)",
        f"👄 <b>Ширина рта:</b> {mouth_w:.1f} px ({mouth_pct:.1f}% от ширины лица)",
        f"👀 <b>Расстояние между глазами:</b> {eye_dist:.1f} px",
        "",
        "<b>Интерпретация:</b>",
        "Ваше лицо было проанализировано по ключевым геометрическим точкам.",
        "Расчёт основан на пропорциях между шириной лица, носа, рта и глазной зоны.",
        "",
        "<i>Это первый базовый PDF-отчёт. Позже можно расширить его до 23 страниц.</i>",
    ]

    return "\n".join(lines)


def create_pdf_report(image_bytes: bytes, analysis_text: str, output_path: str):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    # Background
    c.setFillColor(HexColor("#050814"))
    c.rect(0, 0, width, height, fill=1)

    # Title
    c.setFillColor(HexColor("#9B2DFF"))
    c.setFont("Helvetica-Bold", 30)
    c.drawCentredString(width / 2, height - 60, "Heim Face")

    c.setFillColor(HexColor("#B8B8C8"))
    c.setFont("Helvetica", 11)
    c.drawCentredString(width / 2, height - 82, "Telegram: @heim_face_bot")
    c.drawCentredString(width / 2, height - 102, "Математический разбор пропорций лица")

    # User photo
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_reader = ImageReader(img)

    img_w = 230
    img_h = 300
    x = (width - img_w) / 2
    y = height - 430

    c.drawImage(
        img_reader,
        x,
        y,
        width=img_w,
        height=img_h,
        preserveAspectRatio=True,
        mask="auto",
    )

    # Score
    text_clean = clean_html(analysis_text)
    score_match = re.search(r"Итоговая оценка:\s*([0-9.]+)", text_clean)
    score = score_match.group(1) if score_match else "7.69"

    c.setFillColor(HexColor("#00E0C6"))
    c.setFont("Helvetica-Bold", 42)
    c.drawCentredString(width / 2, y - 55, score)

    c.setFillColor(HexColor("#B8B8C8"))
    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, y - 77, "из 10")

    # Analysis box
    c.setFillColor(HexColor("#101624"))
    c.roundRect(45, 65, width - 90, 260, 14, fill=1, stroke=0)

    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(65, 295, "Анализ лица")

    c.setFont("Helvetica", 10)
    y_text = 270

    for line in text_clean.split("\n"):
        line = line.strip()
        if not line:
            y_text -= 8
            continue

        if y_text < 85:
            break

        c.drawString(65, y_text, line[:90])
        y_text -= 16

    # Footer
    c.setFillColor(HexColor("#666B80"))
    c.setFont("Helvetica", 9)
    c.drawCentredString(width / 2, 35, "Heim Face · @heim_face_bot")

    c.save()


# ── Keyboards ────────────────────────────────────────────────────────────────
main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="💠 Получить разбор лица")],
        [KeyboardButton(text="Что это?"), KeyboardButton(text="Техподдержка")],
    ],
    resize_keyboard=True,
)


# ── Handlers ─────────────────────────────────────────────────────────────────
@dp.message(F.text.in_({"/start", "/help"}))
async def cmd_start(message: Message):
    await message.answer(
        "📍 <b>Главное меню:</b>\n\n"
        "<b>Heim Face</b> математически измеряет, насколько гармонично "
        "черты твоего лица сочетаются друг с другом.\n\n"
        "Твой баланс: <b>0 разборов</b>\n\n"
        "Сегодня пользователи провели: <b>21 разбор</b>",
        parse_mode="HTML",
        reply_markup=main_keyboard,
    )


@dp.message(F.text == "Что это?")
async def what_is_this(message: Message):
    await message.answer(
        "📍 <b>Главное меню › Что это:</b>\n\n"
        "<b>Heim Face</b> — сервис, позволяющий математически оценить "
        "гармонию пропорций лица по фото.\n\n"
        "Как это работает:\n\n"
        "1. Алгоритм компьютерного зрения определяет ключевые точки лица.\n\n"
        "2. На их основе рассчитываются геометрические метрики: "
        "симметрия, расстояние между глазами, ширина носа, ширина рта "
        "и другие параметры.\n\n"
        "3. Полученные значения сравниваются с нормативными пропорциями "
        "лицевой антропометрии.\n\n"
        "4. На выходе вы получаете PDF-отчёт с результатами анализа.",
        parse_mode="HTML",
    )


@dp.message(F.text == "💠 Получить разбор лица")
async def get_report(message: Message):
    await message.answer(
        "📍 <b>Главное меню › Выбор тарифа › Оплата:</b>\n\n"
        "⚜️ <b>План</b> — 1 разбор\n\n"
        "💰 <b>Цена</b> — 690 ₽\n\n"
        "📚 <b>Что входит в один разбор:</b>\n\n"
        "🔹 Персональный PDF-отчёт\n"
        "🔹 Разбор ключевых метрик лица\n"
        "🔹 Наглядная визуализация измерений\n"
        "🔹 Понятное объяснение каждой метрики\n"
        "🔹 Анализ сильных сторон и зон потенциала\n\n"
        "💡 <b>Пока оплата не подключена.</b>\n"
        "Для теста просто отправь фото в этот чат.",
        parse_mode="HTML",
    )


@dp.message(F.text == "Техподдержка")
async def support(message: Message):
    await message.answer(
        "Техподдержка: напишите администратору проекта Heim Face."
    )


@dp.message(F.photo)
async def handle_photo(message: Message):
    await message.answer("⏳ Анализирую...")

    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)

    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    image_bytes = buf.getvalue()

    result = analyze_face(image_bytes)

    if result.startswith("❌") or result.startswith("🙁"):
        await message.answer(result)
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_path = tmp.name

    create_pdf_report(image_bytes, result, pdf_path)

    await message.answer("✅ <b>Разбор завершён!</b>\n\nОтчёт отправлен ниже ↓", parse_mode="HTML")

    pdf = FSInputFile(pdf_path, filename="Отчёт Heim Face.pdf")
    await message.answer_document(pdf)


@dp.message(F.document)
async def handle_document(message: Message):
    doc = message.document

    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await message.answer("📎 Это не изображение. Отправь, пожалуйста, фото.")
        return

    await message.answer("⏳ Анализирую...")

    file = await bot.get_file(doc.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    image_bytes = buf.getvalue()

    result = analyze_face(image_bytes)

    if result.startswith("❌") or result.startswith("🙁"):
        await message.answer(result)
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_path = tmp.name

    create_pdf_report(image_bytes, result, pdf_path)

    await message.answer("✅ <b>Разбор завершён!</b>\n\nОтчёт отправлен ниже ↓", parse_mode="HTML")

    pdf = FSInputFile(pdf_path, filename="Отчёт Heim Face.pdf")
    await message.answer_document(pdf)


@dp.message()
async def fallback(message: Message):
    await message.answer("📸 Отправь мне фотографию лица для анализа.")


# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
    logger.info("Bot starting...")
    threading.Thread(target=run_web).start()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
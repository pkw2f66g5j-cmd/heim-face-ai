from flask import Flask
import threading
import os
import asyncio
import io
import logging
import math
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running"

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

from dotenv import load_dotenv

load_dotenv()

import cv2
import mediapipe as mp
import numpy as np
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Bot init ──────────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("BOT_TOKEN", "")          # set via env or paste here
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN is not set. Export it: export BOT_TOKEN=your_token")

bot = Bot(token=BOT_TOKEN)
dp  = Dispatcher()

# ── MediaPipe FaceMesh ────────────────────────────────────────────────────────
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
)

# ── Landmark indices (MediaPipe 468-point mesh) ───────────────────────────────
#   https://developers.google.com/mediapipe/solutions/vision/face_landmarker
IDX_FACE_LEFT  = 234   # left cheekbone (from viewer's right)
IDX_FACE_RIGHT = 454   # right cheekbone
IDX_NOSE_LEFT  = 129   # left ala of nose
IDX_NOSE_RIGHT = 358   # right ala of nose
IDX_MOUTH_LEFT = 61    # left corner of mouth
IDX_MOUTH_RIGHT= 291   # right corner of mouth
IDX_EYE_LEFT   = 33    # left eye inner/outer anchor
IDX_EYE_RIGHT  = 263   # right eye inner/outer anchor


def euclidean(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def analyze_face(image_bytes: bytes) -> str:
    """Return formatted analysis string or an error message."""
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

    face_w   = euclidean(pt(IDX_FACE_LEFT),   pt(IDX_FACE_RIGHT))
    nose_w   = euclidean(pt(IDX_NOSE_LEFT),   pt(IDX_NOSE_RIGHT))
    mouth_w  = euclidean(pt(IDX_MOUTH_LEFT),  pt(IDX_MOUTH_RIGHT))
    eye_dist = euclidean(pt(IDX_EYE_LEFT),    pt(IDX_EYE_RIGHT))

    # nose/mouth width as % of face width — handy visual proportion
    nose_pct  = nose_w  / face_w * 100
    mouth_pct = mouth_w / face_w * 100

    lines = [
        "📐 <b>Анализ лица</b>",
        "",
        f"📏 <b>Ширина лица:</b>        {face_w:.1f} px",
        f"👃 <b>Ширина носа:</b>         {nose_w:.1f} px  ({nose_pct:.1f}% от ширины лица)",
        f"👄 <b>Ширина рта:</b>          {mouth_w:.1f} px  ({mouth_pct:.1f}% от ширины лица)",
        f"👀 <b>Расстояние между глазами:</b> {eye_dist:.1f} px",
        "",
        "<i>Пиксельные значения зависят от размера фото.</i>",
    ]
    return "\n".join(lines)


# ── Handlers ──────────────────────────────────────────────────────────────────
@dp.message(F.text.in_({"/start", "/help"}))
async def cmd_start(message: Message):
    await message.answer(
        "👋 Привет! Я бот для анализа пропорций лица.\n\n"
        "Просто отправь мне <b>фотографию</b> (фронтальный вид), "
        "и я посчитаю:\n"
        "• ширину лица\n"
        "• ширину носа\n"
        "• ширину рта\n"
        "• расстояние между глазами",
        parse_mode="HTML",
    )


@dp.message(F.photo)
async def handle_photo(message: Message):
    await message.answer("⏳ Анализирую...")

    # Take the highest-resolution version of the photo
    photo = message.photo[-1]
    file  = await bot.get_file(photo.file_id)

    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    image_bytes = buf.getvalue()

    result = analyze_face(image_bytes)
    await message.answer(result, parse_mode="HTML")


@dp.message(F.document)
async def handle_document(message: Message):
    """Accept images sent as files (uncompressed upload)."""
    doc = message.document
    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await message.answer("📎 Это не изображение. Отправь, пожалуйста, фото.")
        return

    await message.answer("⏳ Анализирую...")
    file = await bot.get_file(doc.file_id)
    buf  = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    result = analyze_face(buf.getvalue())
    await message.answer(result, parse_mode="HTML")


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

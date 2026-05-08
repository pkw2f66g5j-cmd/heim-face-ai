from flask import Flask
import threading
import os
import asyncio
import io
import logging
import math
import tempfile
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

import cv2
import mediapipe as mp
import numpy as np

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile, KeyboardButton, ReplyKeyboardMarkup

from PIL import Image, ImageDraw

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ================== BRAND ==================
BOT_NAME = "Heim Face"
BOT_USERNAME = "@heim_face_bot"
PRICE_TEXT = "690 ₽"


# ================== WEB SERVER FOR RENDER ==================
app = Flask(__name__)

@app.route("/")
def home():
    return f"{BOT_NAME} bot is running"

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


# ================== LOGGING ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ================== BOT INIT ==================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN is not set")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


# ================== FONTS ==================
def setup_fonts():
    possible_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]

    try:
        if os.path.exists(possible_fonts[0]):
            pdfmetrics.registerFont(TTFont("MainFont", possible_fonts[0]))
            pdfmetrics.registerFont(TTFont("MainFontBold", possible_fonts[1]))
            return "MainFont", "MainFontBold"

        if os.path.exists(possible_fonts[2]):
            pdfmetrics.registerFont(TTFont("MainFont", possible_fonts[2]))
            pdfmetrics.registerFont(TTFont("MainFontBold", possible_fonts[3]))
            return "MainFont", "MainFontBold"
    except Exception:
        pass

    return "Helvetica", "Helvetica-Bold"


FONT_REGULAR, FONT_BOLD = setup_fonts()


# ================== MEDIAPIPE ==================
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
)


# ================== LANDMARKS ==================
IDX = {
    "face_left": 234,
    "face_right": 454,
    "chin": 152,
    "forehead": 10,

    "nose_bridge": 168,
    "nose_base": 2,
    "nose_left": 129,
    "nose_right": 358,

    "mouth_left": 61,
    "mouth_right": 291,
    "upper_lip": 13,
    "lower_lip": 14,
    "upper_lip_top": 0,
    "lower_lip_bottom": 17,

    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "right_eye_top": 386,
    "right_eye_bottom": 374,

    "left_brow": 105,
    "right_brow": 334,

    "jaw_left": 172,
    "jaw_right": 397,
    "cheek_left": 234,
    "cheek_right": 454,

    "forehead_left": 103,
    "forehead_right": 332,
}


# ================== NORMS ==================
NORMS = {
    "Симметрия лица": {
        "norm": 0.970,
        "sigma": 0.055,
        "formula": "Соотношение левой и правой сторон лица",
    },
    "Пропорции лица": {
        "norm": 0.890,
        "sigma": 0.055,
        "formula": "Высота лица / ширина скул",
    },
    "Вертикальный баланс": {
        "norm": 0.730,
        "sigma": 0.070,
        "formula": "Средняя треть лица / нижняя треть лица",
    },
    "Баланс скул и челюсти": {
        "norm": 1.350,
        "sigma": 0.080,
        "formula": "Ширина скул / ширина челюсти",
    },
    "Размер глаз": {
        "norm": 0.222,
        "sigma": 0.018,
        "formula": "Ширина глаза / ширина лица",
    },
    "Расстояние между глазами": {
        "norm": 0.265,
        "sigma": 0.020,
        "formula": "Расстояние между глазами / ширина лица",
    },
    "Наклон глаз": {
        "norm": 0.045,
        "sigma": 0.030,
        "formula": "Наклон уголков глаза / ширина глаза",
    },
    "Ширина носа": {
        "norm": 0.230,
        "sigma": 0.018,
        "formula": "Ширина крыльев носа / ширина лица",
    },
    "Ширина рта": {
        "norm": 0.405,
        "sigma": 0.030,
        "formula": "Ширина рта / ширина скул",
    },
    "Длина носа": {
        "norm": 0.420,
        "sigma": 0.035,
        "formula": "Длина носа / высота лица",
    },
    "Длина подбородка": {
        "norm": 0.290,
        "sigma": 0.030,
        "formula": "Нижняя губа → подбородок / высота лица",
    },
    "Контур подбородка": {
        "norm": 0.640,
        "sigma": 0.045,
        "formula": "Угол сужения подбородка",
    },
    "Нос к ширине рта": {
        "norm": 0.565,
        "sigma": 0.050,
        "formula": "Ширина носа / ширина рта",
    },
    "Биокулярная ширина": {
        "norm": 0.710,
        "sigma": 0.045,
        "formula": "Между внешними углами глаз / ширина лица",
    },
    "Ширина лба": {
        "norm": 0.910,
        "sigma": 0.055,
        "formula": "Ширина лба / ширина лица",
    },
    "Полнота губ": {
        "norm": 0.335,
        "sigma": 0.055,
        "formula": "Высота губ / ширина рта",
    },
    "Пропорции губ": {
        "norm": 0.630,
        "sigma": 0.090,
        "formula": "Верхняя губа / нижняя губа",
    },
    "Челюсть к ширине рта": {
        "norm": 1.840,
        "sigma": 0.140,
        "formula": "Ширина челюсти / ширина рта",
    },
    "Форма глаз": {
        "norm": 0.350,
        "sigma": 0.045,
        "formula": "Высота глаза / ширина глаза",
    },
    "Высота бровей": {
        "norm": 0.370,
        "sigma": 0.070,
        "formula": "Расстояние бровь-веко / ширина глаза",
    },
}


# ================== HELPERS ==================
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calc_score(value, norm, sigma):
    z = abs(value - norm) / sigma
    score = 10 - z * 2.2
    return round(max(0, min(10, score)), 2)


def wrap_text(text, max_chars=74):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        if len(current + " " + word) <= max_chars:
            current += (" " + word) if current else word
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def get_level(score):
    if score >= 8.5:
        return "Значительно выше среднего"
    if score >= 7.5:
        return "Выше среднего"
    if score >= 6.5:
        return "Средний уровень"
    return "Есть выраженные зоны потенциала"


def get_top_percent(score):
    if score >= 9:
        return 10
    if score >= 8:
        return 20
    if score >= 7.5:
        return 29
    if score >= 7:
        return 38
    if score >= 6.5:
        return 50
    return 65


def metric_description(name, value, norm, score):
    if score >= 8.5:
        closeness = "Показатель близок к статистической норме и визуально воспринимается гармонично."
    elif score >= 6:
        closeness = "Показатель немного отличается от медианного значения, но остаётся в естественном диапазоне."
    else:
        closeness = "Показатель заметно отличается от статистической нормы и формирует индивидуальную особенность лица."

    influence_map = {
        "Симметрия лица": "Симметрия создаёт ощущение целостности и аккуратности лица.",
        "Пропорции лица": "Общий баланс высоты и ширины определяет первое впечатление от формы лица.",
        "Вертикальный баланс": "Баланс третей влияет на восприятие зрелости и мужественности.",
        "Баланс скул и челюсти": "Соотношение скул и челюсти формирует силуэт нижней части лица.",
        "Размер глаз": "Пропорциональный размер глаз делает взгляд естественно выразительным.",
        "Расстояние между глазами": "Межглазничное расстояние влияет на открытость и мягкость взгляда.",
        "Наклон глаз": "Положительный наклон глаз часто воспринимается как уверенный и собранный взгляд.",
        "Ширина носа": "Ширина носа влияет на баланс центральной части лица.",
        "Ширина рта": "Ширина рта формирует выразительность нижней части лица.",
        "Длина носа": "Длина носа определяет визуальный баланс средней трети лица.",
        "Длина подбородка": "Подбородок завершает нижнюю треть и влияет на ощущение мужественности.",
        "Контур подбородка": "Контур подбородка формирует чёткость и силу нижней линии лица.",
        "Нос к ширине рта": "Соотношение носа и рта влияет на гармонию центральной и нижней зоны лица.",
        "Биокулярная ширина": "Биокулярная ширина определяет баланс глазной зоны относительно лица.",
        "Ширина лба": "Ширина лба влияет на восприятие верхней трети лица.",
        "Полнота губ": "Полнота губ влияет на выразительность и мягкость нижней части лица.",
        "Пропорции губ": "Соотношение верхней и нижней губ формирует индивидуальность зоны рта.",
        "Челюсть к ширине рта": "Соотношение челюсти и рта влияет на баланс нижней части лица.",
        "Форма глаз": "Форма глаз определяет степень открытости и эмоциональности взгляда.",
        "Высота бровей": "Высота бровей влияет на восприятие взгляда и верхней части лица.",
    }

    return closeness, influence_map.get(name, "Метрика влияет на общий баланс и гармонию лица.")


# ================== FACE ANALYSIS ==================
def analyze_face(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return None, "Не удалось декодировать изображение."

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return None, "Лицо не обнаружено. Отправьте чёткое фото строго анфас."

    lm = results.multi_face_landmarks[0].landmark

    def pt(name):
        idx = IDX[name]
        return lm[idx].x * w, lm[idx].y * h

    face_w = dist(pt("face_left"), pt("face_right"))
    face_h = dist(pt("nose_bridge"), pt("chin"))
    cheek_w = dist(pt("cheek_left"), pt("cheek_right"))
    jaw_w = dist(pt("jaw_left"), pt("jaw_right"))

    nose_w = dist(pt("nose_left"), pt("nose_right"))
    nose_len = dist(pt("nose_bridge"), pt("nose_base"))

    mouth_w = dist(pt("mouth_left"), pt("mouth_right"))

    left_eye_w = dist(pt("left_eye_outer"), pt("left_eye_inner"))
    right_eye_w = dist(pt("right_eye_inner"), pt("right_eye_outer"))
    eye_w = (left_eye_w + right_eye_w) / 2

    left_eye_h = dist(pt("left_eye_top"), pt("left_eye_bottom"))
    right_eye_h = dist(pt("right_eye_top"), pt("right_eye_bottom"))
    eye_h = (left_eye_h + right_eye_h) / 2

    eye_inner_dist = dist(pt("left_eye_inner"), pt("right_eye_inner"))
    biocular_w = dist(pt("left_eye_outer"), pt("right_eye_outer"))

    forehead_w = dist(pt("forehead_left"), pt("forehead_right"))

    upper_lip_h = dist(pt("upper_lip_top"), pt("upper_lip"))
    lower_lip_h = dist(pt("lower_lip"), pt("lower_lip_bottom"))
    lips_h = upper_lip_h + lower_lip_h

    chin_len = dist(pt("lower_lip"), pt("chin"))

    middle_third = dist(pt("nose_bridge"), pt("nose_base"))
    lower_third = dist(pt("nose_base"), pt("chin"))

    left_half = dist(pt("face_left"), pt("nose_bridge"))
    right_half = dist(pt("face_right"), pt("nose_bridge"))
    symmetry = min(left_half, right_half) / max(left_half, right_half)

    eye_tilt = abs(pt("right_eye_outer")[1] - pt("left_eye_outer")[1]) / max(biocular_w, 1)

    chin_contour = jaw_w / max(cheek_w, 1)

    brow_height = (
        dist(pt("left_brow"), pt("left_eye_top")) +
        dist(pt("right_brow"), pt("right_eye_top"))
    ) / 2 / max(eye_w, 1)

    values = {
        "Симметрия лица": symmetry,
        "Пропорции лица": face_h / face_w,
        "Вертикальный баланс": middle_third / lower_third,
        "Баланс скул и челюсти": cheek_w / jaw_w,
        "Размер глаз": eye_w / face_w,
        "Расстояние между глазами": eye_inner_dist / face_w,
        "Наклон глаз": eye_tilt,
        "Ширина носа": nose_w / face_w,
        "Ширина рта": mouth_w / cheek_w,
        "Длина носа": nose_len / face_h,
        "Длина подбородка": chin_len / face_h,
        "Контур подбородка": chin_contour,
        "Нос к ширине рта": nose_w / mouth_w,
        "Биокулярная ширина": biocular_w / face_w,
        "Ширина лба": forehead_w / face_w,
        "Полнота губ": lips_h / mouth_w,
        "Пропорции губ": upper_lip_h / max(lower_lip_h, 1),
        "Челюсть к ширине рта": jaw_w / mouth_w,
        "Форма глаз": eye_h / eye_w,
        "Высота бровей": brow_height,
    }

    metrics = []

    for name, value in values.items():
        norm = NORMS[name]["norm"]
        sigma = NORMS[name]["sigma"]
        score = calc_score(value, norm, sigma)

        metrics.append({
            "name": name,
            "value": round(value, 3),
            "norm": norm,
            "sigma": sigma,
            "score": score,
            "formula": NORMS[name]["formula"],
        })

    total_score = round(sum(m["score"] for m in metrics) / len(metrics), 2)

    sorted_metrics = sorted(metrics, key=lambda x: x["score"], reverse=True)
    strengths = sorted_metrics[:3]
    weak = sorted_metrics[-3:]

    data = {
        "score": total_score,
        "level": get_level(total_score),
        "top_percent": get_top_percent(total_score),
        "metrics": metrics,
        "strengths": strengths,
        "weak": weak,
        "date": datetime.now().strftime("%d.%m.%Y"),
    }

    return data, None


# ================== OVERLAY IMAGE ==================
def create_overlay_image(image_bytes: bytes, output_path: str):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((900, 900))

    draw = ImageDraw.Draw(image)
    w, h = image.size

    np_arr = np.array(image)
    img_rgb = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        image.save(output_path)
        return

    lm = results.multi_face_landmarks[0].landmark

    important_points = [
        10, 152, 234, 454, 129, 358, 61, 291,
        33, 133, 362, 263, 159, 145, 386, 374,
        172, 397, 103, 332
    ]

    def point(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    lines = [
        (234, 454),
        (129, 358),
        (61, 291),
        (33, 263),
        (10, 152),
        (172, 397),
        (103, 332),
    ]

    for a, b in lines:
        draw.line([point(a), point(b)], fill=(155, 45, 255), width=4)

    for idx in important_points:
        x, y = point(idx)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 224, 198))

    image.save(output_path)


# ================== PDF HELPERS ==================
def draw_page_bg(c, width, height):
    c.setFillColor(HexColor("#050814"))
    c.rect(0, 0, width, height, fill=1)

    c.setFillColor(HexColor("#0B1020"))
    c.roundRect(30, 30, width - 60, height - 60, 22, fill=1, stroke=0)


def draw_footer(c, width, page_num):
    c.setFillColor(HexColor("#626A86"))
    c.setFont(FONT_REGULAR, 8)
    c.drawCentredString(width / 2, 20, f"Telegram: {BOT_USERNAME} · {page_num} / 23")


def draw_wrapped(c, text, x, y, max_chars=72, line_height=14, font=FONT_REGULAR, size=10, color="#FFFFFF"):
    c.setFillColor(HexColor(color))
    c.setFont(font, size)

    for line in wrap_text(text, max_chars):
        c.drawString(x, y, line)
        y -= line_height

    return y


def draw_metric_bar(c, x, y, width, score):
    c.setFillColor(HexColor("#202840"))
    c.roundRect(x, y, width, 8, 4, fill=1, stroke=0)

    c.setFillColor(HexColor("#9B2DFF"))
    c.roundRect(x, y, width * score / 10, 8, 4, fill=1, stroke=0)


# ================== PDF REPORT ==================
def create_pdf_report(image_bytes: bytes, data: dict, output_path: str):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    overlay_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    create_overlay_image(image_bytes, overlay_path)

    # PAGE 1
    draw_page_bg(c, width, height)

    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont(FONT_BOLD, 28)
    c.drawCentredString(width / 2, height - 70, BOT_NAME)

    c.setFillColor(HexColor("#AEB7D5"))
    c.setFont(FONT_REGULAR, 11)
    c.drawCentredString(width / 2, height - 93, f"Telegram: {BOT_USERNAME}")
    c.drawCentredString(width / 2, height - 112, "Математический разбор пропорций лица")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((260, 300))
    img_reader = ImageReader(img)

    c.drawImage(img_reader, width / 2 - 120, height - 435, width=240, height=300, preserveAspectRatio=True)

    c.setFillColor(HexColor("#00E0C6"))
    c.setFont(FONT_BOLD, 44)
    c.drawCentredString(width / 2, height - 490, f"{data['score']:.2f}")

    c.setFillColor(HexColor("#AEB7D5"))
    c.setFont(FONT_REGULAR, 13)
    c.drawCentredString(width / 2, height - 512, "из 10")

    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont(FONT_BOLD, 14)
    c.drawCentredString(width / 2, height - 545, f"Вы входите в топ {data['top_percent']}% людей по гармонии лица")
    c.drawCentredString(width / 2, height - 565, f"Уровень: {data['level']}")

    c.setFillColor(HexColor("#111A2E"))
    c.roundRect(55, 95, width - 110, 135, 16, fill=1, stroke=0)

    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont(FONT_BOLD, 15)
    c.drawString(75, 205, "ОБЩЕЕ ВПЕЧАТЛЕНИЕ")

    impression = (
        "Лицо было проанализировано по ключевым геометрическим точкам. "
        "Алгоритм оценил симметрию, пропорции лица, глазную зону, нос, губы, "
        "подбородок и баланс нижней части лица. Итоговая оценка отражает "
        "среднюю близость ваших пропорций к антропометрическим нормам."
    )

    draw_wrapped(c, impression, 75, 180, max_chars=76, line_height=14, size=10, color="#DDE3FF")

    draw_footer(c, width, 1)
    c.showPage()

    # PAGE 2 PROFILE
    draw_page_bg(c, width, height)

    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont(FONT_BOLD, 24)
    c.drawString(55, height - 70, "Профиль метрик")

    c.setFont(FONT_BOLD, 13)
    c.drawString(55, height - 115, "Топ-3 сильных метрики")
    c.drawString(310, height - 115, "Топ-3 зоны потенциала")

    y = height - 145
    for m in data["strengths"]:
        c.setFillColor(HexColor("#00E0C6"))
        c.setFont(FONT_REGULAR, 11)
        c.drawString(55, y, f"• {m['name']} — {m['score']:.2f}")
        y -= 22

    y = height - 145
    for m in data["weak"]:
        c.setFillColor(HexColor("#FFB86B"))
        c.setFont(FONT_REGULAR, 11)
        c.drawString(310, y, f"• {m['name']} — {m['score']:.2f}")
        y -= 22

    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont(FONT_BOLD, 14)
    c.drawString(55, height - 230, "Вклад каждой метрики")

    y = height - 260
    for m in data["metrics"]:
        c.setFillColor(HexColor("#DDE3FF"))
        c.setFont(FONT_REGULAR, 9)
        c.drawString(55, y, m["name"])
        c.drawRightString(500, y, f"{m['score']:.2f}")
        draw_metric_bar(c, 55, y - 10, 445, m["score"])
        y -= 25

        if y < 80:
            break

    draw_footer(c, width, 2)
    c.showPage()

    # PAGES 3-22 METRICS
    page_num = 3

    for i, m in enumerate(data["metrics"], start=1):
        draw_page_bg(c, width, height)

        c.setFillColor(HexColor("#9B2DFF"))
        c.setFont(FONT_BOLD, 24)
        c.drawString(55, height - 70, f"{i:02d}")

        c.setFillColor(HexColor("#FFFFFF"))
        c.setFont(FONT_BOLD, 24)
        c.drawString(100, height - 70, m["name"])

        c.setFillColor(HexColor("#111A2E"))
        c.roundRect(55, height - 175, 165, 75, 14, fill=1, stroke=0)

        c.setFillColor(HexColor("#AEB7D5"))
        c.setFont(FONT_REGULAR, 9)
        c.drawString(75, height - 123, "Оценка метрики")

        c.setFillColor(HexColor("#00E0C6"))
        c.setFont(FONT_BOLD, 24)
        c.drawString(75, height - 155, f"{m['score']:.2f} / 10")

        c.setFillColor(HexColor("#111A2E"))
        c.roundRect(240, height - 175, 130, 75, 14, fill=1, stroke=0)

        c.setFillColor(HexColor("#AEB7D5"))
        c.setFont(FONT_REGULAR, 9)
        c.drawString(260, height - 123, "Ваш показатель")

        c.setFillColor(HexColor("#FFFFFF"))
        c.setFont(FONT_BOLD, 21)
        c.drawString(260, height - 155, f"{m['value']:.3f}")

        c.setFillColor(HexColor("#111A2E"))
        c.roundRect(390, height - 175, 130, 75, 14, fill=1, stroke=0)

        c.setFillColor(HexColor("#AEB7D5"))
        c.setFont(FONT_REGULAR, 9)
        c.drawString(410, height - 123, "Норма")

        c.setFillColor(HexColor("#FFFFFF"))
        c.setFont(FONT_BOLD, 21)
        c.drawString(410, height - 155, f"{m['norm']:.3f}")

        c.setFillColor(HexColor("#AEB7D5"))
        c.setFont(FONT_REGULAR, 11)
        c.drawString(55, height - 215, m["formula"])

        c.setFillColor(HexColor("#FFFFFF"))
        c.setFont(FONT_BOLD, 15)
        c.drawString(55, height - 255, "Близость к норме")

        closeness, influence = metric_description(m["name"], m["value"], m["norm"], m["score"])

        z = abs(m["value"] - m["norm"]) / m["sigma"]

        paragraph = (
            f"Ваш показатель составляет {m['value']:.3f} при норме {m['norm']:.3f}. "
            f"Отклонение от медианного значения составляет примерно {z:.2f}σ. "
            f"{closeness}"
        )

        y = draw_wrapped(c, paragraph, 55, height - 285, max_chars=82, line_height=15, size=10, color="#DDE3FF")

        c.setFillColor(HexColor("#FFFFFF"))
        c.setFont(FONT_BOLD, 15)
        c.drawString(55, y - 25, "ВЛИЯНИЕ")

        draw_wrapped(c, influence, 55, y - 50, max_chars=82, line_height=15, size=10, color="#DDE3FF")

        if i in [1, 2, 3, 8, 9, 18]:
            c.drawImage(ImageReader(overlay_path), 180, 70, width=240, height=240, preserveAspectRatio=True)

        draw_footer(c, width, page_num)
        page_num += 1
        c.showPage()

    # PAGE 23 RECOMMENDATIONS
    draw_page_bg(c, width, height)

    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont(FONT_BOLD, 24)
    c.drawString(55, height - 70, "Рекомендации")

    c.setFillColor(HexColor("#AEB7D5"))
    c.setFont(FONT_REGULAR, 12)
    c.drawString(55, height - 95, "Персональные советы по улучшению")

    c.setFillColor(HexColor("#111A2E"))
    c.roundRect(55, height - 330, width - 110, 195, 16, fill=1, stroke=0)

    c.setFillColor(HexColor("#00E0C6"))
    c.setFont(FONT_BOLD, 15)
    c.drawString(75, height - 165, "Что уже отлично")

    y = height - 195
    for m in data["strengths"]:
        text = f"{m['name']} — одна из сильных сторон вашего лица. Показатель близок к норме и поддерживает общее впечатление гармонии."
        y = draw_wrapped(c, text, 75, y, max_chars=74, line_height=14, size=10, color="#DDE3FF")
        y -= 8

    c.setFillColor(HexColor("#111A2E"))
    c.roundRect(55, height - 610, width - 110, 240, 16, fill=1, stroke=0)

    c.setFillColor(HexColor("#FFB86B"))
    c.setFont(FONT_BOLD, 15)
    c.drawString(75, height - 400, "Что можно улучшить")

    y = height - 430
    for m in data["weak"]:
        text = f"{m['name']} — зона потенциала. Лёгкая визуальная коррекция через уход, причёску, брови или щетину может сделать пропорции более сбалансированными."
        y = draw_wrapped(c, text, 75, y, max_chars=74, line_height=14, size=10, color="#DDE3FF")
        y -= 8

    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont(FONT_BOLD, 13)
    c.drawString(75, 140, "Советы для фото")

    tips = (
        "Свет спереди, камера на уровне глаз, нейтральное выражение лица, "
        "голова без наклонов, лицо строго анфас. Чем качественнее фото, тем точнее расчёт."
    )

    draw_wrapped(c, tips, 75, 115, max_chars=78, line_height=14, size=10, color="#DDE3FF")

    c.setFillColor(HexColor("#AEB7D5"))
    c.setFont(FONT_REGULAR, 9)
    c.drawString(75, 65, f"Дата разбора: {data['date']}")

    draw_footer(c, width, 23)

    c.save()


# ================== KEYBOARDS ==================
main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="💠 Хочу получить свой разбор")],
        [KeyboardButton(text="Что это?"), KeyboardButton(text="Техподдержка")],
    ],
    resize_keyboard=True,
)


# ================== HANDLERS ==================
@dp.message(F.text.in_({"/start", "/help"}))
async def cmd_start(message: Message):
    await message.answer(
        "📍 <b>Главное меню:</b>\n\n"
        f"<b>{BOT_NAME}</b> математически измеряет, насколько гармонично "
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
        f"<b>{BOT_NAME}</b> — сервис, позволяющий математически оценить "
        "гармонию пропорций лица по фотографии.\n\n"
        "Алгоритм определяет ключевые точки лица, рассчитывает пропорции, "
        "сравнивает их с нормативными значениями лицевой антропометрии "
        "и формирует PDF-отчёт.\n\n"
        "В отчёт входят: итоговая оценка, 20 метрик, визуализация точек, "
        "сильные стороны и зоны потенциала.",
        parse_mode="HTML",
    )


@dp.message(F.text == "💠 Хочу получить свой разбор")
async def get_report(message: Message):
    await message.answer(
        "📍 <b>Главное меню › Выбор тарифа › Оплата:</b>\n\n"
        "⚜️ <b>План</b> — 1 разбор\n\n"
        f"💰 <b>Цена</b> — {PRICE_TEXT}\n\n"
        "📚 <b>Что входит в один разбор:</b>\n\n"
        "🔹 Персональный PDF-отчёт на 23 страницы\n"
        "🔹 Разбор 20 ключевых метрик лица\n"
        "🔹 Сравнение с нормативными значениями\n"
        "🔹 Наглядная визуализация измерений\n"
        "🔹 Понятное объяснение каждой метрики\n"
        "🔹 Анализ сильных сторон и зон потенциала\n\n"
        "💡 <b>Пока оплата не подключена.</b>\n\n"
        "Для теста просто отправь фото в этот чат.",
        parse_mode="HTML",
    )


@dp.message(F.text == "Техподдержка")
async def support(message: Message):
    await message.answer(
        "Техподдержка: напишите администратору проекта."
    )


@dp.message(F.photo)
async def handle_photo(message: Message):
    await message.answer("⏳ Анализирую лицо...\n\nЭто может занять несколько секунд.")

    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)

    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    image_bytes = buf.getvalue()

    data, error = analyze_face(image_bytes)

    if error:
        await message.answer(f"❌ {error}")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_path = tmp.name

    create_pdf_report(image_bytes, data, pdf_path)

    await message.answer("✅ <b>Разбор завершён!</b>\n\nОтчёт отправлен ниже ↓", parse_mode="HTML")

    pdf = FSInputFile(pdf_path, filename=f"Отчёт {BOT_NAME}.pdf")
    await message.answer_document(pdf)


@dp.message(F.document)
async def handle_document(message: Message):
    doc = message.document

    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await message.answer("📎 Это не изображение. Отправь, пожалуйста, фото.")
        return

    await message.answer("⏳ Анализирую лицо...\n\nЭто может занять несколько секунд.")

    file = await bot.get_file(doc.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    image_bytes = buf.getvalue()

    data, error = analyze_face(image_bytes)

    if error:
        await message.answer(f"❌ {error}")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_path = tmp.name

    create_pdf_report(image_bytes, data, pdf_path)

    await message.answer("✅ <b>Разбор завершён!</b>\n\nОтчёт отправлен ниже ↓", parse_mode="HTML")

    pdf = FSInputFile(pdf_path, filename=f"Отчёт {BOT_NAME}.pdf")
    await message.answer_document(pdf)


@dp.message()
async def fallback(message: Message):
    await message.answer("📸 Отправь мне фотографию лица для анализа.")


# ================== ENTRY POINT ==================
async def main():
    logger.info("Bot starting...")
    threading.Thread(target=run_web).start()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
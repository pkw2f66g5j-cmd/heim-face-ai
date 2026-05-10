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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile, KeyboardButton, ReplyKeyboardMarkup, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

from PIL import Image, ImageDraw, ImageFilter

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ================== BRAND ==================
BOT_NAME = "Heim Face"
BOT_USERNAME = "@heim_face_bot"
PRICE_TEXT = "1 000 ₽"


# ================== ROYAL VELVET PALETTE ==================
COLOR_BG = "#14101A"
COLOR_BG_SOFT = "#1C1622"
COLOR_TITLE = "#E8D5A0"
COLOR_ACCENT = "#E5A8A1"
COLOR_TEXT = "#F0E6D8"
COLOR_TEXT_SOFT = "#B8A99A"
COLOR_TEXT_MUTED = "#7A6F66"
COLOR_BAR_BG = "#2A2230"
COLOR_LINE = "#3A2F3F"

METRIC_COLORS = [
    "#D4AF37",
    "#E8A87C",
    "#D4A5C5",
    "#A687C9",
    "#C9A582",
    "#D4AF37",
    "#E8A87C",
    "#D4A5C5",
    "#A687C9",
    "#C9A582",
    "#D4AF37",
    "#E8A87C",
    "#D4A5C5",
    "#A687C9",
    "#C9A582",
    "#D4AF37",
    "#E8A87C",
    "#D4A5C5",
    "#A687C9",
    "#C9A582",
]


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
dp = Dispatcher(storage=MemoryStorage())


# ================== FSM STATES ==================
class AnalysisStates(StatesGroup):
    waiting_for_gender = State()
    waiting_for_photo = State()


# ================== USER DATA ==================
user_gender = {}


# ================== FONTS ==================
def setup_fonts():
    possible_fonts = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ]

    for regular, bold in possible_fonts:
        try:
            if os.path.exists(regular):
                pdfmetrics.registerFont(TTFont("MainFont", regular))
                pdfmetrics.registerFont(TTFont("MainFontBold", bold))
                return "MainFont", "MainFontBold"
        except Exception:
            continue

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
    "nose_top": 6,
    "nose_base": 2,
    "nose_tip": 4,
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

    "left_brow_inner": 55,
    "left_brow_mid": 105,
    "left_brow_outer": 70,
    "right_brow_inner": 285,
    "right_brow_mid": 334,
    "right_brow_outer": 300,

    "jaw_left": 172,
    "jaw_right": 397,
    "jaw_left_lower": 150,
    "jaw_right_lower": 379,
    "cheek_left": 234,
    "cheek_right": 454,

    "forehead_left": 103,
    "forehead_right": 332,
    "forehead_top": 10,
}


# ================== NORMS MALE ==================
NORMS_MALE = {
    "Симметрия лица": {"norm": 0.970, "sigma": 0.055, "formula": "Зеркальность точек относительно центральной оси"},
    "Пропорции лица": {"norm": 0.890, "sigma": 0.055, "formula": "Высота лица (переносица → подбородок) / ширина скул"},
    "Вертикальный баланс": {"norm": 0.730, "sigma": 0.070, "formula": "Средняя треть лица / нижняя треть лица"},
    "Баланс скул и челюсти": {"norm": 1.355, "sigma": 0.080, "formula": "Ширина скул / ширина челюсти"},
    "Размер глаз": {"norm": 0.223, "sigma": 0.018, "formula": "Ширина глаза / ширина лица"},
    "Расстояние между глазами": {"norm": 0.268, "sigma": 0.020, "formula": "Расстояние между глазами / ширина лица"},
    "Наклон глаз": {"norm": 0.040, "sigma": 0.030, "formula": "Наклон уголков глаза / ширина глаза"},
    "Ширина носа": {"norm": 0.233, "sigma": 0.018, "formula": "Ширина крыльев носа / ширина лица"},
    "Ширина рта": {"norm": 0.402, "sigma": 0.030, "formula": "Ширина рта / ширина скул"},
    "Длина носа": {"norm": 0.421, "sigma": 0.035, "formula": "Длина носа / высота лица"},
    "Длина подбородка": {"norm": 0.286, "sigma": 0.030, "formula": "Нижняя губа → подбородок / высота лица"},
    "Контур подбородка": {"norm": 0.632, "sigma": 0.045, "formula": "Угол сужения подбородка"},
    "Нос к ширине рта": {"norm": 0.575, "sigma": 0.050, "formula": "Ширина носа / ширина рта"},
    "Биокулярная ширина": {"norm": 0.711, "sigma": 0.045, "formula": "Между внешними углами глаз / ширина лица"},
    "Ширина лба": {"norm": 0.916, "sigma": 0.055, "formula": "Ширина лба / ширина лица"},
    "Полнота губ": {"norm": 0.339, "sigma": 0.055, "formula": "Высота губ / ширина рта"},
    "Пропорции губ": {"norm": 0.634, "sigma": 0.090, "formula": "Верхняя губа / нижняя губа"},
    "Челюсть к ширине рта": {"norm": 1.841, "sigma": 0.140, "formula": "Ширина челюсти / ширина рта"},
    "Форма глаз": {"norm": 0.350, "sigma": 0.045, "formula": "Высота глаза / ширина глаза"},
    "Высота бровей": {"norm": 0.377, "sigma": 0.070, "formula": "Расстояние от нижнего края брови до века / ширина глаза"},
}


# ================== NORMS FEMALE ==================
NORMS_FEMALE = {
    "Симметрия лица": {"norm": 0.972, "sigma": 0.050, "formula": "Зеркальность точек относительно центральной оси"},
    "Пропорции лица": {"norm": 0.920, "sigma": 0.055, "formula": "Высота лица (переносица → подбородок) / ширина скул"},
    "Вертикальный баланс": {"norm": 0.760, "sigma": 0.070, "formula": "Средняя треть лица / нижняя треть лица"},
    "Баланс скул и челюсти": {"norm": 1.420, "sigma": 0.080, "formula": "Ширина скул / ширина челюсти"},
    "Размер глаз": {"norm": 0.232, "sigma": 0.018, "formula": "Ширина глаза / ширина лица"},
    "Расстояние между глазами": {"norm": 0.265, "sigma": 0.020, "formula": "Расстояние между глазами / ширина лица"},
    "Наклон глаз": {"norm": 0.055, "sigma": 0.030, "formula": "Наклон уголков глаза / ширина глаза"},
    "Ширина носа": {"norm": 0.215, "sigma": 0.018, "formula": "Ширина крыльев носа / ширина лица"},
    "Ширина рта": {"norm": 0.395, "sigma": 0.030, "formula": "Ширина рта / ширина скул"},
    "Длина носа": {"norm": 0.405, "sigma": 0.035, "formula": "Длина носа / высота лица"},
    "Длина подбородка": {"norm": 0.265, "sigma": 0.030, "formula": "Нижняя губа → подбородок / высота лица"},
    "Контур подбородка": {"norm": 0.595, "sigma": 0.045, "formula": "Угол сужения подбородка"},
    "Нос к ширине рта": {"norm": 0.545, "sigma": 0.050, "formula": "Ширина носа / ширина рта"},
    "Биокулярная ширина": {"norm": 0.708, "sigma": 0.045, "formula": "Между внешними углами глаз / ширина лица"},
    "Ширина лба": {"norm": 0.905, "sigma": 0.055, "formula": "Ширина лба / ширина лица"},
    "Полнота губ": {"norm": 0.395, "sigma": 0.055, "formula": "Высота губ / ширина рта"},
    "Пропорции губ": {"norm": 0.665, "sigma": 0.090, "formula": "Верхняя губа / нижняя губа"},
    "Челюсть к ширине рта": {"norm": 1.785, "sigma": 0.140, "formula": "Ширина челюсти / ширина рта"},
    "Форма глаз": {"norm": 0.385, "sigma": 0.045, "formula": "Высота глаза / ширина глаза"},
    "Высота бровей": {"norm": 0.420, "sigma": 0.070, "formula": "Расстояние от нижнего края брови до века / ширина глаза"},
}


def get_norms(gender):
    return NORMS_FEMALE if gender == "female" else NORMS_MALE
# ================== HELPERS ==================
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calc_score(value, norm, sigma):
    z = abs(value - norm) / sigma
    score = 10 - z * 2.2
    return round(max(0, min(10, score)), 2)


def calc_z(value, norm, sigma):
    return (value - norm) / sigma


def wrap_text(text, max_chars=74):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        if len(current + " " + word) <= max_chars:
            current += (" " + word) if current else word
        else:
            if current:
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
    if score >= 5.5:
        return "Чуть ниже среднего"
    return "Есть выраженные зоны потенциала"


def get_top_percent(score):
    if score >= 9.5:
        return 5
    if score >= 9:
        return 10
    if score >= 8.5:
        return 15
    if score >= 8:
        return 20
    if score >= 7.5:
        return 25
    if score >= 7:
        return 35
    if score >= 6.5:
        return 45
    if score >= 6:
        return 55
    return 70


def closeness_dots(score):
    if score >= 9:
        return 5
    if score >= 7.5:
        return 4
    if score >= 6:
        return 3
    if score >= 4:
        return 2
    return 1


def deviation_word(z_abs):
    if z_abs < 0.3:
        return "практически совпадает с нормой"
    if z_abs < 0.7:
        return "близко к норме"
    if z_abs < 1.2:
        return "немного отличается от нормы"
    if z_abs < 1.8:
        return "заметно отличается от нормы"
    return "значительно отличается от нормы"


def direction_word(z, higher_word="выше нормы", lower_word="ниже нормы", equal_word="на уровне нормы"):
    if abs(z) < 0.3:
        return equal_word
    return higher_word if z > 0 else lower_word


# ================== FACE ANALYSIS ==================
def analyze_face(image_bytes: bytes, gender: str):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return None, "Не удалось декодировать изображение."

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return None, "Лицо не обнаружено. Отправь чёткое фото строго анфас, при хорошем свете."

    lm = results.multi_face_landmarks[0].landmark

    def pt(name):
        idx = IDX[name]
        return lm[idx].x * w, lm[idx].y * h

    # Базовые размеры
    face_w = dist(pt("face_left"), pt("face_right"))
    face_h = dist(pt("nose_bridge"), pt("chin"))
    cheek_w = dist(pt("cheek_left"), pt("cheek_right"))
    jaw_w = dist(pt("jaw_left"), pt("jaw_right"))
    jaw_w_lower = dist(pt("jaw_left_lower"), pt("jaw_right_lower"))

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

    # Симметрия — детальная
    midline_x = (pt("forehead")[0] + pt("chin")[0]) / 2

    def axis_dev(point_name):
        x = pt(point_name)[0]
        return (x - midline_x) / face_w

    dev_inner_left_eye = axis_dev("left_eye_inner")
    dev_inner_right_eye = axis_dev("right_eye_inner")
    dev_outer_left_eye = axis_dev("left_eye_outer")
    dev_outer_right_eye = axis_dev("right_eye_outer")
    dev_mouth_left = axis_dev("mouth_left")
    dev_mouth_right = axis_dev("mouth_right")
    dev_nose_left = axis_dev("nose_left")
    dev_nose_right = axis_dev("nose_right")

    sym_inner_eye = abs(abs(dev_inner_left_eye) - abs(dev_inner_right_eye))
    sym_outer_eye = abs(abs(dev_outer_left_eye) - abs(dev_outer_right_eye))
    sym_mouth = abs(abs(dev_mouth_left) - abs(dev_mouth_right))
    sym_nose = abs(abs(dev_nose_left) - abs(dev_nose_right))

    avg_sym_dev = (sym_inner_eye + sym_outer_eye + sym_mouth + sym_nose) / 4
    symmetry = max(0, 1 - avg_sym_dev * 5)

    eye_tilt = abs(pt("right_eye_outer")[1] - pt("left_eye_outer")[1]) / max(biocular_w, 1)

    chin_contour = jaw_w_lower / max(jaw_w, 1)

    brow_height = (
        dist(pt("left_brow_mid"), pt("left_eye_top")) +
        dist(pt("right_brow_mid"), pt("right_eye_top"))
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

    norms = get_norms(gender)
    metrics = []

    for name, value in values.items():
        norm = norms[name]["norm"]
        sigma = norms[name]["sigma"]
        score = calc_score(value, norm, sigma)
        z = calc_z(value, norm, sigma)

        metrics.append({
            "name": name,
            "value": round(value, 4),
            "norm": norm,
            "sigma": sigma,
            "score": score,
            "z": z,
            "formula": norms[name]["formula"],
        })

    total_score = round(sum(m["score"] for m in metrics) / len(metrics), 2)

    sorted_metrics = sorted(metrics, key=lambda x: x["score"], reverse=True)
    strengths = sorted_metrics[:3]
    weak = sorted_metrics[-3:][::-1]

    extra = {
        "sym_inner_eye_pct": sym_inner_eye * 100,
        "sym_outer_eye_pct": sym_outer_eye * 100,
        "sym_mouth_pct": sym_mouth * 100,
        "sym_nose_pct": sym_nose * 100,
        "dev_inner_left_eye": dev_inner_left_eye,
        "dev_inner_right_eye": dev_inner_right_eye,
        "dev_outer_left_eye": dev_outer_left_eye,
        "dev_outer_right_eye": dev_outer_right_eye,
        "nose_w_ratio": nose_w / face_w,
        "mouth_w_ratio": mouth_w / face_w,
        "jaw_w_ratio": jaw_w / face_w,
        "upper_lip_ratio": upper_lip_h / max(face_h, 1),
        "lower_lip_ratio": lower_lip_h / max(face_h, 1),
    }

    data = {
        "score": total_score,
        "level": get_level(total_score),
        "top_percent": get_top_percent(total_score),
        "metrics": metrics,
        "strengths": strengths,
        "weak": weak,
        "extra": extra,
        "gender": gender,
        "date": datetime.now().strftime("%d.%m.%Y"),
    }

    return data, None
# ================== TEXT GENERATORS ==================

def _gender_form(gender, male, female):
    return female if gender == "female" else male


def text_symmetry(m, extra, gender):
    z_abs = abs(m["z"])
    val, norm = m["value"], m["norm"]
    dev_word = deviation_word(z_abs)

    if m["score"] >= 8.5:
        title = "Лицо демонстрирует высокую степень симметрии с минимальными отклонениями."
    elif m["score"] >= 6:
        title = "Симметрия лица находится в естественном диапазоне с небольшими отклонениями."
    else:
        title = "Симметрия имеет заметные отклонения, формирующие индивидуальные особенности."

    p1 = (
        f"Симметрия оценивается через отклонение ключевых точек лица от центральной оси, "
        f"проходящей через переносицу и подбородок. Общий показатель {val:.4f} "
        f"{dev_word} ({norm:.4f}, отклонение ~{z_abs:.2f}σ)."
    )

    inner_eye = extra["sym_inner_eye_pct"]
    outer_eye = extra["sym_outer_eye_pct"]
    mouth_dev = extra["sym_mouth_pct"]
    nose_dev = extra["sym_nose_pct"]

    deviations = []
    if inner_eye > 0.3:
        side = "слева" if extra["dev_inner_left_eye"] < extra["dev_inner_right_eye"] else "справа"
        deviations.append(f"внутренние уголки глаз: {side} на ~{inner_eye:.1f}% дальше от оси")
    if outer_eye > 0.3:
        side = "слева" if extra["dev_outer_left_eye"] < extra["dev_outer_right_eye"] else "справа"
        deviations.append(f"внешние уголки глаз: {side} на ~{outer_eye:.1f}% дальше от оси")
    if mouth_dev > 0.3:
        deviations.append(f"уголки рта смещены на ~{mouth_dev:.1f}%")
    if nose_dev > 0.2:
        deviations.append(f"крылья носа смещены на ~{nose_dev:.1f}%")

    if deviations:
        p2 = "Наиболее заметные отклонения: " + "; ".join(deviations[:2]) + "."
    else:
        p2 = "Все ключевые точки расположены практически зеркально, отклонения в пределах погрешности измерения."

    if m["score"] >= 8:
        p3 = (
            "Согласно исследованиям Rhodes — Facial symmetry and the perception of beauty, "
            "высокая симметрия воспринимается окружающими как признак генетического здоровья "
            "и повышает общую привлекательность."
        )
        infl = "Высокая симметрия создаёт ощущение гармоничного и приятного лица, формируя позитивное первое впечатление."
    else:
        p3 = (
            "Исследования Rhodes — Facial symmetry and the perception of beauty показывают, что "
            "лёгкая асимметрия — естественная особенность, встречающаяся у большинства людей, "
            "и часто формирует характерные узнаваемые черты."
        )
        infl = "Лёгкая асимметрия добавляет характерности и индивидуальности — большинство привлекательных лиц не идеально симметричны."

    return title, p1, p2, p3, infl


def text_proportions(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = _gender_form(gender, "слегка удлинённое лицо", "слегка удлинённое лицо")
    elif z < -0.3:
        descr = "более округлое лицо"
    else:
        descr = "классически пропорциональное лицо"

    if m["score"] >= 8.5:
        title = f"Пропорции лица близки к классическому овалу{' с лёгким удлинением' if z > 0 else ''}."
    elif m["score"] >= 6:
        title = f"Пропорции лица отличаются от среднего, формируя {descr}."
    else:
        title = f"Пропорции лица заметно отклоняются от классического овала."

    p1 = (
        f"Метрика измеряет соотношение высоты лица (от переносицы до подбородка) к ширине скул. "
        f"Значение {val:.4f} {direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ), что указывает на {descr}."
    )

    if z > 0.3:
        p2 = _gender_form(
            gender,
            "Визуально это проявляется в вытянутом, подтянутом силуэте лица. У мужчин такие пропорции считаются одними из наиболее привлекательных, ассоциируясь со зрелостью и мужественностью.",
            "Удлинённые пропорции придают лицу элегантность и утончённость. Такая форма ассоциируется с выразительностью и часто встречается у моделей."
        )
    elif z < -0.3:
        p2 = _gender_form(
            gender,
            "Более округлое лицо визуально воспринимается как молодое и располагающее. У мужчин это смягчает общее впечатление и добавляет дружелюбности.",
            "Округлые пропорции лица ассоциируются с молодостью, мягкостью и женственностью — это одна из классических черт миловидности."
        )
    else:
        p2 = "Классические пропорции воспринимаются как наиболее гармоничные и универсально привлекательные."

    p3 = (
        "Исследования Perrett et al. — Facial attractiveness judgements reflect learning of parental "
        "age characteristics показали, что пропорции лица напрямую влияют на восприятие зрелости и статуса."
    )

    infl = _gender_form(
        gender,
        f"{'Удлинённые' if z > 0 else ('Округлые' if z < 0 else 'Сбалансированные')} пропорции формируют общее впечатление от формы лица — это одна из базовых характеристик восприятия.",
        f"{'Удлинённые' if z > 0 else ('Мягкие округлые' if z < 0 else 'Гармоничные')} пропорции определяют первое впечатление от овала лица."
    )

    return title, p1, p2, p3, infl


def text_vertical_balance(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z < -0.3:
        descr = "нижняя часть лица пропорционально длиннее средней"
    elif z > 0.3:
        descr = "средняя часть лица доминирует над нижней"
    else:
        descr = "средняя и нижняя трети сбалансированы"

    if m["score"] >= 8.5:
        title = "Вертикальные пропорции лица сбалансированы."
    elif z < -0.3:
        title = _gender_form(
            gender,
            "Нижняя треть лица выражена сильнее средней, что подчёркивает мужественность.",
            "Нижняя треть лица выражена сильнее средней."
        )
    else:
        title = "Средняя треть лица доминирует, формируя индивидуальные пропорции."

    p1 = (
        f"Метрика сравнивает среднюю треть лица (переносица → основание носа) с нижней третью "
        f"(основание носа → подбородок). Значение {val:.4f} {direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ) — {descr}."
    )

    if z < -0.3:
        p2 = _gender_form(
            gender,
            "На лице это проявляется как выраженная подбородочная зона и удлинённая нижняя часть. Развитая нижняя треть — один из ключевых маркеров мужественности, подчёркивающий волевой характер.",
            "Удлинённая нижняя треть лица создаёт выразительный силуэт. У женщин это часто компенсируется мягкими чертами и формирует характерный образ."
        )
    elif z > 0.3:
        p2 = "Доминирование средней трети визуально удлиняет нос и центральную зону, что создаёт эффект утончённости и аристократичности."
    else:
        p2 = "Сбалансированные трети — признак классических пропорций, воспринимаемых как наиболее гармоничные."

    p3 = (
        "Согласно Cunningham et al. — What do women want? Facialmetric assessment, "
        "вертикальный баланс лица — один из ключевых факторов восприятия привлекательности и доминантности."
    )

    infl = "Вертикальные пропорции формируют общее впечатление от лица и влияют на восприятие зрелости, силы характера и пола."

    return title, p1, p2, p3, infl


def text_cheek_jaw(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = "скулы заметно выделяются на фоне челюсти"
    elif z < -0.3:
        descr = "челюсть приближена по ширине к скулам"
    else:
        descr = "скулы и челюсть гармонично сбалансированы"

    if m["score"] >= 8.5:
        title = "Соотношение скул и челюсти близко к классическому."
    elif z > 0:
        title = _gender_form(
            gender,
            "Скулы умеренно шире челюсти, формируя классическую мужскую форму лица.",
            "Скулы выделяются на фоне челюсти, формируя выраженный овал."
        )
    else:
        title = "Челюсть приближена к скулам, формируя более квадратный силуэт."

    p1 = (
        f"Метрика измеряет соотношение ширины скул к ширине челюсти. "
        f"Значение {val:.4f} {direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ) — {descr}."
    )

    if z > 0.3:
        p2 = _gender_form(
            gender,
            "Визуально это создаёт сужение от скул к челюсти, формируя подтянутый контур. Для мужского лица такое соотношение гармонично — оно подчёркивает скуловую зону без чрезмерного контраста.",
            "Узкая челюсть и широкие скулы создают элегантный овал лица — одну из наиболее ценимых черт женской внешности."
        )
    elif z < -0.3:
        p2 = _gender_form(
            gender,
            "Широкая челюсть в сочетании со скулами создаёт сильный квадратный контур — один из классических маркеров мужественности и доминантности.",
            "Широкая челюсть формирует структурное лицо с выраженным характером."
        )
    else:
        p2 = "Сбалансированное соотношение создаёт универсально привлекательную форму лица."

    p3 = (
        "Исследования Rhodes — The evolutionary psychology of facial beauty подтверждают, что "
        "соотношение скул и челюсти — один из ключевых эстетических маркеров."
    )

    infl = _gender_form(
        gender,
        "Соотношение скул и челюсти определяет силуэт нижней части лица и влияет на восприятие мужественности.",
        "Соотношение скул и челюсти формирует овал лица — одну из главных черт женской привлекательности."
    )

    return title, p1, p2, p3, infl


def text_eye_size(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = "глаза немного крупнее среднего"
    elif z < -0.3:
        descr = "глаза немного компактнее среднего"
    else:
        descr = "идеально пропорциональный размер глаз"

    if m["score"] >= 8.5:
        title = "Размер глаз практически идеально соответствует классическим пропорциям."
    else:
        title = f"Размер глаз отличается от среднего — {descr}."

    p1 = (
        f"Метрика оценивает ширину глаза относительно ширины лица. Значение {val:.4f} "
        f"{deviation_word(z_abs)} {norm:.4f} (отклонение ~{z_abs:.2f}σ), что указывает на {descr}."
    )

    if abs(z) < 0.5:
        p2 = "Глаза выглядят гармонично вписанными в общие черты лица — не слишком крупные и не мелкие. Такой баланс позволяет другим чертам равномерно участвовать в формировании впечатления."
    elif z > 0:
        p2 = "Крупные глаза визуально привлекают внимание и часто ассоциируются с открытостью и эмоциональностью. Это одна из черт, повышающих общую выразительность лица."
    else:
        p2 = "Компактные глаза создают более сосредоточенный взгляд, что часто ассоциируется со зрелостью и серьёзностью."

    p3 = (
        "Согласно работам Farkas — Anthropometry of the Head and Face, пропорциональный размер глаз — "
        "один из базовых критериев лицевой гармонии."
    )

    infl = "Размер глаз напрямую влияет на выразительность взгляда и общее восприятие лица."

    return title, p1, p2, p3, infl


def text_eye_distance(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = "глаза расположены немного шире, чем в среднем"
    elif z < -0.3:
        descr = "глаза посажены немного ближе, чем в среднем"
    else:
        descr = "идеально пропорциональное расположение глаз"

    if m["score"] >= 8.5:
        title = "Расстояние между глазами близко к классической норме."
    elif z > 0:
        title = "Глаза посажены чуть шире среднего, что придаёт взгляду открытость."
    else:
        title = "Глаза посажены чуть ближе среднего, что создаёт сосредоточенный взгляд."

    p1 = (
        f"Метрика измеряет расстояние между внутренними уголками глаз относительно ширины лица. "
        f"Значение {val:.4f} {direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ), что означает — {descr}."
    )

    if z > 0.3:
        p2 = "Визуально это создаёт ощущение открытого, спокойного взгляда. Широко посаженные глаза часто ассоциируются с доброжелательностью и доступностью."
    elif z < -0.3:
        p2 = "Близко посаженные глаза создают эффект сосредоточенного, вдумчивого взгляда. Это часто ассоциируется с интеллектуальностью и решительностью."
    else:
        p2 = "Сбалансированное расстояние между глазами — признак классической гармонии лица."

    p3 = (
        "Исследования Zebrowitz — Reading Faces показали, что расстояние между глазами "
        "влияет на восприятие надёжности и открытости."
    )

    infl = "Расстояние между глазами формирует общее впечатление от взгляда и влияет на считываемые эмоции."

    return title, p1, p2, p3, infl


def text_eye_tilt(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = "выраженный позитивный наклон — наружные уголки глаз приподняты"
    elif z < -0.3:
        descr = "лёгкое опущение наружных уголков глаз"
    else:
        descr = "нейтральный, слегка приподнятый наклон уголков глаз"

    if m["score"] >= 8.5:
        title = "Наклон уголков глаз близок к идеальному."
    elif z > 0:
        title = "Наружные уголки глаз заметно приподняты, формируя уверенный взгляд."
    else:
        title = "Наружные уголки глаз слегка опущены, придавая взгляду мягкость."

    p1 = (
        f"Метрика оценивает наклон линии от внутреннего к наружному уголку глаза. "
        f"Значение {val:.4f} {direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ) — {descr}."
    )

    if z > 0.3:
        p2 = "На лице это проявляется как сфокусированный, уверенный взгляд, который часто называют «fox eyes». Приподнятые уголки глаз — один из наиболее привлекательных маркеров для обоих полов."
    elif z < -0.3:
        p2 = "Опущенные уголки глаз создают мягкий, располагающий взгляд, который часто ассоциируется с эмпатией и добротой."
    else:
        p2 = "Нейтральный наклон уголков глаз — универсально гармоничная характеристика."

    p3 = (
        "Согласно исследованиям Little & Jones — Evidence against perceptual bias views for symmetry "
        "preferences in human faces, позитивный кантальный наклон устойчиво оценивается как привлекательный."
    )

    infl = "Наклон глаз — один из ключевых факторов восприятия взгляда: позитивный наклон ассоциируется с уверенностью, негативный — с мягкостью."

    return title, p1, p2, p3, infl


def text_nose_width(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = "нос немного шире среднего относительно лица"
    elif z < -0.3:
        descr = "нос немного уже среднего относительно лица"
    else:
        descr = "нос точно пропорционален ширине лица"

    if m["score"] >= 8.5:
        title = "Ширина носа идеально вписывается в пропорции лица."
    else:
        title = f"Ширина носа отличается от среднего — {descr}."

    p1 = (
        f"Метрика сравнивает ширину крыльев носа с шириной лица. Значение {val:.4f} "
        f"{deviation_word(z_abs)} {norm:.4f} (отклонение ~{z_abs:.2f}σ), что означает — {descr}."
    )

    if abs(z) < 0.5:
        p2 = "Визуально нос не привлекает к себе избыточного внимания и гармонично вписывается в общую картину. Пропорциональный нос позволяет другим чертам играть ведущую роль в формировании впечатления."
    elif z > 0:
        p2 = "Более широкий нос создаёт акцент на центральной зоне лица. Это характерная черта, которая может усиливать выразительность профиля."
    else:
        p2 = "Узкий нос визуально утончает центральную зону лица, делая её более деликатной."

    p3 = (
        "Согласно антропометрическим стандартам Farkas et al. — International anthropometric study "
        "of facial morphology, ширина носа около 23% ширины лица считается оптимальной."
    )

    infl = "Ширина носа влияет на баланс центральной части лица — пропорциональный нос работает незаметно, но важно."

    return title, p1, p2, p3, infl


def text_mouth_width(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = "рот заметно шире среднестатистического"
    elif z < -0.3:
        descr = "рот более компактный, чем среднестатистический"
    else:
        descr = "ширина рта точно пропорциональна лицу"

    if m["score"] >= 8.5:
        title = "Ширина рта пропорциональна лицу."
    elif z > 0:
        title = "Рот заметно шире статистической нормы относительно ширины скул."
    else:
        title = "Рот компактнее статистической нормы относительно ширины скул."

    p1 = (
        f"Метрика измеряет ширину рта относительно ширины скул. Значение {val:.4f} "
        f"{direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ), что означает — {descr}."
    )

    if z > 0.3:
        p2 = "На лице это проявляется как широкая линия рта, которая визуально расширяет нижнюю часть лица. Это индивидуальная особенность строения — многие харизматичные и узнаваемые лица отличаются именно широким ртом."
    elif z < -0.3:
        p2 = "Компактный рот визуально утончает нижнюю часть лица и часто ассоциируется с миловидностью и аккуратностью черт."
    else:
        p2 = "Сбалансированная ширина рта формирует универсально гармоничный нижний контур лица."

    p3 = (
        "Исследования Cunningham — Measuring the physical in physical attractiveness показали, "
        "что ширина рта влияет на восприятие выразительности и харизмы."
    )

    infl = "Ширина рта формирует акцент в нижней части лица и влияет на выразительность улыбки."

    return title, p1, p2, p3, infl


def text_nose_length(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = "слегка удлинённый нос"
    elif z < -0.3:
        descr = "слегка укороченный нос"
    else:
        descr = "идеально пропорциональная длина носа"

    if m["score"] >= 8.5:
        title = "Длина носа гармонично вписывается в пропорции лица."
    elif z > 0:
        title = "Длина носа чуть выше среднего."
    else:
        title = "Длина носа чуть короче среднего, что придаёт лицу молодой вид."

    p1 = (
        f"Метрика оценивает длину носа (от переносицы до основания) относительно высоты лица. "
        f"Значение {val:.4f} {direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ), что указывает на {descr}."
    )

    if z > 0.3:
        p2 = "Удлинённый нос визуально вытягивает центральную часть лица и часто ассоциируется с аристократичностью."
    elif z < -0.3:
        p2 = "Укороченный нос делает среднюю часть лица более компактной, а нижнюю — более выраженной. Это омолаживает образ."
    else:
        p2 = "Пропорциональная длина носа — признак классической гармонии лица."

    p3 = (
        "Согласно Grammer & Thornhill — Human facial attractiveness and sexual selection, "
        "длина носа является одним из факторов восприятия привлекательности."
    )

    infl = "Длина носа влияет на восприятие средней трети лица и общую визуальную сбалансированность."

    return title, p1, p2, p3, infl
def text_chin_length(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = "удлинённый подбородок"
    elif z < -0.3:
        descr = "более компактный подбородок"
    else:
        descr = "идеально пропорциональный подбородок"

    if m["score"] >= 8.5:
        title = _gender_form(
            gender,
            "Подбородок выражен и пропорционально удлинён — классический маркер мужественности.",
            "Подбородок гармонично пропорционален лицу."
        )
    elif z > 0:
        title = _gender_form(
            gender,
            "Подбородок выражен и пропорционально удлинён.",
            "Подбородок чуть длиннее среднего, что добавляет лицу выразительности."
        )
    else:
        title = "Подбородок компактный, что смягчает нижнюю часть лица."

    p1 = (
        f"Метрика измеряет расстояние от нижней губы до подбородка относительно высоты лица. "
        f"Значение {val:.4f} {direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ), что указывает на {descr}."
    )

    if z > 0.3:
        p2 = _gender_form(
            gender,
            "Визуально это проявляется как чётко выраженная подбородочная зона, придающая лицу завершённость. Выраженный подбородок — один из классических маркеров мужественности, ассоциирующийся с решительностью и внутренней силой.",
            "Удлинённый подбородок придаёт лицу выразительность и характер."
        )
    elif z < -0.3:
        p2 = _gender_form(
            gender,
            "Компактный подбородок смягчает нижнюю треть лица и может визуально омолаживать образ.",
            "Компактный подбородок ассоциируется с миловидностью и юностью — это одна из черт, придающих лицу мягкость."
        )
    else:
        p2 = "Пропорциональный подбородок завершает нижнюю треть лица гармонично."

    p3 = (
        "Исследования Thornhill & Gangestad — Facial attractiveness подтверждают, "
        "что длина подбородка связана с восприятием доминантности и зрелости."
    )

    infl = "Подбородок завершает силуэт лица и формирует впечатление силы и определённости."

    return title, p1, p2, p3, infl


def text_chin_contour(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = "более тупой угол — широкая, квадратная челюсть"
    elif z < -0.3:
        descr = "более острый угол — узкий, треугольный подбородок"
    else:
        descr = "сбалансированный контур подбородка"

    if m["score"] >= 8.5:
        title = _gender_form(
            gender,
            "Угол челюсти широкий и квадратный — выраженный признак волевого характера.",
            "Контур подбородка гармонично сбалансирован."
        )
    elif z > 0:
        title = "Угол челюсти шире среднего, формируя структурный контур."
    else:
        title = "Угол челюсти острее среднего, формируя V-образный силуэт."

    p1 = (
        f"Метрика оценивает угол сужения подбородка через соотношение нижней и верхней частей челюсти. "
        f"Значение {val:.4f} {direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ), что указывает на {descr}."
    )

    if z > 0.3:
        p2 = _gender_form(
            gender,
            "На лице это проявляется как чёткий, угловатый контур нижней части, который сразу привлекает внимание. Квадратная челюсть — один из ключевых маркеров мужественности, ассоциирующийся с волевым характером и физической силой.",
            "Структурный контур челюсти добавляет лицу характера и архитектурности."
        )
    elif z < -0.3:
        p2 = _gender_form(
            gender,
            "Острый контур подбородка создаёт более деликатный нижний силуэт.",
            "V-образный подбородок ассоциируется с женственностью и утончённостью — это одна из ценимых черт миловидности."
        )
    else:
        p2 = "Сбалансированный угол челюсти создаёт универсально гармоничный контур."

    p3 = (
        "Согласно Cunningham et al. — Their ideas of beauty are, on the whole, the same as ours, "
        "контур челюсти устойчиво оценивается как один из ключевых факторов восприятия лица."
    )

    infl = _gender_form(
        gender,
        "Квадратная челюсть — один из главных визуальных якорей мужского лица, формирующий впечатление надёжности.",
        "Контур подбородка определяет силуэт нижней части лица и формирует общее впечатление."
    )

    return title, p1, p2, p3, infl


def text_nose_to_mouth(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    nose_w_ratio = extra["nose_w_ratio"]
    mouth_w_ratio = extra["mouth_w_ratio"]

    if z > 0.3:
        descr = "нос относительно широк к ширине рта"
    elif z < -0.3:
        descr = "нос относительно узок к ширине рта"
    else:
        descr = "нос и рот пропорциональны друг другу"

    if m["score"] >= 8.5:
        title = "Соотношение носа и рта гармонично сбалансировано."
    elif z < 0:
        title = "Нос узкий относительно ширины рта, что создаёт заметный контраст."
    else:
        title = "Нос шире рта относительно нормы."

    cause = ""
    if z < -0.3:
        if mouth_w_ratio > 0.41:
            cause = f" Основная причина — увеличенная ширина рта (B = {mouth_w_ratio:.4f}), а не сужение носа."
        else:
            cause = f" Основная причина — узкий нос (A = {nose_w_ratio:.4f})."
    elif z > 0.3:
        if nose_w_ratio > 0.24:
            cause = f" Основная причина — увеличенная ширина носа (A = {nose_w_ratio:.4f})."
        else:
            cause = f" Основная причина — компактный рот (B = {mouth_w_ratio:.4f})."

    p1 = (
        f"Метрика сравнивает ширину носа с шириной рта. Значение {val:.4f} "
        f"{direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ), что означает — {descr}.{cause}"
    )

    if abs(z) < 0.5:
        p2 = "Сбалансированное соотношение носа и рта создаёт гармонию центральной и нижней зоны лица."
    elif z < 0:
        p2 = "Контраст между компактным носом и широким ртом создаёт динамику в нижней части лица. Это индивидуальная особенность строения, формирующая узнаваемый характер."
    else:
        p2 = "Нос, преобладающий над ртом, визуально центрирует лицо и создаёт классические пропорции."

    p3 = (
        "Исследования Perrett et al. — Effects of sexual dimorphism on facial attractiveness "
        "показали, что соотношение носа и рта влияет на восприятие маскулинности."
    )

    infl = "Соотношение носа и рта определяет баланс центральной и нижней части лица — отклонения формируют характерные индивидуальные черты."

    return title, p1, p2, p3, infl


def text_biocular(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = "слегка расширенная биокулярная зона"
    elif z < -0.3:
        descr = "слегка зауженная биокулярная зона"
    else:
        descr = "идеально пропорциональная биокулярная зона"

    if m["score"] >= 8.5:
        title = "Расстояние между внешними уголками глаз пропорционально ширине лица."
    else:
        title = f"Биокулярная ширина отличается от среднего — {descr}."

    p1 = (
        f"Метрика измеряет расстояние между внешними уголками глаз относительно ширины лица. "
        f"Значение {val:.4f} {direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ), что указывает на {descr}."
    )

    if abs(z) < 0.5:
        p2 = "Визуально это создаёт ощущение пропорционально расположенных глаз, что гармонирует с общими пропорциями лица."
    elif z > 0:
        p2 = "Расширенная биокулярная зона создаёт открытый, выразительный взгляд."
    else:
        p2 = "Более узкая биокулярная зона создаёт сосредоточенный, концентрированный взгляд."

    p3 = (
        "Согласно Farkas — Anthropometry of the Head and Face, биокулярная ширина около 71% "
        "ширины лица считается оптимальной."
    )

    infl = "Биокулярная ширина обеспечивает визуальный баланс верхней части лица."

    return title, p1, p2, p3, infl


def text_forehead(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = "лоб заметно шире среднестатистического относительно лица"
    elif z < -0.3:
        descr = "лоб заметно уже среднестатистического относительно лица"
    else:
        descr = "пропорциональная ширина лба"

    if m["score"] >= 8.5:
        title = "Ширина лба пропорциональна общим чертам лица."
    elif z > 0:
        title = "Лоб чуть шире среднего, что придаёт лицу ощущение интеллектуальности."
    else:
        title = "Лоб чуть уже среднего, что создаёт более компактный верхний контур."

    p1 = (
        f"Метрика сравнивает ширину лба (между краями бровей) с шириной лица. "
        f"Значение {val:.4f} {direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ), что означает — {descr}."
    )

    if z > 0.3:
        p2 = "Визуально это проявляется как широкая верхняя часть лица, которая может создавать впечатление интеллектуальности и открытости. Широкий лоб — нейтральная или позитивная черта, особенно в сочетании с выраженной челюстью."
    elif z < -0.3:
        p2 = "Узкий лоб визуально утончает верхнюю часть лица, создавая более деликатные пропорции."
    else:
        p2 = "Пропорциональный лоб обеспечивает гармоничный баланс верхней трети лица."

    p3 = (
        "Исследования Zebrowitz & Montepare — Social psychological face perception показали, "
        "что ширина лба влияет на восприятие интеллекта и компетентности."
    )

    infl = "Ширина лба формирует впечатление от верхней части лица и влияет на считываемые когнитивные характеристики."

    return title, p1, p2, p3, infl


def text_lips_fullness(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = "губы полнее среднего"
    elif z < -0.3:
        descr = "губы тоньше среднего"
    else:
        descr = "идеально пропорциональная полнота губ"

    if m["score"] >= 8.5:
        title = "Полнота губ гармонична для общих черт лица."
    elif z > 0:
        title = _gender_form(
            gender,
            "Губы полнее среднего, что добавляет выразительности нижней части лица.",
            "Губы умеренно полные — одна из ценимых черт женской привлекательности."
        )
    else:
        title = _gender_form(
            gender,
            "Губы умеренно тонкие, что типично для мужского лица.",
            "Губы тоньше среднего, что создаёт более деликатный контур."
        )

    p1 = (
        f"Метрика измеряет высоту губ относительно ширины рта. Значение {val:.4f} "
        f"{direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ), что указывает на {descr}."
    )

    if z > 0.3:
        p2 = _gender_form(
            gender,
            "Полные губы создают акцент в нижней части лица и придают образу чувственность.",
            "Полные губы — один из ключевых маркеров женской привлекательности, ассоциирующийся с молодостью и выразительностью."
        )
    elif z < -0.3:
        p2 = _gender_form(
            gender,
            "Визуально губы выглядят аккуратными и не доминируют в общей картине лица. Для мужчин умеренно тонкие губы — нейтральная характеристика, не снижающая привлекательности.",
            "Тонкие губы создают более деликатный контур и часто ассоциируются с утончённостью."
        )
    else:
        p2 = "Сбалансированная полнота губ — признак гармоничных пропорций нижней части лица."

    p3 = (
        "Исследования Cunningham — Measuring the physical in physical attractiveness показали, "
        "что полнота губ значимо для женщин и менее значима для мужчин."
    )

    infl = _gender_form(
        gender,
        "Полнота губ слабо влияет на восприятие мужского лица — приоритет за чёткостью контура.",
        "Полнота губ — одна из ключевых характеристик женской привлекательности."
    )

    return title, p1, p2, p3, infl


def text_lips_proportions(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = "верхняя губа доминирует или равна нижней"
    elif z < -0.3:
        descr = "нижняя губа существенно полнее верхней"
    else:
        descr = "классическое соотношение губ 1:1.6"

    if m["score"] >= 8.5:
        title = "Пропорции губ близки к классическому соотношению."
    elif z > 0:
        title = "Верхняя губа непропорционально доминирует над нижней."
    else:
        title = "Нижняя губа существенно полнее верхней."

    p1 = (
        f"Метрика сравнивает высоту верхней губы с высотой нижней. Значение {val:.4f} "
        f"{direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ), что указывает на {descr}."
    )

    if abs(z) < 0.5:
        p2 = "Сбалансированное соотношение губ создаёт гармоничный контур рта, близкий к классическому золотому сечению."
    elif z > 0:
        p2 = "Визуально это проявляется как нетипичное соотношение губ, где верхняя губа выглядит относительно выраженной. Это индивидуальная особенность строения — характерные пропорции часто формируют узнаваемость лица."
    else:
        p2 = "Полная нижняя губа создаёт акцент в нижней части лица и часто воспринимается как чувственная черта."

    p3 = (
        "Согласно Perrett — In Your Face: The New Science of Human Attractiveness, "
        "классическое соотношение губ 1:1.6 (верхняя к нижней) считается идеальным."
    )

    infl = "Соотношение губ формирует индивидуальный контур рта — отклонения от нормы создают узнаваемые черты."

    return title, p1, p2, p3, infl


def text_jaw_to_mouth(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    jaw_w_ratio = extra["jaw_w_ratio"]
    mouth_w_ratio = extra["mouth_w_ratio"]

    if z > 0.3:
        descr = "челюсть существенно шире рта"
    elif z < -0.3:
        descr = "челюсть недостаточно широка относительно рта"
    else:
        descr = "челюсть и рот пропорциональны друг другу"

    if m["score"] >= 8.5:
        title = "Соотношение челюсти и рта гармонично сбалансировано."
    elif z < 0:
        title = "Челюсть узкая относительно ширины рта."
    else:
        title = "Челюсть существенно шире рта, формируя выраженный контур."

    cause = ""
    if z < -0.3:
        if mouth_w_ratio > 0.41:
            cause = f" Основная причина — увеличенная ширина рта (B = {mouth_w_ratio:.4f}), а также несколько зауженная челюсть (A = {jaw_w_ratio:.4f})."
        else:
            cause = f" Основная причина — узкая челюсть (A = {jaw_w_ratio:.4f})."

    p1 = (
        f"Метрика сравнивает ширину челюсти с шириной рта. Значение {val:.4f} "
        f"{direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ), что означает — {descr}.{cause}"
    )

    if abs(z) < 0.5:
        p2 = "Сбалансированное соотношение челюсти и рта создаёт гармоничный контур нижней части лица."
    elif z < 0:
        p2 = "Визуально это создаёт впечатление, что рот занимает большую часть нижней зоны лица. Это индивидуальная особенность строения — многие привлекательные лица имеют выраженные отклонения в 1-2 метриках, и именно это формирует характер и узнаваемость."
    else:
        p2 = "Широкая челюсть относительно рта создаёт мощный нижний контур и подчёркивает архитектурность лица."

    p3 = (
        "Исследования Rhodes et al. — Facial symmetry and the perception of beauty подтверждают, "
        "что общее впечатление от лица определяется совокупностью черт, а не отдельными пропорциями."
    )

    infl = "Соотношение челюсти и рта формирует силуэт нижней части лица — отклонения часто компенсируются другими сильными чертами."

    return title, p1, p2, p3, infl


def text_eye_shape(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = "более открытую, округлую форму глаз"
    elif z < -0.3:
        descr = "более вытянутую, миндалевидную форму глаз"
    else:
        descr = "классическую миндалевидную форму глаз"

    if m["score"] >= 8.5:
        title = "Форма глаз близка к идеальной."
    elif z > 0:
        title = "Глаза открытые и выразительные — форма близка к округлой."
    else:
        title = "Глаза вытянуты — форма миндалевидная."

    p1 = (
        f"Метрика измеряет соотношение высоты глаза к его ширине. Значение {val:.4f} "
        f"{direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ), что указывает на {descr}."
    )

    if z > 0.3:
        p2 = "Визуально это проявляется как выразительный, широко раскрытый взгляд, который сразу привлекает внимание. Открытые глаза делают взгляд более запоминающимся и эмоционально насыщенным."
    elif z < -0.3:
        p2 = "Миндалевидная форма глаз — классический эстетический эталон, ассоциирующийся с экзотичностью и выразительностью."
    else:
        p2 = "Сбалансированная форма глаз — универсально гармоничная характеристика."

    p3 = (
        "Согласно Grammer et al. — Darwinian aesthetics: sexual selection and the biology of beauty, "
        "открытая форма глаз ассоциируется с молодостью и здоровьем."
    )

    infl = "Форма глаз — один из главных факторов выразительности взгляда и эмоциональности лица."

    return title, p1, p2, p3, infl


def text_brow_height(m, extra, gender):
    z = m["z"]
    z_abs = abs(z)
    val, norm = m["value"], m["norm"]

    if z > 0.3:
        descr = "брови расположены заметно выше среднего"
    elif z < -0.3:
        descr = "брови расположены заметно ниже среднего"
    else:
        descr = "пропорциональное расположение бровей"

    if m["score"] >= 8.5:
        title = "Высота бровей пропорциональна глазной зоне."
    elif z > 0:
        title = "Брови посажены высоко, что увеличивает расстояние до век."
    else:
        title = "Брови посажены низко, что создаёт сосредоточенный взгляд."

    p1 = (
        f"Метрика измеряет расстояние от нижнего края брови до века относительно ширины глаза. "
        f"Значение {val:.4f} {direction_word(z, 'выше нормы', 'ниже нормы', 'совпадает с нормой')} "
        f"{norm:.4f} (отклонение ~{z_abs:.2f}σ), что означает — {descr}."
    )

    if z > 0.3:
        p2 = _gender_form(
            gender,
            "Визуально это создаёт увеличенное пространство между бровью и глазом, которое может слегка «разрежать» верхнюю часть лица. Высоко посаженные брови — индивидуальная особенность, которая у мужчин иногда воспринимается как признак удивлённости.",
            "Высоко посаженные брови — классическая черта женской миловидности, открывающая взгляд."
        )
    elif z < -0.3:
        p2 = _gender_form(
            gender,
            "Низко посаженные брови ассоциируются с доминантностью и сосредоточенностью — это усиливает мужественное впечатление.",
            "Низко посаженные брови создают глубокий, выразительный взгляд с акцентом на глаза."
        )
    else:
        p2 = "Пропорциональная высота бровей формирует гармоничную глазную зону."

    p3 = (
        "Исследования Zebrowitz — Reading Faces показали, что расстояние бровь-веко "
        "влияет на восприятие эмоционального состояния и характера."
    )

    infl = "Высота бровей формирует выражение взгляда и одну из ключевых эмоциональных характеристик лица."

    return title, p1, p2, p3, infl


# ================== TEXT ROUTER ==================
TEXT_GENERATORS = {
    "Симметрия лица": text_symmetry,
    "Пропорции лица": text_proportions,
    "Вертикальный баланс": text_vertical_balance,
    "Баланс скул и челюсти": text_cheek_jaw,
    "Размер глаз": text_eye_size,
    "Расстояние между глазами": text_eye_distance,
    "Наклон глаз": text_eye_tilt,
    "Ширина носа": text_nose_width,
    "Ширина рта": text_mouth_width,
    "Длина носа": text_nose_length,
    "Длина подбородка": text_chin_length,
    "Контур подбородка": text_chin_contour,
    "Нос к ширине рта": text_nose_to_mouth,
    "Биокулярная ширина": text_biocular,
    "Ширина лба": text_forehead,
    "Полнота губ": text_lips_fullness,
    "Пропорции губ": text_lips_proportions,
    "Челюсть к ширине рта": text_jaw_to_mouth,
    "Форма глаз": text_eye_shape,
    "Высота бровей": text_brow_height,
}


def generate_metric_text(metric, extra, gender):
    gen = TEXT_GENERATORS.get(metric["name"])
    if gen:
        return gen(metric, extra, gender)
    return ("Метрика лица.", "", "", "", "Метрика влияет на общий баланс лица.")


# ================== RECOMMENDATIONS ==================
RECOMMENDATIONS_BY_METRIC = {
    "Симметрия лица": {
        "male": "Лёгкие визуальные приёмы (симметричная стрижка, аккуратные брови) помогут визуально сбалансировать черты.",
        "female": "Симметричный макияж и аккуратная укладка волос визуально выровняют черты лица.",
    },
    "Пропорции лица": {
        "male": "Подбор стрижки с объёмом по бокам визуально расширит лицо, если оно вытянутое, или вытянет, если округлое.",
        "female": "Стрижка и укладка с объёмом помогут визуально подкорректировать форму лица в нужном направлении.",
    },
    "Вертикальный баланс": {
        "male": "Лёгкая щетина может визуально удлинить или сбалансировать нижнюю треть лица.",
        "female": "Контурирование скул поможет визуально подкорректировать вертикальные пропорции лица.",
    },
    "Баланс скул и челюсти": {
        "male": "Лёгкая щетина по линии челюсти визуально расширит её и сбалансирует с скулами.",
        "female": "Контурирование скул и челюсти подчеркнёт овал и улучшит соотношение зон.",
    },
    "Размер глаз": {
        "male": "Уход за кожей вокруг глаз и аккуратные брови визуально подчеркнут размер и форму глаз.",
        "female": "Макияж глаз с акцентом на ресницы и стрелки визуально увеличит и подчеркнёт глаза.",
    },
    "Расстояние между глазами": {
        "male": "Грамотная коррекция формы бровей поможет визуально скорректировать расстояние между глазами.",
        "female": "Светлые тени во внутреннем уголке глаза визуально приближают глаза, тёмные — отдаляют.",
    },
    "Наклон глаз": {
        "male": "Лёгкая корректировка хвостика брови может визуально приподнять уголки глаз.",
        "female": "Стрелки с подъёмом во внешнем уголке создадут эффект «fox eyes» и приподнимут взгляд.",
    },
    "Ширина носа": {
        "male": "Контурирование боковых частей носа визуально утончит его при необходимости.",
        "female": "Лёгкое контурирование переносицы и крыльев носа визуально скорректирует ширину.",
    },
    "Ширина рта": {
        "male": "Уход за губами и щетина по контуру визуально сбалансируют ширину рта.",
        "female": "Чёткий контур губ карандашом визуально скорректирует ширину рта.",
    },
    "Длина носа": {
        "male": "Контурирование кончика носа поможет визуально подкорректировать длину.",
        "female": "Контурирование основания носа визуально укоротит его при необходимости.",
    },
    "Длина подбородка": {
        "male": "Лёгкая щетина в области подбородка визуально удлинит или сбалансирует нижнюю треть.",
        "female": "Контурирование подбородка визуально удлинит или укоротит нижнюю часть лица.",
    },
    "Контур подбородка": {
        "male": "Щетина по линии челюсти усилит контур и подчеркнёт мужественную форму нижней части лица.",
        "female": "Контурирование угла челюсти подчеркнёт овал и сделает контур более выраженным.",
    },
    "Нос к ширине рта": {
        "male": "Щетина или борода визуально смягчит контраст между носом и ртом.",
        "female": "Чёткий контур губ и лёгкое контурирование носа сбалансируют центральную и нижнюю зоны.",
    },
    "Биокулярная ширина": {
        "male": "Аккуратная коррекция бровей визуально сбалансирует биокулярную зону.",
        "female": "Грамотный макияж бровей и глаз визуально гармонизирует верхнюю часть лица.",
    },
    "Ширина лба": {
        "male": "Стрижка с правильно подобранной чёлкой визуально скорректирует ширину лба.",
        "female": "Чёлка или укладка с обрамлением лба визуально подкорректируют его пропорции.",
    },
    "Полнота губ": {
        "male": "Увлажняющий бальзам поддержит чёткий и здоровый контур губ.",
        "female": "Прозрачный блеск или увлажняющая помада визуально добавят губам объёма.",
    },
    "Пропорции губ": {
        "male": "Уход за губами поможет визуально выровнять пропорции верхней и нижней.",
        "female": "Контурирование губ карандашом скорректирует соотношение верхней и нижней.",
    },
    "Челюсть к ширине рта": {
        "male": "Лёгкая щетина (3-5 мм) визуально расширит челюсть и улучшит её соотношение с ртом.",
        "female": "Контурирование угла челюсти визуально расширит её и сбалансирует с шириной рта.",
    },
    "Форма глаз": {
        "male": "Уход за кожей вокруг глаз и аккуратные брови подчеркнут форму глаз.",
        "female": "Макияж с подводкой и тенями подчеркнёт миндалевидную или открытую форму глаз.",
    },
    "Высота бровей": {
        "male": "Коррекция формы бровей карандашом или гелем поможет визуально скорректировать расстояние до век.",
        "female": "Профессиональная коррекция бровей сбалансирует расстояние от брови до века.",
    },
}


STRENGTH_DESCRIPTIONS = {
    "Симметрия лица": "Высокая симметрия — лицо практически зеркально по обеим сторонам, что воспринимается как признак генетического здоровья.",
    "Пропорции лица": "Гармоничные пропорции лица создают универсально привлекательный овал.",
    "Вертикальный баланс": "Сбалансированные трети лица создают классические гармоничные пропорции.",
    "Баланс скул и челюсти": "Соотношение скул и челюсти близко к идеальному, формируя архитектурный контур.",
    "Размер глаз": "Пропорциональный размер глаз делает взгляд естественно выразительным.",
    "Расстояние между глазами": "Оптимальное расстояние между глазами создаёт гармоничную глазную зону.",
    "Наклон глаз": "Приподнятые уголки глаз создают сфокусированный, уверенный взгляд — одна из самых ценимых эстетических черт.",
    "Ширина носа": "Пропорциональный нос гармонично вписывается в общую картину лица.",
    "Ширина рта": "Сбалансированная ширина рта формирует гармоничный нижний контур.",
    "Длина носа": "Пропорциональная длина носа поддерживает классические пропорции лица.",
    "Длина подбородка": "Выраженный подбородок завершает силуэт лица и формирует впечатление силы.",
    "Контур подбородка": "Чёткий контур челюсти создаёт архитектурный нижний силуэт лица.",
    "Нос к ширине рта": "Сбалансированное соотношение носа и рта создаёт гармонию центральной зоны.",
    "Биокулярная ширина": "Пропорциональная биокулярная ширина обеспечивает баланс верхней части лица.",
    "Ширина лба": "Пропорциональный лоб формирует гармоничную верхнюю треть лица.",
    "Полнота губ": "Гармоничная полнота губ создаёт привлекательный контур рта.",
    "Пропорции губ": "Классическое соотношение губ создаёт гармоничный контур рта.",
    "Челюсть к ширине рта": "Сбалансированное соотношение челюсти и рта формирует мощный нижний контур.",
    "Форма глаз": "Выразительная форма глаз делает взгляд запоминающимся и эмоционально насыщенным.",
    "Высота бровей": "Пропорциональная высота бровей формирует гармоничную глазную зону.",
}


def generate_recommendations(data, gender):
    strengths = []
    for m in data["strengths"]:
        text = STRENGTH_DESCRIPTIONS.get(m["name"], "Это одна из сильных сторон вашего лица.")
        strengths.append((m["name"], text))

    improvements = []
    for m in data["weak"]:
        rec = RECOMMENDATIONS_BY_METRIC.get(m["name"], {})
        text = rec.get("female" if gender == "female" else "male", "Лёгкая визуальная коррекция поможет улучшить эту зону.")
        improvements.append((m["name"], text))

    extra_tips = _gender_form(
        gender,
        [
            ("Уход за кожей лица", "Регулярное очищение и увлажнение кожи улучшит общий вид и придаст чертам свежесть."),
            ("Сон и режим", "Полноценный сон 7-8 часов уменьшит отёчность и придаст лицу подтянутость."),
        ],
        [
            ("Уход за кожей лица", "Регулярное очищение, увлажнение и SPF-защита поддержат свежий и здоровый вид кожи."),
            ("Уход за бровями и ресницами", "Профессиональная коррекция бровей и уход за ресницами усилят выразительность взгляда."),
        ]
    )

    improvements.extend(extra_tips)

    return strengths, improvements
# ================== OVERLAY GENERATION ==================
def get_landmarks_for_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, None
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None, None
    return results.multi_face_landmarks[0].landmark, (w, h)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def base_image_for_overlay(image_bytes, max_size=900):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((max_size, max_size))
    return image


def lm_pt(lm, name, w, h):
    idx = IDX[name]
    return int(lm[idx].x * w), int(lm[idx].y * h)


def draw_point(draw, p, color, r=5):
    x, y = p
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def draw_line(draw, p1, p2, color, width=3):
    draw.line([p1, p2], fill=color, width=width)


def overlay_for_metric(image_bytes, metric_name, color_hex):
    img = base_image_for_overlay(image_bytes)
    w, h = img.size
    draw = ImageDraw.Draw(img)
    color = hex_to_rgb(color_hex)
    accent = hex_to_rgb(COLOR_ACCENT)

    np_arr = np.array(img)
    img_rgb = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return img

    lm = results.multi_face_landmarks[0].landmark

    def P(name):
        return lm_pt(lm, name, w, h)

    if metric_name == "Симметрия лица":
        midline_top = P("forehead")
        midline_bot = P("chin")
        draw_line(draw, midline_top, midline_bot, color, 3)
        for n in ["left_eye_inner", "right_eye_inner", "left_eye_outer", "right_eye_outer",
                  "mouth_left", "mouth_right", "nose_left", "nose_right"]:
            draw_point(draw, P(n), accent, 5)

    elif metric_name == "Пропорции лица":
        draw_line(draw, P("nose_bridge"), P("chin"), color, 3)
        draw_line(draw, P("face_left"), P("face_right"), color, 3)

    elif metric_name == "Вертикальный баланс":
        draw_line(draw, P("nose_bridge"), P("nose_base"), color, 3)
        draw_line(draw, P("nose_base"), P("chin"), color, 3)
        draw_point(draw, P("nose_bridge"), accent, 5)
        draw_point(draw, P("nose_base"), accent, 5)
        draw_point(draw, P("chin"), accent, 5)

    elif metric_name == "Баланс скул и челюсти":
        draw_line(draw, P("cheek_left"), P("cheek_right"), color, 3)
        draw_line(draw, P("jaw_left"), P("jaw_right"), color, 3)

    elif metric_name == "Размер глаз":
        draw_line(draw, P("left_eye_outer"), P("left_eye_inner"), color, 3)
        draw_line(draw, P("right_eye_inner"), P("right_eye_outer"), color, 3)

    elif metric_name == "Расстояние между глазами":
        draw_line(draw, P("left_eye_inner"), P("right_eye_inner"), color, 3)
        draw_point(draw, P("left_eye_inner"), accent, 5)
        draw_point(draw, P("right_eye_inner"), accent, 5)

    elif metric_name == "Наклон глаз":
        draw_line(draw, P("left_eye_inner"), P("left_eye_outer"), color, 3)
        draw_line(draw, P("right_eye_inner"), P("right_eye_outer"), color, 3)

    elif metric_name == "Ширина носа":
        draw_line(draw, P("nose_left"), P("nose_right"), color, 3)
        draw_point(draw, P("nose_left"), accent, 5)
        draw_point(draw, P("nose_right"), accent, 5)

    elif metric_name == "Ширина рта":
        draw_line(draw, P("mouth_left"), P("mouth_right"), color, 3)

    elif metric_name == "Длина носа":
        draw_line(draw, P("nose_bridge"), P("nose_base"), color, 3)

    elif metric_name == "Длина подбородка":
        draw_line(draw, P("lower_lip"), P("chin"), color, 3)
        draw_point(draw, P("lower_lip"), accent, 5)
        draw_point(draw, P("chin"), accent, 5)

    elif metric_name == "Контур подбородка":
        draw_line(draw, P("jaw_left"), P("chin"), color, 3)
        draw_line(draw, P("jaw_right"), P("chin"), color, 3)
        draw_line(draw, P("jaw_left_lower"), P("jaw_right_lower"), color, 3)

    elif metric_name == "Нос к ширине рта":
        draw_line(draw, P("nose_left"), P("nose_right"), color, 3)
        draw_line(draw, P("mouth_left"), P("mouth_right"), color, 3)

    elif metric_name == "Биокулярная ширина":
        draw_line(draw, P("left_eye_outer"), P("right_eye_outer"), color, 3)

    elif metric_name == "Ширина лба":
        draw_line(draw, P("forehead_left"), P("forehead_right"), color, 3)

    elif metric_name == "Полнота губ":
        draw_line(draw, P("upper_lip_top"), P("lower_lip_bottom"), color, 3)

    elif metric_name == "Пропорции губ":
        draw_line(draw, P("upper_lip_top"), P("upper_lip"), color, 3)
        draw_line(draw, P("lower_lip"), P("lower_lip_bottom"), color, 3)

    elif metric_name == "Челюсть к ширине рта":
        draw_line(draw, P("jaw_left"), P("jaw_right"), color, 3)
        draw_line(draw, P("mouth_left"), P("mouth_right"), color, 3)

    elif metric_name == "Форма глаз":
        draw_line(draw, P("left_eye_top"), P("left_eye_bottom"), color, 3)
        draw_line(draw, P("right_eye_top"), P("right_eye_bottom"), color, 3)
        draw_line(draw, P("left_eye_outer"), P("left_eye_inner"), color, 2)
        draw_line(draw, P("right_eye_inner"), P("right_eye_outer"), color, 2)

    elif metric_name == "Высота бровей":
        draw_line(draw, P("left_brow_mid"), P("left_eye_top"), color, 3)
        draw_line(draw, P("right_brow_mid"), P("right_eye_top"), color, 3)

    return img


def overlay_for_cover(image_bytes):
    img = base_image_for_overlay(image_bytes)
    return img


# ================== RADAR CHART ==================
def create_radar_chart(metrics, output_path):
    labels = [m["name"] for m in metrics]
    scores = [m["score"] for m in metrics]

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    ax.plot(angles_plot, scores_plot, color=COLOR_ACCENT, linewidth=2)
    ax.fill(angles_plot, scores_plot, color=COLOR_ACCENT, alpha=0.25)

    for i, (angle, label, color_hex) in enumerate(zip(angles, labels, METRIC_COLORS)):
        ax.scatter([angle], [scores[i]], color=color_hex, s=60, zorder=5, edgecolors=COLOR_BG, linewidths=1.5)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color=COLOR_TEXT_SOFT, size=8)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color=COLOR_TEXT_MUTED, size=7)
    ax.tick_params(axis="x", pad=12)
    ax.grid(color=COLOR_LINE, linewidth=0.6, alpha=0.7)
    ax.spines["polar"].set_color(COLOR_LINE)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=COLOR_BG, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ================== DISTRIBUTION CHART ==================
def create_distribution_chart(score, output_path):
    fig, ax = plt.subplots(figsize=(6, 2))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    x = np.linspace(0, 10, 200)
    mean = 6.5
    sigma = 1.4
    y = np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    ax.fill_between(x, y, color=COLOR_BAR_BG, alpha=0.8)
    ax.plot(x, y, color=COLOR_LINE, linewidth=1)
    ax.axvline(score, color=COLOR_ACCENT, linewidth=2.5)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.1)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, facecolor=COLOR_BG, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
# ================== PDF HELPERS ==================
def draw_page_bg(c, width, height):
    c.setFillColor(HexColor(COLOR_BG))
    c.rect(0, 0, width, height, fill=1, stroke=0)


def draw_footer(c, width, page_num, total=23):
    c.setFillColor(HexColor(COLOR_TEXT_MUTED))
    c.setFont(FONT_REGULAR, 8)
    c.drawCentredString(width / 2, 22, f"Telegram: {BOT_USERNAME}  ·  {page_num} / {total}")


def draw_wrapped(c, text, x, y, max_chars=82, line_height=14, font=FONT_REGULAR, size=10, color=COLOR_TEXT):
    c.setFillColor(HexColor(color))
    c.setFont(font, size)
    for line in wrap_text(text, max_chars):
        c.drawString(x, y, line)
        y -= line_height
    return y


def draw_progress_bar(c, x, y, width, score, color_hex, height=10):
    c.setFillColor(HexColor(COLOR_BAR_BG))
    c.roundRect(x, y, width, height, height / 2, fill=1, stroke=0)
    fill_w = width * (score / 10)
    if fill_w > 1:
        c.setFillColor(HexColor(color_hex))
        c.roundRect(x, y, fill_w, height, height / 2, fill=1, stroke=0)


def draw_dots(c, x, y, filled, color_hex, total=5, size=5, gap=12):
    for i in range(total):
        cx = x + i * gap
        if i < filled:
            c.setFillColor(HexColor(color_hex))
        else:
            c.setFillColor(HexColor(COLOR_BAR_BG))
        c.circle(cx, y, size, fill=1, stroke=0)


def draw_left_accent_line(c, x, y_top, y_bottom, color_hex, width=3):
    c.setFillColor(HexColor(color_hex))
    c.rect(x, y_bottom, width, y_top - y_bottom, fill=1, stroke=0)


def draw_h_line(c, x1, x2, y, color_hex=COLOR_LINE, width=0.5):
    c.setStrokeColor(HexColor(color_hex))
    c.setLineWidth(width)
    c.line(x1, y, x2, y)


# ================== PDF: COVER PAGE ==================
def draw_cover(c, width, height, image_bytes, data):
    draw_page_bg(c, width, height)

    c.setFillColor(HexColor(COLOR_TITLE))
    c.setFont(FONT_BOLD, 44)
    c.drawCentredString(width / 2, height - 100, BOT_NAME)

    c.setFillColor(HexColor(COLOR_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 11)
    c.drawCentredString(width / 2, height - 122, f"Telegram: {BOT_USERNAME}")

    c.setFillColor(HexColor(COLOR_TEXT))
    c.setFont(FONT_REGULAR, 12)
    c.drawCentredString(width / 2, height - 142, "Математический разбор пропорций лица.")

    draw_h_line(c, 60, width - 60, height - 165)

    cover_img = overlay_for_cover(image_bytes)
    cover_img.thumbnail((280, 320))
    img_reader = ImageReader(cover_img)

    img_w, img_h = cover_img.size
    img_x = (width - img_w) / 2
    img_y = height - 200 - img_h

    c.drawImage(img_reader, img_x, img_y, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')

    score_y = img_y - 30

    c.setFillColor(HexColor(COLOR_ACCENT))
    c.setFont(FONT_BOLD, 56)
    c.drawCentredString(width / 2, score_y - 50, f"{data['score']:.2f}")

    c.setFillColor(HexColor(COLOR_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 12)
    c.drawCentredString(width / 2, score_y - 70, "из 10")

    bar_w = 280
    bar_x = (width - bar_w) / 2
    bar_y = score_y - 100
    draw_progress_bar(c, bar_x, bar_y, bar_w, data["score"], COLOR_ACCENT, height=12)

    c.setFillColor(HexColor(COLOR_ACCENT))
    c.setFont(FONT_BOLD, 12)
    c.drawCentredString(width / 2, bar_y - 22, f"Вы входите в топ {data['top_percent']}% людей по гармонии лица!")

    dist_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    create_distribution_chart(data["score"], dist_path)
    dist_reader = ImageReader(dist_path)
    c.drawImage(dist_reader, width / 2 - 130, bar_y - 110, width=260, height=80, preserveAspectRatio=True, mask='auto')

    c.setFillColor(HexColor(COLOR_TITLE))
    c.setFont(FONT_BOLD, 13)
    c.drawCentredString(width / 2, bar_y - 130, f"Уровень: {data['level']}")

    strengths_text = "Сильные стороны: " + ", ".join([m["name"].lower() for m in data["strengths"]])
    c.setFillColor(HexColor(COLOR_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 10)
    c.drawCentredString(width / 2, bar_y - 148, strengths_text)

    impr_y = 175
    draw_left_accent_line(c, 60, impr_y + 18, impr_y - 78, COLOR_ACCENT)

    c.setFillColor(HexColor(COLOR_TITLE))
    c.setFont(FONT_BOLD, 11)
    c.drawString(80, impr_y, "ОБЩЕЕ ВПЕЧАТЛЕНИЕ")

    impression = (
        "Лицо проанализировано по ключевым геометрическим точкам с расчётом 20 антропометрических "
        "метрик. Каждая метрика сравнивается с медианными значениями и сигма-отклонением. "
        "Итоговая оценка отражает совокупную близость пропорций к статистическим нормам гармонии."
    )

    draw_wrapped(c, impression, 80, impr_y - 18, max_chars=86, line_height=14, size=10, color=COLOR_TEXT)

    draw_footer(c, width, 1)


# ================== PDF: PROFILE PAGE ==================
def draw_profile_page(c, width, height, data):
    draw_page_bg(c, width, height)

    c.setFillColor(HexColor(COLOR_TEXT))
    c.setFont(FONT_BOLD, 28)
    c.drawString(60, height - 80, "Профиль метрик")

    draw_h_line(c, 60, width - 60, height - 100)

    radar_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    create_radar_chart(data["metrics"], radar_path)
    c.drawImage(ImageReader(radar_path), width / 2 - 180, height - 470, width=360, height=360, preserveAspectRatio=True, mask='auto')

    draw_h_line(c, 60, width - 60, height - 490)

    c.setFillColor(HexColor(COLOR_ACCENT))
    c.setFont(FONT_BOLD, 12)
    c.drawString(60, height - 510, "Топ-3 сильных метрики")

    c.drawString(width / 2 + 10, height - 510, "Топ-3 зоны потенциала")

    y = height - 530
    for i, m in enumerate(data["strengths"]):
        color = METRIC_COLORS[data["metrics"].index(m)]
        c.setFillColor(HexColor(color))
        c.circle(70, y + 4, 3, fill=1, stroke=0)
        c.setFillColor(HexColor(COLOR_TEXT))
        c.setFont(FONT_REGULAR, 10)
        c.drawString(82, y, f"{m['name']}  —  {m['score']:.2f}")
        y -= 18

    y = height - 530
    for i, m in enumerate(data["weak"]):
        color = METRIC_COLORS[data["metrics"].index(m)]
        c.setFillColor(HexColor(color))
        c.circle(width / 2 + 20, y + 4, 3, fill=1, stroke=0)
        c.setFillColor(HexColor(COLOR_TEXT))
        c.setFont(FONT_REGULAR, 10)
        c.drawString(width / 2 + 32, y, f"{m['name']}  —  {m['score']:.2f}")
        y -= 18

    draw_h_line(c, 60, width - 60, height - 600)

    c.setFillColor(HexColor(COLOR_TEXT))
    c.setFont(FONT_BOLD, 13)
    c.drawString(60, height - 620, "Вклад каждой метрики")

    col1_x = 60
    col2_x = width / 2 + 10
    col_w = (width / 2) - 80

    y_start = height - 645
    y_l = y_start
    y_r = y_start

    for i, m in enumerate(data["metrics"]):
        color = METRIC_COLORS[i]
        if i < 10:
            c.setFillColor(HexColor(COLOR_TEXT_SOFT))
            c.setFont(FONT_REGULAR, 8.5)
            c.drawString(col1_x, y_l, m["name"])
            c.setFillColor(HexColor(COLOR_TEXT))
            c.drawRightString(col1_x + col_w, y_l, f"{m['score']:.2f}")
            draw_progress_bar(c, col1_x, y_l - 8, col_w, m["score"], color, height=6)
            y_l -= 22
        else:
            c.setFillColor(HexColor(COLOR_TEXT_SOFT))
            c.setFont(FONT_REGULAR, 8.5)
            c.drawString(col2_x, y_r, m["name"])
            c.setFillColor(HexColor(COLOR_TEXT))
            c.drawRightString(col2_x + col_w, y_r, f"{m['score']:.2f}")
            draw_progress_bar(c, col2_x, y_r - 8, col_w, m["score"], color, height=6)
            y_r -= 22

    bottom_y = min(y_l, y_r) - 10

    c.setFillColor(HexColor(COLOR_ACCENT))
    c.setFont(FONT_BOLD, 10)
    c.drawString(60, bottom_y, "КАК ЧИТАТЬ ОЦЕНКУ")

    legend = (
        "Каждая метрика — это соотношение двух расстояний на лице (безразмерная пропорция). "
        "Норма — медианное значение по выборке (для вашего пола), основанное на антропометрических "
        "данных (Farkas et al.). Оценка показывает близость к норме: 10 = совпадение, чем больше "
        "отклонение, тем ниже балл. σ (сигма) — мера естественного разброса: в ±1σ попадают ~68% "
        "людей, в ±2σ — ~95%. Низкая оценка не означает «некрасиво» — это лишь отражает "
        "статистическое отклонение."
    )

    draw_wrapped(c, legend, 60, bottom_y - 18, max_chars=110, line_height=12, size=8.5, color=COLOR_TEXT_SOFT)

    draw_footer(c, width, 2)


# ================== PDF: METRIC PAGE ==================
def draw_metric_page(c, width, height, image_bytes, m, idx, extra, gender, page_num):
    draw_page_bg(c, width, height)
    color = METRIC_COLORS[idx - 1]

    c.setFillColor(HexColor(COLOR_TEXT_MUTED))
    c.setFont(FONT_BOLD, 38)
    c.drawString(60, height - 95, f"{idx:02d}")

    c.setFillColor(HexColor(color))
    c.setFont(FONT_BOLD, 22)
    c.drawString(60, height - 135, m["name"])

    draw_h_line(c, 60, width - 60, height - 155)

    overlay_img = overlay_for_metric(image_bytes, m["name"], color)
    overlay_img.thumbnail((230, 270))
    img_reader = ImageReader(overlay_img)

    img_w, img_h = overlay_img.size
    img_x = 70
    img_y = height - 180 - img_h

    c.drawImage(img_reader, img_x, img_y, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')

    right_x = 340
    info_y = height - 195

    c.setFillColor(HexColor(COLOR_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 10)
    c.drawString(right_x, info_y, "Оценка метрики")

    c.setFillColor(HexColor(color))
    c.setFont(FONT_BOLD, 38)
    c.drawString(right_x, info_y - 42, f"{m['score']:.2f}")

    c.setFillColor(HexColor(COLOR_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 11)
    score_w = c.stringWidth(f"{m['score']:.2f}", FONT_BOLD, 38)
    c.drawString(right_x + score_w + 6, info_y - 42, "/ 10")

    draw_progress_bar(c, right_x, info_y - 60, 190, m["score"], color, height=8)

    c.setFillColor(HexColor(COLOR_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 10)
    c.drawString(right_x, info_y - 85, "Ваш показатель:")
    c.setFillColor(HexColor(COLOR_TEXT))
    c.setFont(FONT_BOLD, 14)
    c.drawString(right_x, info_y - 102, f"{m['value']:.3f}")

    c.setFillColor(HexColor(COLOR_TEXT_MUTED))
    c.setFont(FONT_REGULAR, 8)
    formula_lines = wrap_text(m["formula"], max_chars=46)
    fy = info_y - 117
    for line in formula_lines[:2]:
        c.drawString(right_x, fy, line)
        fy -= 10

    c.setFillColor(HexColor(COLOR_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 10)
    c.drawString(right_x, fy - 8, "Норма:")
    c.setFillColor(HexColor(COLOR_TEXT))
    c.setFont(FONT_BOLD, 14)
    c.drawString(right_x, fy - 25, f"{m['norm']:.3f}")

    dots_filled = closeness_dots(m["score"])
    draw_dots(c, right_x, fy - 50, dots_filled, color)
    c.setFillColor(HexColor(COLOR_TEXT_MUTED))
    c.setFont(FONT_REGULAR, 9)
    c.drawString(right_x + 70, fy - 53, "Близость к норме")

    title, p1, p2, p3, infl = generate_metric_text(m, extra, gender)

    text_y = img_y - 30

    draw_h_line(c, 60, width - 60, text_y + 10)

    c.setFillColor(HexColor(COLOR_TEXT))
    c.setFont(FONT_BOLD, 12)
    title_lines = wrap_text(title, max_chars=78)
    for line in title_lines:
        c.drawString(60, text_y, line)
        text_y -= 16

    text_y -= 8

    text_y = draw_wrapped(c, p1, 60, text_y, max_chars=98, line_height=13, size=9.5, color=COLOR_TEXT)
    text_y -= 10

    if p2:
        text_y = draw_wrapped(c, p2, 60, text_y, max_chars=98, line_height=13, size=9.5, color=COLOR_TEXT)
        text_y -= 10

    if p3:
        text_y = draw_wrapped(c, p3, 60, text_y, max_chars=98, line_height=13, size=9.5, color=COLOR_TEXT_SOFT)
        text_y -= 12

    if text_y > 110:
        draw_h_line(c, 60, width - 60, text_y + 4)
        text_y -= 18

        infl_top = text_y + 12
        infl_bottom = max(60, text_y - 50)
        draw_left_accent_line(c, 60, infl_top, infl_bottom, color)

        c.setFillColor(HexColor(color))
        c.setFont(FONT_BOLD, 10)
        c.drawString(80, text_y, "ВЛИЯНИЕ")
        text_y -= 16

        draw_wrapped(c, infl, 80, text_y, max_chars=92, line_height=13, size=9.5, color=COLOR_TEXT)

    draw_footer(c, width, page_num)


# ================== PDF: RECOMMENDATIONS PAGE ==================
def draw_recommendations(c, width, height, data, gender):
    draw_page_bg(c, width, height)

    c.setFillColor(HexColor(COLOR_TEXT))
    c.setFont(FONT_BOLD, 28)
    c.drawString(60, height - 80, "Рекомендации")

    c.setFillColor(HexColor(COLOR_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 11)
    c.drawString(60, height - 100, "Персональные советы по улучшению")

    draw_h_line(c, 60, width - 60, height - 115)

    strengths, improvements = generate_recommendations(data, gender)

    c.setFillColor(HexColor(COLOR_ACCENT))
    c.setFont(FONT_BOLD, 13)
    c.drawString(60, height - 140, "Что уже отлично")

    y = height - 165
    for i, (name, text) in enumerate(strengths):
        color = METRIC_COLORS[i % len(METRIC_COLORS)]
        block_top = y + 12
        block_bottom = y - 30

        c.setFillColor(HexColor(COLOR_TEXT))
        c.setFont(FONT_BOLD, 10)
        c.drawString(85, y, name)

        text_y = draw_wrapped(c, text, 85, y - 14, max_chars=92, line_height=12, size=9, color=COLOR_TEXT)

        block_bottom = text_y + 4
        draw_left_accent_line(c, 65, block_top, block_bottom, color)

        y = text_y - 10

    y -= 10
    draw_h_line(c, 60, width - 60, y)
    y -= 20

    c.setFillColor(HexColor(COLOR_TITLE))
    c.setFont(FONT_BOLD, 13)
    c.drawString(60, y, "Что можно улучшить")
    y -= 25

    for i, (name, text) in enumerate(improvements):
        color = METRIC_COLORS[(i + 3) % len(METRIC_COLORS)]
        block_top = y + 12

        c.setFillColor(HexColor(COLOR_TEXT))
        c.setFont(FONT_BOLD, 10)
        c.drawString(85, y, name)

        text_y = draw_wrapped(c, text, 85, y - 14, max_chars=92, line_height=12, size=9, color=COLOR_TEXT)

        block_bottom = text_y + 4
        draw_left_accent_line(c, 65, block_top, block_bottom, color)

        y = text_y - 10

        if y < 70:
            break

    c.setFillColor(HexColor(COLOR_TEXT_MUTED))
    c.setFont(FONT_REGULAR, 9)
    c.drawString(60, 50, f"Дата разбора: {data['date']}")

    draw_footer(c, width, 23)


# ================== PDF MAIN ==================
def create_pdf_report(image_bytes, data, gender, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    draw_cover(c, width, height, image_bytes, data)
    c.showPage()

    draw_profile_page(c, width, height, data)
    c.showPage()

    page_num = 3
    for idx, m in enumerate(data["metrics"], start=1):
        draw_metric_page(c, width, height, image_bytes, m, idx, data["extra"], gender, page_num)
        page_num += 1
        c.showPage()

    draw_recommendations(c, width, height, data, gender)
    c.save()
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
    keyboard=[
        [KeyboardButton(text="◀️ Назад в меню")],
    ],
    resize_keyboard=True,
)


# ================== HANDLERS ==================
@dp.message(F.text.in_({"/start", "/help", "◀️ Назад в меню"}))
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "📍 <b>Главное меню</b>\n\n"
        f"<b>{BOT_NAME}</b> математически измеряет, насколько гармонично "
        "черты твоего лица сочетаются друг с другом.\n\n"
        "Твой баланс: <b>0 разборов</b>\n"
        "Сегодня пользователи провели: <b>21 разбор</b>",
        parse_mode="HTML",
        reply_markup=main_keyboard,
    )


@dp.message(F.text == "Что это?")
async def what_is_this(message: Message):
    await message.answer(
        "📍 <b>Главное меню › Что это</b>\n\n"
        f"<b>{BOT_NAME}</b> — сервис, позволяющий математически оценить "
        "гармонию пропорций лица по фотографии.\n\n"
        "🔬 Алгоритм определяет ключевые точки лица, рассчитывает 20 "
        "антропометрических метрик и сравнивает их с нормативными значениями "
        "(отдельно для мужчин и женщин).\n\n"
        "📄 В PDF-отчёт входят:\n"
        "• Итоговая оценка и уровень\n"
        "• Профиль метрик с радар-чартом\n"
        "• 20 страниц с разбором каждой метрики\n"
        "• Визуализация измерений на фото\n"
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
        "📍 <b>Главное меню › Выбор тарифа</b>\n\n"
        "⚜️ <b>План</b> — 1 разбор\n"
        f"💰 <b>Цена</b> — {PRICE_TEXT}\n\n"
        "📚 <b>Что входит:</b>\n"
        "• Персональный PDF-отчёт на 23 страницы\n"
        "• Разбор 20 ключевых метрик лица\n"
        "• Сравнение с нормативами для вашего пола\n"
        "• Наглядная визуализация измерений\n"
        "• Подробное объяснение каждой метрики\n"
        "• Анализ сильных сторон и зон потенциала\n"
        "• Персональные рекомендации\n\n"
        "💡 <b>Пока оплата не подключена — тестовый режим.</b>\n\n"
        "👤 <b>Выберите ваш пол</b>, чтобы алгоритм использовал "
        "корректные антропометрические нормы:",
        parse_mode="HTML",
        reply_markup=gender_keyboard,
    )


@dp.message(AnalysisStates.waiting_for_gender, F.text.in_({"👨 Мужской", "👩 Женский"}))
async def choose_gender(message: Message, state: FSMContext):
    gender = "male" if "Мужской" in message.text else "female"
    user_gender[message.from_user.id] = gender
    await state.set_state(AnalysisStates.waiting_for_photo)

    gender_word = "мужской" if gender == "male" else "женский"
    await message.answer(
        f"✅ Пол выбран: <b>{gender_word}</b>\n\n"
        "📸 <b>Теперь отправьте фото лица.</b>\n\n"
        "Требования к фото:\n"
        "• Лицо строго анфас (прямо в камеру)\n"
        "• Нейтральное выражение, рот закрыт\n"
        "• Хорошее равномерное освещение\n"
        "• Без очков, маски, головного убора\n"
        "• Волосы не закрывают лоб и брови\n"
        "• Фото в высоком качестве",
        parse_mode="HTML",
        reply_markup=cancel_keyboard,
    )


@dp.message(AnalysisStates.waiting_for_gender)
async def wrong_gender(message: Message):
    await message.answer(
        "Пожалуйста, выберите пол кнопкой ниже 👇",
        reply_markup=gender_keyboard,
    )


async def process_image(message: Message, image_bytes: bytes, state: FSMContext):
    gender = user_gender.get(message.from_user.id, "male")

    await message.answer("⏳ <b>Анализирую лицо...</b>\n\nЭто может занять до 30 секунд.", parse_mode="HTML")

    data, error = analyze_face(image_bytes, gender)

    if error:
        await message.answer(
            f"❌ {error}\n\nПопробуйте отправить другое фото.",
            reply_markup=cancel_keyboard,
        )
        return

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_path = tmp.name

        create_pdf_report(image_bytes, data, gender, pdf_path)

        await message.answer(
            "✅ <b>Разбор завершён!</b>\n\n"
            f"Итоговая оценка: <b>{data['score']:.2f} / 10</b>\n"
            f"Уровень: <b>{data['level']}</b>\n\n"
            "Полный отчёт ниже ↓",
            parse_mode="HTML",
        )

        pdf = FSInputFile(pdf_path, filename=f"Отчёт {BOT_NAME}.pdf")
        await message.answer_document(pdf, reply_markup=main_keyboard)

        await state.clear()

    except Exception as e:
        logger.exception("PDF generation failed")
        await message.answer(
            f"❌ Ошибка при создании отчёта: {str(e)[:200]}\n\nПопробуйте ещё раз или отправьте другое фото.",
            reply_markup=cancel_keyboard,
        )


@dp.message(AnalysisStates.waiting_for_photo, F.photo)
async def handle_photo_state(message: Message, state: FSMContext):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    await process_image(message, buf.getvalue(), state)


@dp.message(AnalysisStates.waiting_for_photo, F.document)
async def handle_doc_state(message: Message, state: FSMContext):
    doc = message.document
    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await message.answer("📎 Это не изображение. Пожалуйста, отправьте фото.")
        return
    file = await bot.get_file(doc.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    await process_image(message, buf.getvalue(), state)


@dp.message(AnalysisStates.waiting_for_photo)
async def wrong_state_photo(message: Message):
    await message.answer(
        "📸 Жду фото лица. Отправьте изображение в этот чат.",
        reply_markup=cancel_keyboard,
    )


@dp.message(F.photo)
async def handle_photo_no_state(message: Message):
    await message.answer(
        "👋 Чтобы получить разбор, сначала нажмите кнопку <b>«💠 Хочу получить свой разбор»</b> "
        "и выберите пол.",
        parse_mode="HTML",
        reply_markup=main_keyboard,
    )


@dp.message(F.document)
async def handle_doc_no_state(message: Message):
    await message.answer(
        "👋 Чтобы получить разбор, сначала нажмите кнопку <b>«💠 Хочу получить свой разбор»</b> "
        "и выберите пол.",
        parse_mode="HTML",
        reply_markup=main_keyboard,
    )


@dp.message()
async def fallback(message: Message):
    await message.answer(
        "📸 Чтобы начать разбор — нажмите <b>«💠 Хочу получить свой разбор»</b>.",
        parse_mode="HTML",
        reply_markup=main_keyboard,
    )


# ================== ENTRY POINT ==================
async def main():
    logger.info(f"{BOT_NAME} starting...")
    threading.Thread(target=run_web, daemon=True).start()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
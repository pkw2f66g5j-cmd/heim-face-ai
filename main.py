from flask import Flask
import threading
import os
import asyncio
import io
import json
import logging
import math
import tempfile
from datetime import datetime, date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import cv2
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile, KeyboardButton, ReplyKeyboardMarkup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

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
PRICE_TEXT = "1 000 ₽"

# ================== PALETTE ==================
COLOR_BG         = "#14101A"
COLOR_BG_SOFT    = "#1C1622"
COLOR_TITLE      = "#E8D5A0"
COLOR_ACCENT     = "#E5A8A1"
COLOR_TEXT       = "#F0E6D8"
COLOR_TEXT_SOFT  = "#B8A99A"
COLOR_TEXT_MUTED = "#7A6F66"
COLOR_BAR_BG     = "#2A2230"
COLOR_LINE       = "#3A2F3F"

METRIC_COLORS = [
    "#D4AF37","#E8A87C","#D4A5C5","#A687C9","#C9A582",
    "#D4AF37","#E8A87C","#D4A5C5","#A687C9","#C9A582",
    "#D4AF37","#E8A87C","#D4A5C5","#A687C9","#C9A582",
    "#D4AF37","#E8A87C","#D4A5C5","#A687C9","#C9A582",
]

# ================== TIER SYSTEM ==================
TIERS_MALE = [
    (0.0,  3.0,  "Sub3",      "Sub3",                    "#FF4444"),
    (3.0,  4.5,  "Sub4",      "Sub4 · Below Average",    "#FF7744"),
    (4.5,  5.5,  "LTN",       "LTN · Low Tier Normie",   "#FFAA44"),
    (5.5,  6.5,  "NTN",       "NTN · Mid Tier Normie",   "#FFCC44"),
    (6.5,  7.2,  "HTN-",      "HTN- · Lower High Tier",  "#CCDD44"),
    (7.2,  7.8,  "HTN",       "HTN · High Tier Normie",  "#AADD44"),
    (7.8,  8.4,  "HTN+",      "HTN+ · Upper High Tier",  "#88DD44"),
    (8.4,  9.0,  "Chad",      "Chad · High Value Male",  "#44DDAA"),
    (9.0,  9.5,  "High Chad", "High Chad",               "#44CCFF"),
    (9.5, 10.1,  "Gigachad",  "Gigachad · Top 1%",       "#D4AF37"),
]

TIERS_FEMALE = [
    (0.0,  3.0,  "Subpar",    "Subpar",                        "#FF4444"),
    (3.0,  4.5,  "Below Avg", "Below Average",                  "#FF7744"),
    (4.5,  5.5,  "Average",   "Average · Plain Jane",           "#FFAA44"),
    (5.5,  6.5,  "Pretty",    "Pretty · Above Average",         "#FFCC44"),
    (6.5,  7.2,  "Attractive","Attractive · LTG",               "#CCDD44"),
    (7.2,  7.8,  "HTG-",      "HTG- · Lower High Tier Girl",    "#AADD44"),
    (7.8,  8.4,  "HTG",       "HTG · High Tier Girl",           "#88DD44"),
    (8.4,  9.0,  "Top Girl",  "Top Girl · Highly Attractive",   "#44DDAA"),
    (9.0,  9.5,  "Model",     "Model Tier",                     "#44CCFF"),
    (9.5, 10.1,  "Goddess",   "Goddess · Top 1%",               "#D4AF37"),
]


def get_tier(score, gender):
    tiers = TIERS_FEMALE if gender == "female" else TIERS_MALE
    for lo, hi, short, full, color in tiers:
        if lo <= score < hi:
            return short, full, color
    last = tiers[-1]
    return last[2], last[3], last[4]


# ================== WEB SERVER ==================
app = Flask(__name__)

@app.route("/")
def home():
    return f"{BOT_NAME} bot is running"

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


# ================== LOGGING ==================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ================== BOT INIT ==================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN is not set")

bot = Bot(token=BOT_TOKEN)
dp  = Dispatcher(storage=MemoryStorage())


# ================== FSM ==================
class AnalysisStates(StatesGroup):
    waiting_for_gender = State()
    waiting_for_photo  = State()

user_gender = {}


# ================== COUNTERS ==================
COUNTERS_FILE = "counters.json"

def _load_counters():
    if os.path.exists(COUNTERS_FILE):
        try:
            with open(COUNTERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"total": 0, "today_date": str(date.today()), "today": 0, "users": {}}

def _save_counters(c):
    with open(COUNTERS_FILE, "w") as f:
        json.dump(c, f)

def increment_counter(user_id: int):
    c = _load_counters()
    today = str(date.today())
    if c.get("today_date") != today:
        c["today_date"] = today
        c["today"] = 0
    c["total"] += 1
    c["today"] += 1
    uid = str(user_id)
    c["users"][uid] = c["users"].get(uid, 0) + 1
    _save_counters(c)
    return c

def get_counters(user_id: int):
    c = _load_counters()
    today = str(date.today())
    if c.get("today_date") != today:
        c["today"] = 0
    uid = str(user_id)
    return c.get("total", 0), c.get("today", 0), c["users"].get(uid, 0)


# ================== FONTS ==================
def setup_fonts():
    candidates = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ]
    for reg, bold in candidates:
        if os.path.exists(reg):
            try:
                pdfmetrics.registerFont(TTFont("MainFont", reg))
                pdfmetrics.registerFont(TTFont("MainFontBold", bold))
                return "MainFont", "MainFontBold"
            except Exception:
                continue
    return "Helvetica", "Helvetica-Bold"

FONT_REGULAR, FONT_BOLD = setup_fonts()


# ================== MEDIAPIPE ==================
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True,
)

IDX = {
    "face_left": 234, "face_right": 454, "chin": 152, "forehead": 10,
    "nose_bridge": 168, "nose_top": 6, "nose_base": 2, "nose_tip": 4,
    "nose_left": 129, "nose_right": 358,
    "mouth_left": 61, "mouth_right": 291,
    "upper_lip": 13, "lower_lip": 14,
    "upper_lip_top": 0, "lower_lip_bottom": 17,
    "left_eye_outer": 33, "left_eye_inner": 133,
    "right_eye_inner": 362, "right_eye_outer": 263,
    "left_eye_top": 159, "left_eye_bottom": 145,
    "right_eye_top": 386, "right_eye_bottom": 374,
    "left_brow_inner": 55, "left_brow_mid": 105, "left_brow_outer": 70,
    "right_brow_inner": 285, "right_brow_mid": 334, "right_brow_outer": 300,
    "jaw_left": 172, "jaw_right": 397,
    "jaw_left_lower": 150, "jaw_right_lower": 379,
    "cheek_left": 234, "cheek_right": 454,
    "forehead_left": 103, "forehead_right": 332, "forehead_top": 10,
}


# ================== NORMS ==================
NORMS_MALE = {
    "Симметрия лица":         {"norm": 0.970, "sigma": 0.055, "formula": "Зеркальность точек / центральная ось"},
    "Пропорции лица":         {"norm": 0.890, "sigma": 0.055, "formula": "Высота лица (переносица→подбородок) / ширина скул"},
    "Вертикальный баланс":    {"norm": 0.730, "sigma": 0.070, "formula": "Средняя треть / нижняя треть лица"},
    "Баланс скул и челюсти":  {"norm": 1.355, "sigma": 0.080, "formula": "Ширина скул / ширина челюсти"},
    "Размер глаз":            {"norm": 0.223, "sigma": 0.018, "formula": "Ширина глаза / ширина лица"},
    "Расстояние между глазами":{"norm": 0.268,"sigma": 0.020, "formula": "Расстояние между глазами / ширина лица"},
    "Наклон глаз":            {"norm": 0.040, "sigma": 0.030, "formula": "Наклон уголков глаза / ширина глаза"},
    "Ширина носа":            {"norm": 0.233, "sigma": 0.018, "formula": "Ширина крыльев носа / ширина лица"},
    "Ширина рта":             {"norm": 0.402, "sigma": 0.030, "formula": "Ширина рта / ширина скул"},
    "Длина носа":             {"norm": 0.421, "sigma": 0.035, "formula": "Длина носа / высота лица"},
    "Длина подбородка":       {"norm": 0.286, "sigma": 0.030, "formula": "Нижняя губа → подбородок / высота лица"},
    "Контур подбородка":      {"norm": 0.632, "sigma": 0.045, "formula": "Угол сужения подбородка"},
    "Нос к ширине рта":       {"norm": 0.575, "sigma": 0.050, "formula": "Ширина носа / ширина рта"},
    "Биокулярная ширина":     {"norm": 0.711, "sigma": 0.045, "formula": "Внешние углы глаз / ширина лица"},
    "Ширина лба":             {"norm": 0.916, "sigma": 0.055, "formula": "Ширина лба / ширина лица"},
    "Полнота губ":            {"norm": 0.339, "sigma": 0.055, "formula": "Высота губ / ширина рта"},
    "Пропорции губ":          {"norm": 0.634, "sigma": 0.090, "formula": "Верхняя губа / нижняя губа"},
    "Челюсть к ширине рта":   {"norm": 1.841, "sigma": 0.140, "formula": "Ширина челюсти / ширина рта"},
    "Форма глаз":             {"norm": 0.350, "sigma": 0.045, "formula": "Высота глаза / ширина глаза"},
    "Высота бровей":          {"norm": 0.377, "sigma": 0.070, "formula": "Расстояние брови до века / ширина глаза"},
}

NORMS_FEMALE = {
    "Симметрия лица":         {"norm": 0.972, "sigma": 0.050, "formula": "Зеркальность точек / центральная ось"},
    "Пропорции лица":         {"norm": 0.920, "sigma": 0.055, "formula": "Высота лица (переносица→подбородок) / ширина скул"},
    "Вертикальный баланс":    {"norm": 0.760, "sigma": 0.070, "formula": "Средняя треть / нижняя треть лица"},
    "Баланс скул и челюсти":  {"norm": 1.420, "sigma": 0.080, "formula": "Ширина скул / ширина челюсти"},
    "Размер глаз":            {"norm": 0.232, "sigma": 0.018, "formula": "Ширина глаза / ширина лица"},
    "Расстояние между глазами":{"norm": 0.265,"sigma": 0.020, "formula": "Расстояние между глазами / ширина лица"},
    "Наклон глаз":            {"norm": 0.055, "sigma": 0.030, "formula": "Наклон уголков глаза / ширина глаза"},
    "Ширина носа":            {"norm": 0.215, "sigma": 0.018, "formula": "Ширина крыльев носа / ширина лица"},
    "Ширина рта":             {"norm": 0.395, "sigma": 0.030, "formula": "Ширина рта / ширина скул"},
    "Длина носа":             {"norm": 0.405, "sigma": 0.035, "formula": "Длина носа / высота лица"},
    "Длина подбородка":       {"norm": 0.265, "sigma": 0.030, "formula": "Нижняя губа → подбородок / высота лица"},
    "Контур подбородка":      {"norm": 0.595, "sigma": 0.045, "formula": "Угол сужения подбородка"},
    "Нос к ширине рта":       {"norm": 0.545, "sigma": 0.050, "formula": "Ширина носа / ширина рта"},
    "Биокулярная ширина":     {"norm": 0.708, "sigma": 0.045, "formula": "Внешние углы глаз / ширина лица"},
    "Ширина лба":             {"norm": 0.905, "sigma": 0.055, "formula": "Ширина лба / ширина лица"},
    "Полнота губ":            {"norm": 0.395, "sigma": 0.055, "formula": "Высота губ / ширина рта"},
    "Пропорции губ":          {"norm": 0.665, "sigma": 0.090, "formula": "Верхняя губа / нижняя губа"},
    "Челюсть к ширине рта":   {"norm": 1.785, "sigma": 0.140, "formula": "Ширина челюсти / ширина рта"},
    "Форма глаз":             {"norm": 0.385, "sigma": 0.045, "formula": "Высота глаза / ширина глаза"},
    "Высота бровей":          {"norm": 0.420, "sigma": 0.070, "formula": "Расстояние брови до века / ширина глаза"},
}

def get_norms(gender):
    return NORMS_FEMALE if gender == "female" else NORMS_MALE


# ================== HELPERS ==================
def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def calc_score(value, norm, sigma):
    z = abs(value - norm) / sigma
    return round(max(0, min(10, 10 - z * 2.2)), 2)

def calc_z(value, norm, sigma):
    return (value - norm) / sigma

def wrap_text(text, max_chars=74):
    words = text.split()
    lines, current = [], ""
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
    if score >= 9.0: return "Исключительный уровень"
    if score >= 8.5: return "Значительно выше среднего"
    if score >= 7.5: return "Выше среднего"
    if score >= 6.5: return "Средний уровень"
    if score >= 5.5: return "Чуть ниже среднего"
    return "Есть выраженные зоны потенциала"

def get_top_percent(score):
    if score >= 9.5: return 1
    if score >= 9.0: return 3
    if score >= 8.5: return 7
    if score >= 8.0: return 12
    if score >= 7.5: return 20
    if score >= 7.0: return 30
    if score >= 6.5: return 45
    if score >= 6.0: return 60
    return 75

def closeness_dots(score):
    if score >= 9: return 5
    if score >= 7.5: return 4
    if score >= 6: return 3
    if score >= 4: return 2
    return 1

def _gender_form(gender, male_val, female_val):
    return female_val if gender == "female" else male_val

def direction_word(z, hw="выше нормы", lw="ниже нормы", ew="на уровне нормы"):
    if abs(z) < 0.3: return ew
    return hw if z > 0 else lw

def deviation_label(z_abs):
    if z_abs < 0.3: return "практически совпадает с нормой"
    if z_abs < 0.7: return "близко к норме"
    if z_abs < 1.2: return "немного отличается от нормы"
    if z_abs < 1.8: return "заметно отличается от нормы"
    return "значительно отличается от нормы"


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
        return None, "Лицо не обнаружено. Отправь чёткое фото строго анфас при хорошем освещении."

    lm = results.multi_face_landmarks[0].landmark

    def pt(name):
        idx = IDX[name]
        return lm[idx].x * w, lm[idx].y * h

    face_w       = dist(pt("face_left"),       pt("face_right"))
    face_h       = dist(pt("nose_bridge"),      pt("chin"))
    cheek_w      = dist(pt("cheek_left"),       pt("cheek_right"))
    jaw_w        = dist(pt("jaw_left"),         pt("jaw_right"))
    jaw_w_lower  = dist(pt("jaw_left_lower"),   pt("jaw_right_lower"))
    nose_w       = dist(pt("nose_left"),        pt("nose_right"))
    nose_len     = dist(pt("nose_bridge"),      pt("nose_base"))
    mouth_w      = dist(pt("mouth_left"),       pt("mouth_right"))
    left_eye_w   = dist(pt("left_eye_outer"),   pt("left_eye_inner"))
    right_eye_w  = dist(pt("right_eye_inner"),  pt("right_eye_outer"))
    eye_w        = (left_eye_w + right_eye_w) / 2
    left_eye_h   = dist(pt("left_eye_top"),     pt("left_eye_bottom"))
    right_eye_h  = dist(pt("right_eye_top"),    pt("right_eye_bottom"))
    eye_h        = (left_eye_h + right_eye_h) / 2
    eye_inner_dist = dist(pt("left_eye_inner"), pt("right_eye_inner"))
    biocular_w   = dist(pt("left_eye_outer"),   pt("right_eye_outer"))
    forehead_w   = dist(pt("forehead_left"),    pt("forehead_right"))
    upper_lip_h  = dist(pt("upper_lip_top"),    pt("upper_lip"))
    lower_lip_h  = dist(pt("lower_lip"),        pt("lower_lip_bottom"))
    chin_len     = dist(pt("lower_lip"),        pt("chin"))
    middle_third = dist(pt("nose_bridge"),      pt("nose_base"))
    lower_third  = dist(pt("nose_base"),        pt("chin"))

    midline_x = (pt("forehead")[0] + pt("chin")[0]) / 2
    def axis_dev(name):
        return (pt(name)[0] - midline_x) / face_w

    devs = {k: axis_dev(k) for k in [
        "left_eye_inner","right_eye_inner","left_eye_outer","right_eye_outer",
        "mouth_left","mouth_right","nose_left","nose_right"
    ]}

    sym_inner = abs(abs(devs["left_eye_inner"])  - abs(devs["right_eye_inner"]))
    sym_outer = abs(abs(devs["left_eye_outer"])  - abs(devs["right_eye_outer"]))
    sym_mouth = abs(abs(devs["mouth_left"])      - abs(devs["mouth_right"]))
    sym_nose  = abs(abs(devs["nose_left"])       - abs(devs["nose_right"]))
    avg_sym   = (sym_inner + sym_outer + sym_mouth + sym_nose) / 4
    symmetry  = max(0, 1 - avg_sym * 5)

    eye_tilt     = abs(pt("right_eye_outer")[1] - pt("left_eye_outer")[1]) / max(biocular_w, 1)
    chin_contour = jaw_w_lower / max(jaw_w, 1)
    brow_height  = (dist(pt("left_brow_mid"), pt("left_eye_top")) +
                    dist(pt("right_brow_mid"), pt("right_eye_top"))) / 2 / max(eye_w, 1)

    values = {
        "Симметрия лица":          symmetry,
        "Пропорции лица":          face_h / face_w,
        "Вертикальный баланс":     middle_third / lower_third,
        "Баланс скул и челюсти":   cheek_w / jaw_w,
        "Размер глаз":             eye_w / face_w,
        "Расстояние между глазами":eye_inner_dist / face_w,
        "Наклон глаз":             eye_tilt,
        "Ширина носа":             nose_w / face_w,
        "Ширина рта":              mouth_w / cheek_w,
        "Длина носа":              nose_len / face_h,
        "Длина подбородка":        chin_len / face_h,
        "Контур подбородка":       chin_contour,
        "Нос к ширине рта":        nose_w / mouth_w,
        "Биокулярная ширина":      biocular_w / face_w,
        "Ширина лба":              forehead_w / face_w,
        "Полнота губ":             (upper_lip_h + lower_lip_h) / mouth_w,
        "Пропорции губ":           upper_lip_h / max(lower_lip_h, 1),
        "Челюсть к ширине рта":    jaw_w / mouth_w,
        "Форма глаз":              eye_h / eye_w,
        "Высота бровей":           brow_height,
    }

    norms   = get_norms(gender)
    metrics = []
    for name, value in values.items():
        n   = norms[name]
        s   = calc_score(value, n["norm"], n["sigma"])
        z   = calc_z(value, n["norm"], n["sigma"])
        metrics.append({"name": name, "value": round(value,4),
                        "norm": n["norm"], "sigma": n["sigma"],
                        "score": s, "z": z, "formula": n["formula"]})

    total = round(sum(m["score"] for m in metrics) / len(metrics), 2)
    srt   = sorted(metrics, key=lambda x: x["score"], reverse=True)

    extra = {
        "sym_inner_pct": sym_inner*100, "sym_outer_pct": sym_outer*100,
        "sym_mouth_pct": sym_mouth*100, "sym_nose_pct":  sym_nose*100,
        "devs": devs,
        "nose_w_ratio":   nose_w / face_w,
        "mouth_w_ratio":  mouth_w / face_w,
        "jaw_w_ratio":    jaw_w / face_w,
        "upper_lip_ratio":upper_lip_h / max(face_h,1),
        "lower_lip_ratio":lower_lip_h / max(face_h,1),
    }

    tier_short, tier_full, tier_color = get_tier(total, gender)

    return {
        "score": total, "level": get_level(total),
        "top_percent": get_top_percent(total),
        "tier_short": tier_short, "tier_full": tier_full, "tier_color": tier_color,
        "metrics": metrics,
        "strengths": srt[:3], "weak": srt[-3:][::-1],
        "extra": extra, "gender": gender,
        "date": datetime.now().strftime("%d.%m.%Y"),
    }, None


# ================== TEXT GENERATORS ==================
def _gender_form(gender, male_val, female_val):
    return female_val if gender == "female" else male_val


def text_symmetry(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]
    si = extra["sym_inner_pct"]
    so = extra["sym_outer_pct"]
    sm = extra["sym_mouth_pct"]
    sn = extra["sym_nose_pct"]

    if m["score"] >= 9.0:
        title = "Симметрия лица практически идеальна — редкий результат."
    elif m["score"] >= 7.5:
        title = "Симметрия лица высокая — лицо воспринимается гармонично."
    elif m["score"] >= 6.0:
        title = "Симметрия лица умеренная — небольшие отклонения присутствуют."
    else:
        title = "Симметрия лица ниже среднего — есть выраженные асимметрии."

    p1 = (f"Метрика оценивает зеркальность левой и правой сторон лица по четырём парам "
          f"контрольных точек. Твой результат — {val:.4f} при норме {norm:.4f} "
          f"(отклонение {z_abs:.2f}σ). "
          f"Наибольшие отклонения: внутренние углы глаз — {si:.1f}%, "
          f"внешние углы глаз — {so:.1f}%, углы рта — {sm:.1f}%, крылья носа — {sn:.1f}%.")

    worst = max([(si,"нижняя линия щёк"),(so,"внешние углы глаз"),
                 (sm,"углы рта"),(sn,"крылья носа")], key=lambda x: x[0])
    if worst[0] < 3.0:
        p2 = ("Даже наибольшее отклонение находится в пределах незаметных невооружённым глазом. "
              "Такой уровень симметрии — один из ключевых маркеров генетического здоровья "
              "и воспринимается как признак высокой привлекательности.")
    elif worst[0] < 6.0:
        p2 = (f"Наибольшее отклонение ({worst[1]}, ~{worst[0]:.1f}%) находится в пределах "
              f"естественного разброса — большинство людей не заметят его при взгляде на лицо. "
              f"Незначительная асимметрия — норма даже для привлекательных лиц.")
    else:
        p2 = (f"Заметное отклонение в зоне «{worst[1]}» (~{worst[0]:.1f}%) может визуально "
              f"восприниматься как асимметрия. Это поддаётся визуальной коррекции стрижкой, "
              f"укладкой и правильным ракурсом.")

    p3 = ("Исследования Thornhill & Gangestad (Facial attractiveness, 1999) подтвердили: "
          "симметрия лица устойчиво коррелирует с восприятием привлекательности "
          "и считается маркером генетического здоровья.")
    infl = "Симметрия — фундамент первого впечатления: лицо воспринимается гармоничным ещё до того, как мозг успевает проанализировать отдельные черты."
    return title, p1, p2, p3, infl


def text_proportions(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = _gender_form(gender,
            "Пропорции лица близки к классическому мужскому эталону.",
            "Пропорции лица близки к классическому женскому эталону.")
    elif z > 0:
        title = "Лицо вытянутое — вертикаль преобладает над горизонталью."
    else:
        title = "Лицо широкое — горизонталь преобладает над вертикалью."

    p1 = (f"Метрика измеряет соотношение высоты лица (от переносицы до подбородка) "
          f"к ширине скул. Твой результат — {val:.4f} при норме {norm:.4f} "
          f"(отклонение {z_abs:.2f}σ, {direction_word(z)}).")

    if abs(z) < 0.5:
        p2 = ("Сбалансированные пропорции создают универсально привлекательный овал лица. "
              "Это одна из базовых метрик гармонии — близость к норме здесь "
              "работает в твою пользу.")
    elif z > 0:
        p2 = _gender_form(gender,
            ("Вытянутое лицо визуально воспринимается как более утончённое и аристократичное. "
             "Стрижки с объёмом по бокам и горизонтальные элементы стиля визуально "
             "балансируют пропорции."),
            ("Вытянутое лицо ассоциируется с элегантностью. Стрижки с объёмом по бокам "
             "визуально скорректируют овал."))
    else:
        p2 = _gender_form(gender,
            ("Широкое лицо воспринимается как мощное и доминантное. Вертикальные элементы "
             "в стрижке и стиле визуально удлинят пропорции."),
            ("Широкое лицо создаёт миловидный образ. Вертикальные элементы стрижки "
             "визуально удлинят овал."))

    p3 = ("Farkas et al. (Anthropometry of the Head and Face, 1994) установили нормативные "
          "пропорции лица для европейской популяции — именно они использованы как эталон.")
    infl = "Пропорции лица определяют первое считываемое впечатление об овале — вытянутый, округлый или гармоничный."
    return title, p1, p2, p3, infl


def text_vertical_balance(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = "Вертикальный баланс лица близок к классическому."
    elif z > 0.3:
        title = "Средняя треть лица доминирует над нижней."
    elif z < -0.3:
        title = "Нижняя треть лица увеличена относительно средней."
    else:
        title = "Вертикальный баланс в пределах нормы."

    p1 = (f"Метрика сравнивает среднюю треть лица (переносица → основание носа) "
          f"с нижней третью (основание носа → подбородок). "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).")

    if abs(z) < 0.5:
        p2 = "Классическое деление на трети — один из ключевых принципов гармонии лица по Леонардо да Винчи. Твои пропорции близки к этому эталону."
    elif z > 0:
        p2 = "Доминирование средней трети создаёт выразительную носовую зону. Щетина или коррекция линии нижней трети визуально восстановит баланс."
    else:
        p2 = "Удлинённая нижняя треть — характеристика выразительного, структурного лица. Правильная стрижка с объёмом на лбу визуально выровняет трети."

    p3 = ("Принцип деления лица на три равные трети описан ещё Леонардо да Винчи "
          "и подтверждён современными антропометрическими исследованиями Farkas (1994).")
    infl = "Вертикальный баланс определяет, насколько лицо воспринимается как 'правильно сложенное' при взгляде анфас."
    return title, p1, p2, p3, infl


def text_cheek_jaw(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = _gender_form(gender,
            "Соотношение скул и челюсти близко к идеальному — архитектурный мужской контур.",
            "Скулы заметно шире челюсти — классический признак женственности.")
    elif z > 0:
        title = "Скулы существенно шире челюсти — выраженный контраст зон."
    else:
        title = "Соотношение скул и челюсти менее выражено, чем в норме."

    p1 = (f"Метрика измеряет отношение ширины скул к ширине челюсти. "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ, {direction_word(z)}).")

    p2 = _gender_form(gender,
        ("Для мужского лица идеальный баланс скул и челюсти создаёт архитектурный, "
         "структурный контур. Широкие скулы при выраженной челюсти — сочетание, "
         "ассоциирующееся с доминантностью и физической силой."),
        ("Для женского лица превышение скул над челюстью создаёт характерный "
         "«сердцевидный» или «овальный» контур — один из наиболее привлекательных "
         "типов по данным исследований Cunningham et al."))

    p3 = ("Cunningham et al. (Their ideas of beauty are, on the whole, the same as ours, 1995): "
          "соотношение скул и челюсти — один из ключевых предикторов воспринимаемой привлекательности.")
    infl = _gender_form(gender,
        "Баланс скул и челюсти формирует мужественный силуэт нижней части лица.",
        "Баланс скул и челюсти определяет форму овала и считываемую женственность.")
    return title, p1, p2, p3, infl


def text_eye_size(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = "Размер глаз пропорционален лицу — естественно выразительный взгляд."
    elif z > 0:
        title = "Глаза крупнее среднего относительно лица."
    else:
        title = "Глаза компактнее среднего относительно лица."

    p1 = (f"Метрика — отношение средней ширины глаза к ширине лица. "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).")

    p2 = _gender_form(gender,
        ("Пропорциональный размер глаз у мужчин создаёт сбалансированный, уверенный взгляд. "
         "Слишком крупные глаза воспринимаются как более юношеские, слишком малые — "
         "как жёсткие. Норма здесь даёт наибольший эффект."),
        ("Крупные глаза — один из главных маркеров женской привлекательности, "
         "ассоциирующихся с молодостью и здоровьем. Чем ближе к верхней границе нормы, "
         "тем выразительнее воспринимается взгляд."))

    p3 = ("Grammer et al. (Darwinian aesthetics, 2003): относительный размер глаз "
          "устойчиво коррелирует с воспринимаемым возрастом и привлекательностью.")
    infl = "Размер глаз — первое, что считывается при зрительном контакте; он задаёт тон всему впечатлению от взгляда."
    return title, p1, p2, p3, infl


def text_eye_distance(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = "Расстояние между глазами оптимально — идеальный баланс глазной зоны."
    elif z > 0:
        title = "Глаза посажены шире нормы — гипертелоризм."
    else:
        title = "Глаза посажены ближе нормы — гипотелоризм."

    p1 = (f"Метрика — расстояние между внутренними углами глаз / ширина лица. "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).")

    if abs(z) < 0.5:
        p2 = "Оптимальное расстояние между глазами создаёт гармоничную среднюю зону лица и сбалансированный взгляд."
    elif z > 0:
        p2 = ("Широко посаженные глаза ассоциируются с открытостью и доброжелательностью. "
              "Визуально это можно скорректировать макияжем или стрижкой с акцентом на переносицу.")
    else:
        p2 = ("Близко посаженные глаза создают сфокусированный, интенсивный взгляд. "
              "Визуально корректируется светлыми тенями во внутреннем уголке.")

    p3 = ("Farkas (Anthropometry of the Head and Face, 1994): норма межглазного расстояния "
          "составляет ~26-27% ширины лица для взрослых европейской популяции.")
    infl = "Межглазное расстояние формирует характер взгляда — от открытого до интенсивного."
    return title, p1, p2, p3, infl


def text_eye_tilt(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = _gender_form(gender,
            "Наклон глаз близок к норме — уверенный, сфокусированный взгляд.",
            "Наклон глаз близок к норме — мягкий, гармоничный взгляд.")
    elif z > 0.3:
        title = "Уголки глаз опущены — нисходящий наклон."
    else:
        title = "Уголки глаз приподняты — восходящий наклон (hunter eyes / fox eyes)."

    p1 = (f"Метрика оценивает разницу высот внешних уголков глаз относительно ширины "
          f"биокулярной зоны. Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).")

    if z < -0.3:
        p2 = _gender_form(gender,
            ("Приподнятые уголки глаз (hunter eyes) — один из наиболее ценимых признаков "
             "мужской привлекательности. Создают хищный, доминантный взгляд, "
             "ассоциирующийся с высоким статусом."),
            ("Приподнятые уголки глаз (fox eyes) — один из самых привлекательных типов взгляда "
             "для женского лица. Создают экзотический, соблазнительный образ."))
    elif z > 0.3:
        p2 = ("Опущенные уголки глаз создают более мягкий, добродушный взгляд. "
              "Визуально корректируется стрелками с подъёмом во внешнем уголке.")
    else:
        p2 = "Нейтральный наклон глаз создаёт сбалансированный, открытый взгляд."

    p3 = ("Fink & Penton-Voak (Evolutionary psychology of facial attractiveness, 2002): "
          "наклон глазной щели устойчиво влияет на восприятие доминантности и "
          "сексуальной привлекательности.")
    infl = "Наклон глаз — один из главных факторов «характера» взгляда; формирует впечатление от сотых долей секунды."
    return title, p1, p2, p3, infl


def text_nose_width(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = "Ширина носа пропорциональна лицу — гармоничная носовая зона."
    elif z > 0:
        title = "Нос шире нормы относительно ширины лица."
    else:
        title = "Нос уже нормы относительно ширины лица."

    p1 = (f"Метрика — ширина крыльев носа / ширина лица. "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).")

    p2 = _gender_form(gender,
        ("Для мужчины пропорциональный нос поддерживает баланс центральной зоны лица. "
         f"{'Широкий нос визуально расширяет среднюю зону — контурирование боковых поверхностей поможет скорректировать.' if z>0.3 else 'Узкий нос создаёт утончённый центр лица.' if z<-0.3 else 'Норма здесь работает на гармонию всего овала.'}"),
        ("Для женщины узкий аккуратный нос — одна из ценимых черт. "
         f"{'Широкий нос корректируется контурированием переносицы и крыльев.' if z>0.3 else 'Аккуратный нос гармонично вписывается в женский овал.' if z<-0.3 else 'Пропорциональный нос — нейтральная и гармоничная черта.'}"))

    p3 = ("Perrett et al. (Effects of sexual dimorphism on facial attractiveness, 1998): "
          "ширина носа влияет на считываемый уровень маскулинности/фемининности лица.")
    infl = "Ширина носа задаёт визуальный «якорь» центральной зоны — от неё зависит баланс между глазами и ртом."
    return title, p1, p2, p3, infl


def text_mouth_width(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = "Ширина рта гармонично вписывается в пропорции лица."
    elif z > 0:
        title = "Рот шире нормы относительно ширины скул."
    else:
        title = "Рот уже нормы относительно ширины скул."

    p1 = (f"Метрика — ширина рта / ширина скул. "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).")

    p2 = _gender_form(gender,
        (f"{'Широкий рот добавляет лицу выразительности и харизмы.' if z>0.3 else 'Компактный рот создаёт аккуратный нижний контур.' if z<-0.3 else 'Пропорциональная ширина рта — один из маркеров гармоничного мужского лица.'}"),
        (f"{'Широкий рот ассоциируется с чувственностью и экспрессивностью.' if z>0.3 else 'Компактный рот создаёт утончённый образ.' if z<-0.3 else 'Пропорциональный рот — один из ключевых элементов женской гармонии нижней зоны лица.'}"))

    p3 = ("Cunningham (Measuring the physical in physical attractiveness, 1986): "
          "ширина рта в соотношении со скулами — значимый предиктор воспринимаемой привлекательности.")
    infl = "Ширина рта определяет визуальный вес нижней части лица и экспрессивность улыбки."
    return title, p1, p2, p3, infl


def text_nose_length(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = "Длина носа пропорциональна высоте лица — классические пропорции."
    elif z > 0:
        title = "Нос длиннее нормы относительно высоты лица."
    else:
        title = "Нос короче нормы относительно высоты лица."

    p1 = (f"Метрика — длина носа (переносица → основание) / высота лица. "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).")

    if abs(z) < 0.5:
        p2 = "Пропорциональная длина носа — один из классических маркеров гармонии средней трети лица."
    elif z > 0:
        p2 = ("Длинный нос визуально удлиняет среднюю треть лица. "
              "Контурирование кончика поможет визуально скорректировать длину.")
    else:
        p2 = ("Короткий нос визуально укорачивает среднюю зону — "
              "это часто воспринимается как более юношеская черта.")

    p3 = ("Farkas (1994): длина носа в норме составляет около 42% высоты лица "
          "для мужчин и 40% для женщин европейской популяции.")
    infl = "Длина носа задаёт визуальное ощущение средней трети и влияет на баланс всех трёх зон лица."
    return title, p1, p2, p3, infl


def text_chin_length(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = _gender_form(gender,
            "Подбородок выражен и пропорционально удлинён — классический маркер мужественности.",
            "Подбородок гармонично пропорционален лицу.")
    elif z > 0:
        title = _gender_form(gender,
            "Подбородок удлинён — выраженный нижний силуэт.",
            "Подбородок чуть длиннее среднего.")
    else:
        title = "Подбородок компактный — смягчает нижнюю часть лица."

    p1 = (f"Метрика — расстояние нижняя губа → подбородок / высота лица. "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ, {direction_word(z)}).")

    p2 = _gender_form(gender,
        (f"{'Выраженный подбородок — один из классических маркеров мужественности, ассоциирующийся с решительностью.' if z>=0 else 'Компактный подбородок смягчает нижнюю треть — может визуально омолаживать образ.'}"),
        (f"{'Удлинённый подбородок придаёт лицу выразительность и характер.' if z>0 else 'Компактный подбородок ассоциируется с миловидностью и юностью.'}"))

    p3 = ("Thornhill & Gangestad (1999): длина подбородка связана с восприятием "
          "доминантности и зрелости у мужчин.")
    infl = "Подбородок завершает силуэт лица и формирует впечатление силы и определённости."
    return title, p1, p2, p3, infl


def text_chin_contour(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = _gender_form(gender,
            "Угол челюсти широкий и квадратный — признак волевого характера.",
            "Контур подбородка гармонично сбалансирован.")
    elif z > 0:
        title = "Угол челюсти шире среднего — структурный нижний контур."
    else:
        title = "Угол челюсти острее среднего — V-образный силуэт."

    p1 = (f"Метрика оценивает угол сужения подбородка через соотношение нижней и верхней "
          f"частей челюсти. Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).")

    p2 = _gender_form(gender,
        ("Квадратная челюсть — ключевой маркер мужественности, ассоциирующийся с волевым "
         "характером и физической силой. Для мужского лица отклонение в сторону «шире» — "
         "позитивный признак."),
        ("V-образный подбородок ассоциируется с женственностью и утончённостью. "
         "Для женского лица острый контур — одна из ценимых черт."))

    p3 = ("Cunningham et al. (1995): контур челюсти устойчиво оценивается как один "
          "из ключевых факторов восприятия лица.")
    infl = _gender_form(gender,
        "Квадратная челюсть — главный визуальный якорь мужского лица.",
        "Контур подбородка определяет силуэт нижней части и общее впечатление.")
    return title, p1, p2, p3, infl


def text_nose_to_mouth(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]
    nwr = extra["nose_w_ratio"]
    mwr = extra["mouth_w_ratio"]

    if m["score"] >= 8.5:
        title = "Соотношение носа и рта гармонично — идеальный баланс центральной зоны."
    elif z < 0:
        title = "Нос узкий относительно ширины рта."
    else:
        title = "Нос шире рта относительно нормы."

    cause = ""
    if z < -0.3:
        cause = f" Основная причина: {'широкий рот (B={mwr:.4f})' if mwr > 0.41 else 'узкий нос (A={nwr:.4f})'}."
    elif z > 0.3:
        cause = f" Основная причина: {'широкий нос (A={nwr:.4f})' if nwr > 0.24 else 'компактный рот (B={mwr:.4f})'}."

    p1 = (f"Метрика сравнивает ширину носа с шириной рта. "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).{cause}")
    p2 = ("Сбалансированное соотношение создаёт гармонию центральной и нижней зон. "
          "Отклонения формируют характерные индивидуальные черты, которые часто "
          "становятся 'фишкой' лица.")
    p3 = ("Perrett et al. (1998): соотношение носа и рта влияет на воспринимаемый "
          "уровень маскулинности/фемининности.")
    infl = "Соотношение носа и рта определяет баланс центра лица — отклонения формируют характерные индивидуальные черты."
    return title, p1, p2, p3, infl


def text_biocular(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = "Биокулярная ширина пропорциональна лицу — открытый, гармоничный взгляд."
    elif z > 0:
        title = "Расширенная биокулярная зона — выразительный, открытый взгляд."
    else:
        title = "Зауженная биокулярная зона — сосредоточенный, концентрированный взгляд."

    p1 = (f"Метрика — расстояние между внешними углами глаз / ширина лица. "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).")
    p2 = ("Farkas (1994) установил, что биокулярная ширина около 71% ширины лица "
          "считается оптимальной. "
          f"{'Расширенная зона создаёт открытый взгляд.' if z>0 else 'Зауженная зона — более сфокусированный взгляд.'}")
    p3 = ("Farkas (Anthropometry of the Head and Face, 1994): "
          "биокулярная ширина — базовый параметр пропорций верхней трети лица.")
    infl = "Биокулярная ширина обеспечивает визуальный баланс верхней части лица."
    return title, p1, p2, p3, infl


def text_forehead(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = "Ширина лба пропорциональна лицу — гармоничная верхняя треть."
    elif z > 0:
        title = "Лоб шире среднего — выраженная верхняя зона."
    else:
        title = "Лоб уже среднего — компактная верхняя зона."

    p1 = (f"Метрика — ширина лба (между краями бровей) / ширина лица. "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).")
    p2 = (f"{'Широкий лоб ассоциируется с интеллектуальностью и открытостью.' if z>0.3 else 'Узкий лоб визуально утончает верхнюю часть лица.' if z<-0.3 else 'Пропорциональный лоб обеспечивает гармоничный баланс верхней трети.'}")
    p3 = ("Zebrowitz & Montepare (Social psychological face perception, 2008): "
          "ширина лба влияет на восприятие интеллекта и компетентности.")
    infl = "Ширина лба формирует впечатление верхней части лица и влияет на считываемые когнитивные характеристики."
    return title, p1, p2, p3, infl


def text_lips_fullness(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = "Полнота губ гармонична — идеальный контур рта."
    elif z > 0:
        title = "Губы полнее среднего — выразительный контур."
    else:
        title = _gender_form(gender,
            "Губы умеренно тонкие — типично для мужского лица.",
            "Губы тоньше среднего — деликатный контур.")

    p1 = (f"Метрика — высота губ / ширина рта. "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).")
    p2 = _gender_form(gender,
        (f"{'Полные губы создают акцент в нижней части и придают образу чувственность.' if z>0 else 'Умеренно тонкие губы — нейтральная характеристика для мужского лица.'}"),
        (f"{'Полные губы — ключевой маркер женской привлекательности, ассоциирующийся с молодостью.' if z>0 else 'Тонкие губы создают более деликатный и утончённый контур.'}"))
    p3 = ("Cunningham (1986): полнота губ значима для женской привлекательности "
          "и менее значима для мужской.")
    infl = _gender_form(gender,
        "Полнота губ слабо влияет на мужскую привлекательность — приоритет за чёткостью контура.",
        "Полнота губ — один из ключевых параметров женской привлекательности.")
    return title, p1, p2, p3, infl


def text_lips_proportions(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = "Пропорции губ близки к классическому соотношению 1:1.6."
    elif z > 0:
        title = "Верхняя губа непропорционально доминирует над нижней."
    else:
        title = "Нижняя губа существенно полнее верхней."

    p1 = (f"Метрика — высота верхней губы / высота нижней губы. "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).")
    p2 = ("Классическое соотношение губ 1:1.6 (верхняя к нижней) — "
          "одно из проявлений золотого сечения в чертах лица. "
          f"{'Ваш рот близок к этому эталону.' if abs(z)<0.5 else 'Индивидуальные пропорции часто формируют узнаваемую черту лица.'}")
    p3 = ("Perrett (In Your Face: The New Science of Human Attractiveness, 2010): "
          "соотношение губ 1:1.6 считается наиболее привлекательным.")
    infl = "Соотношение губ формирует индивидуальный контур рта — отклонения создают узнаваемые черты."
    return title, p1, p2, p3, infl


def text_jaw_to_mouth(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]
    jwr = extra["jaw_w_ratio"]
    mwr = extra["mouth_w_ratio"]

    if m["score"] >= 8.5:
        title = "Соотношение челюсти и рта сбалансировано — гармоничный нижний контур."
    elif z < 0:
        title = "Челюсть узкая относительно ширины рта."
    else:
        title = "Челюсть существенно шире рта — выраженный нижний контур."

    cause = ""
    if z < -0.3:
        cause = f" {'Причина: широкий рот (B={mwr:.3f}).' if mwr>0.41 else f'Причина: узкая челюсть (A={jwr:.3f}).'}"

    p1 = (f"Метрика — ширина челюсти / ширина рта. "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).{cause}")
    p2 = ("Широкая челюсть относительно рта создаёт мощный нижний контур — "
          "у мужчин ассоциируется с доминантностью. "
          "Отклонения в этой метрике часто компенсируются другими сильными чертами.")
    p3 = ("Rhodes et al. (Facial symmetry and the perception of beauty, 1998): "
          "общее впечатление определяется совокупностью черт, а не отдельными пропорциями.")
    infl = "Соотношение челюсти и рта формирует силуэт нижней части — отклонения часто компенсируются сильными чертами."
    return title, p1, p2, p3, infl


def text_eye_shape(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = "Форма глаз близка к идеальной миндалевидной."
    elif z > 0:
        title = "Глаза открытые, округлые — выразительный взгляд."
    else:
        title = "Глаза вытянутые, миндалевидные — экзотический взгляд."

    p1 = (f"Метрика — высота глаза / ширина глаза. "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).")
    p2 = (f"{'Открытые глаза создают запоминающийся, эмоционально насыщенный взгляд.' if z>0.3 else 'Миндалевидная форма — классический эстетический эталон, ассоциирующийся с экзотичностью.' if z<-0.3 else 'Сбалансированная форма глаз — универсально гармоничная характеристика.'}")
    p3 = ("Grammer et al. (2003): открытая форма глаз ассоциируется с молодостью и здоровьем.")
    infl = "Форма глаз — один из главных факторов выразительности взгляда и эмоциональности лица."
    return title, p1, p2, p3, infl


def text_brow_height(m, extra, gender):
    z, z_abs = m["z"], abs(m["z"])
    val, norm = m["value"], m["norm"]

    if m["score"] >= 8.5:
        title = "Высота бровей пропорциональна глазной зоне."
    elif z > 0:
        title = "Брови посажены высоко — увеличенное пространство до века."
    else:
        title = _gender_form(gender,
            "Брови посажены низко — сосредоточенный, доминантный взгляд.",
            "Брови посажены низко — глубокий, выразительный взгляд.")

    p1 = (f"Метрика — расстояние от нижнего края брови до века / ширина глаза. "
          f"Твой результат — {val:.4f} при норме {norm:.4f} ({z_abs:.2f}σ).")
    p2 = _gender_form(gender,
        (f"{'Низко посаженные брови ассоциируются с доминантностью и усиливают мужественное впечатление.' if z<-0.3 else 'Высоко посаженные брови — индивидуальная особенность; у мужчин иногда воспринимается как удивлённость.' if z>0.3 else 'Пропорциональная высота бровей формирует гармоничную глазную зону.'}"),
        (f"{'Низко посаженные брови создают глубокий, выразительный взгляд.' if z<-0.3 else 'Высоко посаженные брови — классическая черта женской миловидности.' if z>0.3 else 'Пропорциональная высота бровей — гармоничный элемент глазной зоны.'}"))
    p3 = ("Zebrowitz (Reading Faces, 1997): расстояние бровь-веко влияет на "
          "восприятие эмоционального состояния и характера.")
    infl = "Высота бровей формирует выражение взгляда и одну из ключевых эмоциональных характеристик лица."
    return title, p1, p2, p3, infl


TEXT_GENERATORS = {
    "Симметрия лица":          text_symmetry,
    "Пропорции лица":          text_proportions,
    "Вертикальный баланс":     text_vertical_balance,
    "Баланс скул и челюсти":   text_cheek_jaw,
    "Размер глаз":             text_eye_size,
    "Расстояние между глазами":text_eye_distance,
    "Наклон глаз":             text_eye_tilt,
    "Ширина носа":             text_nose_width,
    "Ширина рта":              text_mouth_width,
    "Длина носа":              text_nose_length,
    "Длина подбородка":        text_chin_length,
    "Контур подбородка":       text_chin_contour,
    "Нос к ширине рта":        text_nose_to_mouth,
    "Биокулярная ширина":      text_biocular,
    "Ширина лба":              text_forehead,
    "Полнота губ":             text_lips_fullness,
    "Пропорции губ":           text_lips_proportions,
    "Челюсть к ширине рта":    text_jaw_to_mouth,
    "Форма глаз":              text_eye_shape,
    "Высота бровей":           text_brow_height,
}

def generate_metric_text(metric, extra, gender):
    gen = TEXT_GENERATORS.get(metric["name"])
    if gen:
        return gen(metric, extra, gender)
    return ("Метрика лица.", "", "", "", "Метрика влияет на общий баланс лица.")


# ================== RECOMMENDATIONS ==================
RECS = {
    "Симметрия лица":          {"male": "Симметричная стрижка и ровная посадка бровей визуально выровняют черты.", "female": "Симметричный макияж и укладка визуально сбалансируют лицо."},
    "Пропорции лица":          {"male": "Стрижка с нужным объёмом по бокам или в высоту визуально скорректирует овал.", "female": "Стрижка с объёмом в нужных зонах визуально подкорректирует форму лица."},
    "Вертикальный баланс":     {"male": "Щетина визуально удлинит или сбалансирует нижнюю треть лица.", "female": "Контурирование скул поможет визуально подкорректировать вертикальные пропорции."},
    "Баланс скул и челюсти":   {"male": "Щетина по линии челюсти визуально расширит её и сбалансирует со скулами.", "female": "Контурирование скул и челюсти подчеркнёт овал."},
    "Размер глаз":             {"male": "Аккуратные брови и уход за кожей вокруг глаз подчеркнут их размер.", "female": "Макияж с акцентом на ресницы и стрелки визуально увеличит глаза."},
    "Расстояние между глазами":{"male": "Коррекция формы бровей визуально скорректирует межглазное расстояние.", "female": "Светлые тени во внутреннем углу приближают глаза, тёмные — отдаляют."},
    "Наклон глаз":             {"male": "Коррекция хвостика брови визуально приподнимет уголки глаз.", "female": "Стрелки с подъёмом во внешнем уголке создадут эффект fox eyes."},
    "Ширина носа":             {"male": "Контурирование боковых частей носа визуально утончит его.", "female": "Лёгкое контурирование переносицы и крыльев скорректирует ширину."},
    "Ширина рта":              {"male": "Щетина по контуру визуально сбалансирует ширину рта.", "female": "Чёткий контур губ карандашом визуально скорректирует ширину рта."},
    "Длина носа":              {"male": "Контурирование кончика носа визуально подкорректирует длину.", "female": "Контурирование основания носа визуально укоротит его."},
    "Длина подбородка":        {"male": "Щетина в области подбородка визуально удлинит или сбалансирует нижнюю треть.", "female": "Контурирование подбородка скорректирует нижнюю часть лица."},
    "Контур подбородка":       {"male": "Щетина по линии челюсти усилит контур.", "female": "Контурирование угла челюсти сделает контур более выраженным."},
    "Нос к ширине рта":        {"male": "Щетина или борода визуально смягчит контраст между носом и ртом.", "female": "Чёткий контур губ и лёгкое контурирование носа сбалансируют центральную зону."},
    "Биокулярная ширина":      {"male": "Коррекция бровей визуально сбалансирует биокулярную зону.", "female": "Грамотный макияж бровей и глаз гармонизирует верхнюю часть лица."},
    "Ширина лба":              {"male": "Стрижка с правильной чёлкой визуально скорректирует ширину лба.", "female": "Чёлка или укладка с обрамлением лба подкорректируют его пропорции."},
    "Полнота губ":             {"male": "Увлажняющий бальзам поддержит чёткий контур губ.", "female": "Прозрачный блеск или увлажняющая помада добавят губам объёма."},
    "Пропорции губ":           {"male": "Уход за губами поможет визуально выровнять пропорции.", "female": "Контур карандашом скорректирует соотношение верхней и нижней губы."},
    "Челюсть к ширине рта":    {"male": "Щетина 3–5 мм визуально расширит челюсть и улучшит её соотношение с ртом.", "female": "Контурирование угла челюсти визуально расширит её."},
    "Форма глаз":              {"male": "Аккуратные брови и уход подчеркнут форму глаз.", "female": "Подводка и тени подчеркнут миндалевидную или округлую форму."},
    "Высота бровей":           {"male": "Коррекция формы бровей скорректирует расстояние до век.", "female": "Профессиональная коррекция бровей сбалансирует высоту."},
}

STRENGTH_DESCS = {
    "Симметрия лица":          "Высокая симметрия — лицо практически зеркально по обеим сторонам, маркер генетического здоровья.",
    "Пропорции лица":          "Гармоничные пропорции создают универсально привлекательный овал.",
    "Вертикальный баланс":     "Сбалансированные трети лица — классическая гармония по Леонардо да Винчи.",
    "Баланс скул и челюсти":   "Соотношение скул и челюсти близко к идеальному — архитектурный контур.",
    "Размер глаз":             "Пропорциональный размер глаз делает взгляд естественно выразительным.",
    "Расстояние между глазами":"Оптимальное межглазное расстояние — гармоничная глазная зона.",
    "Наклон глаз":             "Приподнятые уголки глаз — один из наиболее ценимых эстетических признаков.",
    "Ширина носа":             "Пропорциональный нос гармонично вписывается в общую картину лица.",
    "Ширина рта":              "Сбалансированная ширина рта формирует гармоничный нижний контур.",
    "Длина носа":              "Пропорциональная длина носа поддерживает классические пропорции лица.",
    "Длина подбородка":        "Выраженный подбородок завершает силуэт лица и формирует впечатление силы.",
    "Контур подбородка":       "Чёткий контур челюсти создаёт архитектурный нижний силуэт.",
    "Нос к ширине рта":        "Сбалансированное соотношение носа и рта — гармония центральной зоны.",
    "Биокулярная ширина":      "Пропорциональная биокулярная ширина — баланс верхней части лица.",
    "Ширина лба":              "Пропорциональный лоб формирует гармоничную верхнюю треть.",
    "Полнота губ":             "Гармоничная полнота губ создаёт привлекательный контур рта.",
    "Пропорции губ":           "Классическое соотношение губ — гармоничный контур рта.",
    "Челюсть к ширине рта":    "Сбалансированное соотношение челюсти и рта — мощный нижний контур.",
    "Форма глаз":              "Выразительная форма глаз делает взгляд запоминающимся.",
    "Высота бровей":           "Пропорциональная высота бровей формирует гармоничную глазную зону.",
}

def generate_recommendations(data, gender):
    strengths = [(m["name"], STRENGTH_DESCS.get(m["name"], "Сильная сторона лица."))
                 for m in data["strengths"]]
    improvements = [(m["name"], RECS.get(m["name"], {}).get(
                     "female" if gender=="female" else "male",
                     "Лёгкая визуальная коррекция поможет улучшить эту зону."))
                    for m in data["weak"]]
    extra_tips = _gender_form(gender,
        [("Уход за кожей", "Регулярное очищение и увлажнение придадут лицу свежесть."),
         ("Режим сна", "7–8 часов сна уменьшат отёчность и придадут лицу подтянутость.")],
        [("Уход за кожей", "Очищение, увлажнение и SPF поддержат здоровый вид кожи."),
         ("Брови и ресницы", "Профессиональная коррекция бровей усилит выразительность взгляда.")])
    improvements.extend(extra_tips)
    return strengths, improvements


# ================== OVERLAY ==================
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def base_image_for_overlay(image_bytes, max_size=900):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((max_size, max_size))
    return img

def lm_pt(lm, name, w, h):
    idx = IDX[name]
    return int(lm[idx].x*w), int(lm[idx].y*h)

def draw_pt(draw, p, color, r=5):
    draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r), fill=color)

def draw_ln(draw, p1, p2, color, width=3):
    draw.line([p1,p2], fill=color, width=width)

def overlay_for_metric(image_bytes, metric_name, color_hex):
    img = base_image_for_overlay(image_bytes)
    w, h = img.size
    draw = ImageDraw.Draw(img)
    color = hex_to_rgb(color_hex)
    accent = hex_to_rgb(COLOR_ACCENT)

    arr = np.array(img)
    rgb = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return img
    lm = res.multi_face_landmarks[0].landmark

    def P(name): return lm_pt(lm, name, w, h)

    draws = {
        "Симметрия лица":          lambda: [draw_ln(draw,P("forehead"),P("chin"),color,3)] + [draw_pt(draw,P(n),accent) for n in ["left_eye_inner","right_eye_inner","left_eye_outer","right_eye_outer","mouth_left","mouth_right","nose_left","nose_right"]],
        "Пропорции лица":          lambda: [draw_ln(draw,P("nose_bridge"),P("chin"),color,3), draw_ln(draw,P("face_left"),P("face_right"),color,3)],
        "Вертикальный баланс":     lambda: [draw_ln(draw,P("nose_bridge"),P("nose_base"),color,3), draw_ln(draw,P("nose_base"),P("chin"),color,3)] + [draw_pt(draw,P(n),accent) for n in ["nose_bridge","nose_base","chin"]],
        "Баланс скул и челюсти":   lambda: [draw_ln(draw,P("cheek_left"),P("cheek_right"),color,3), draw_ln(draw,P("jaw_left"),P("jaw_right"),color,3)],
        "Размер глаз":             lambda: [draw_ln(draw,P("left_eye_outer"),P("left_eye_inner"),color,3), draw_ln(draw,P("right_eye_inner"),P("right_eye_outer"),color,3)],
        "Расстояние между глазами":lambda: [draw_ln(draw,P("left_eye_inner"),P("right_eye_inner"),color,3)] + [draw_pt(draw,P(n),accent) for n in ["left_eye_inner","right_eye_inner"]],
        "Наклон глаз":             lambda: [draw_ln(draw,P("left_eye_inner"),P("left_eye_outer"),color,3), draw_ln(draw,P("right_eye_inner"),P("right_eye_outer"),color,3)],
        "Ширина носа":             lambda: [draw_ln(draw,P("nose_left"),P("nose_right"),color,3)] + [draw_pt(draw,P(n),accent) for n in ["nose_left","nose_right"]],
        "Ширина рта":              lambda: [draw_ln(draw,P("mouth_left"),P("mouth_right"),color,3)],
        "Длина носа":              lambda: [draw_ln(draw,P("nose_bridge"),P("nose_base"),color,3)],
        "Длина подбородка":        lambda: [draw_ln(draw,P("lower_lip"),P("chin"),color,3)] + [draw_pt(draw,P(n),accent) for n in ["lower_lip","chin"]],
        "Контур подбородка":       lambda: [draw_ln(draw,P("jaw_left"),P("chin"),color,3), draw_ln(draw,P("jaw_right"),P("chin"),color,3), draw_ln(draw,P("jaw_left_lower"),P("jaw_right_lower"),color,3)],
        "Нос к ширине рта":        lambda: [draw_ln(draw,P("nose_left"),P("nose_right"),color,3), draw_ln(draw,P("mouth_left"),P("mouth_right"),color,3)],
        "Биокулярная ширина":      lambda: [draw_ln(draw,P("left_eye_outer"),P("right_eye_outer"),color,3)],
        "Ширина лба":              lambda: [draw_ln(draw,P("forehead_left"),P("forehead_right"),color,3)],
        "Полнота губ":             lambda: [draw_ln(draw,P("upper_lip_top"),P("lower_lip_bottom"),color,3)],
        "Пропорции губ":           lambda: [draw_ln(draw,P("upper_lip_top"),P("upper_lip"),color,3), draw_ln(draw,P("lower_lip"),P("lower_lip_bottom"),color,3)],
        "Челюсть к ширине рта":    lambda: [draw_ln(draw,P("jaw_left"),P("jaw_right"),color,3), draw_ln(draw,P("mouth_left"),P("mouth_right"),color,3)],
        "Форма глаз":              lambda: [draw_ln(draw,P("left_eye_top"),P("left_eye_bottom"),color,3), draw_ln(draw,P("right_eye_top"),P("right_eye_bottom"),color,3), draw_ln(draw,P("left_eye_outer"),P("left_eye_inner"),color,2), draw_ln(draw,P("right_eye_inner"),P("right_eye_outer"),color,2)],
        "Высота бровей":           lambda: [draw_ln(draw,P("left_brow_mid"),P("left_eye_top"),color,3), draw_ln(draw,P("right_brow_mid"),P("right_eye_top"),color,3)],
    }
    fn = draws.get(metric_name)
    if fn: fn()
    return img


# ================== CHARTS ==================
def create_radar_chart(metrics, output_path):
    labels = [m["name"] for m in metrics]
    scores = [m["score"] for m in metrics]
    n = len(labels)
    angles = np.linspace(0,2*np.pi,n,endpoint=False).tolist()
    sp = scores+[scores[0]]; ap = angles+[angles[0]]

    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLOR_BG); ax.set_facecolor(COLOR_BG)
    ax.plot(ap, sp, color=COLOR_ACCENT, linewidth=2)
    ax.fill(ap, sp, color=COLOR_ACCENT, alpha=0.25)
    for i,(angle,score,c) in enumerate(zip(angles,scores,METRIC_COLORS)):
        ax.scatter([angle],[score],color=c,s=60,zorder=5,edgecolors=COLOR_BG,linewidths=1.5)
    ax.set_xticks(angles); ax.set_xticklabels(labels,color=COLOR_TEXT_SOFT,size=8)
    ax.set_ylim(0,10); ax.set_yticks([2,4,6,8,10])
    ax.set_yticklabels(["2","4","6","8","10"],color=COLOR_TEXT_MUTED,size=7)
    ax.tick_params(axis="x",pad=12)
    ax.grid(color=COLOR_LINE,linewidth=0.6,alpha=0.7)
    ax.spines["polar"].set_color(COLOR_LINE)
    plt.tight_layout()
    plt.savefig(output_path,facecolor=COLOR_BG,dpi=150,bbox_inches="tight"); plt.close(fig)

def create_distribution_chart(score, output_path):
    fig, ax = plt.subplots(figsize=(6,2))
    fig.patch.set_facecolor(COLOR_BG); ax.set_facecolor(COLOR_BG)
    x = np.linspace(0,10,200)
    y = np.exp(-((x-6.5)**2)/(2*1.4**2))
    ax.fill_between(x,y,color=COLOR_BAR_BG,alpha=0.8)
    ax.plot(x,y,color=COLOR_LINE,linewidth=1)
    ax.axvline(score,color=COLOR_ACCENT,linewidth=2.5)
    ax.set_xlim(0,10); ax.set_ylim(0,1.1); ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path,facecolor=COLOR_BG,dpi=150,bbox_inches="tight",pad_inches=0.1); plt.close(fig)


# ================== PDF HELPERS ==================
def draw_page_bg(c, w, h):
    c.setFillColor(HexColor(COLOR_BG)); c.rect(0,0,w,h,fill=1,stroke=0)

def draw_footer(c, w, page_num, total=23):
    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_REGULAR,8)
    c.drawCentredString(w/2, 22, f"Telegram: {BOT_USERNAME}  ·  {page_num} / {total}")

def draw_wrapped(c, text, x, y, max_chars=82, lh=14, font=None, size=10, color=COLOR_TEXT):
    if font is None: font = FONT_REGULAR
    c.setFillColor(HexColor(color)); c.setFont(font, size)
    for line in wrap_text(text, max_chars):
        c.drawString(x, y, line); y -= lh
    return y

def draw_progress_bar(c, x, y, w, score, color_hex, h=10):
    c.setFillColor(HexColor(COLOR_BAR_BG)); c.roundRect(x,y,w,h,h/2,fill=1,stroke=0)
    fw = w*(score/10)
    if fw>1:
        c.setFillColor(HexColor(color_hex)); c.roundRect(x,y,fw,h,h/2,fill=1,stroke=0)

def draw_dots(c, x, y, filled, color_hex, total=5, size=5, gap=12):
    for i in range(total):
        c.setFillColor(HexColor(color_hex if i<filled else COLOR_BAR_BG))
        c.circle(x+i*gap, y, size, fill=1, stroke=0)

def draw_left_accent(c, x, y_top, y_bot, color_hex, w=3):
    c.setFillColor(HexColor(color_hex)); c.rect(x,y_bot,w,y_top-y_bot,fill=1,stroke=0)

def draw_hline(c, x1, x2, y, color=COLOR_LINE, w=0.5):
    c.setStrokeColor(HexColor(color)); c.setLineWidth(w); c.line(x1,y,x2,y)


# ================== PDF: COVER ==================
def draw_cover(c, W, H, image_bytes, data):
    draw_page_bg(c, W, H)

    # Title
    c.setFillColor(HexColor(COLOR_TITLE)); c.setFont(FONT_BOLD, 44)
    c.drawCentredString(W/2, H-95, BOT_NAME)
    c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR, 11)
    c.drawCentredString(W/2, H-117, f"Telegram: {BOT_USERNAME}")
    c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_REGULAR, 12)
    c.drawCentredString(W/2, H-137, "Математический разбор пропорций лица.")
    draw_hline(c, 60, W-60, H-158)

    # Photo
    cover_img = base_image_for_overlay(image_bytes)
    cover_img.thumbnail((260,300))
    img_w, img_h = cover_img.size
    img_x = (W-img_w)/2
    img_y = H-175-img_h
    c.drawImage(ImageReader(cover_img), img_x, img_y, width=img_w, height=img_h,
                preserveAspectRatio=True, mask='auto')

    # Score
    score_y = img_y - 25
    c.setFillColor(HexColor(COLOR_ACCENT)); c.setFont(FONT_BOLD, 58)
    c.drawCentredString(W/2, score_y-52, f"{data['score']:.2f}")
    c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR, 13)
    c.drawCentredString(W/2, score_y-72, "из 10")

    # Progress bar
    bar_w = 280; bar_x = (W-bar_w)/2; bar_y = score_y-100
    draw_progress_bar(c, bar_x, bar_y, bar_w, data["score"], data["tier_color"], h=12)

    # TIER — главная строка
    tier_short = data["tier_short"]
    tier_full  = data["tier_full"]
    tier_color = data["tier_color"]
    c.setFillColor(HexColor(tier_color)); c.setFont(FONT_BOLD, 16)
    tier_line = f"{tier_short}  ·  {tier_full}"
    c.drawCentredString(W/2, bar_y-22, tier_line)

    # Top percent
    c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR, 11)
    c.drawCentredString(W/2, bar_y-40, f"Ты в топ {data['top_percent']}% по геометрии лица!")

    # Distribution curve
    dist_path = tempfile.NamedTemporaryFile(delete=False,suffix=".png").name
    create_distribution_chart(data["score"], dist_path)
    c.drawImage(ImageReader(dist_path), W/2-130, bar_y-125,
                width=260, height=80, preserveAspectRatio=True, mask='auto')

    c.setFillColor(HexColor(COLOR_TITLE)); c.setFont(FONT_BOLD, 13)
    c.drawCentredString(W/2, bar_y-142, f"Уровень: {data['level']}")

    s_text = "Сильные стороны: " + ", ".join(m["name"].lower() for m in data["strengths"])
    c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR, 10)
    c.drawCentredString(W/2, bar_y-158, s_text)

    # General impression block
    impr_y = 180
    draw_left_accent(c, 60, impr_y+18, impr_y-72, COLOR_ACCENT)
    c.setFillColor(HexColor(COLOR_TITLE)); c.setFont(FONT_BOLD, 11)
    c.drawString(80, impr_y, "ОБЩЕЕ ВПЕЧАТЛЕНИЕ")
    impression = (
        "Лицо проанализировано по ключевым геометрическим точкам с расчётом 20 "
        "антропометрических метрик. Каждая метрика сравнивается с медианными значениями "
        "и сигма-отклонением для твоего пола. Итоговая оценка отражает совокупную "
        "близость пропорций к статистическим нормам гармонии."
    )
    draw_wrapped(c, impression, 80, impr_y-18, max_chars=86, lh=14, size=10, color=COLOR_TEXT)
    draw_footer(c, W, 1)


# ================== PDF: PROFILE ==================
def draw_profile_page(c, W, H, data):
    draw_page_bg(c, W, H)
    c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD, 28)
    c.drawString(60, H-80, "Профиль метрик")
    draw_hline(c, 60, W-60, H-100)

    radar_path = tempfile.NamedTemporaryFile(delete=False,suffix=".png").name
    create_radar_chart(data["metrics"], radar_path)
    c.drawImage(ImageReader(radar_path), W/2-180, H-470,
                width=360, height=360, preserveAspectRatio=True, mask='auto')
    draw_hline(c, 60, W-60, H-490)

    c.setFillColor(HexColor(COLOR_ACCENT)); c.setFont(FONT_BOLD, 12)
    c.drawString(60, H-510, "Топ-3 сильных метрики")
    c.drawString(W/2+10, H-510, "Топ-3 зоны потенциала")

    y = H-530
    for m in data["strengths"]:
        ci = data["metrics"].index(m)
        c.setFillColor(HexColor(METRIC_COLORS[ci])); c.circle(70,y+4,3,fill=1,stroke=0)
        c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_REGULAR,10)
        c.drawString(82, y, f"{m['name']}  —  {m['score']:.2f}"); y-=18

    y = H-530
    for m in data["weak"]:
        ci = data["metrics"].index(m)
        c.setFillColor(HexColor(METRIC_COLORS[ci])); c.circle(W/2+20,y+4,3,fill=1,stroke=0)
        c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_REGULAR,10)
        c.drawString(W/2+32, y, f"{m['name']}  —  {m['score']:.2f}"); y-=18

    draw_hline(c, 60, W-60, H-600)
    c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD,13)
    c.drawString(60, H-620, "Вклад каждой метрики")

    col1_x=60; col2_x=W/2+10; col_w=(W/2)-80
    y_l=y_r=H-645
    for i,m in enumerate(data["metrics"]):
        color=METRIC_COLORS[i]
        if i<10:
            c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR,8.5)
            c.drawString(col1_x,y_l,m["name"])
            c.setFillColor(HexColor(COLOR_TEXT)); c.drawRightString(col1_x+col_w,y_l,f"{m['score']:.2f}")
            draw_progress_bar(c,col1_x,y_l-8,col_w,m["score"],color,h=6); y_l-=22
        else:
            c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR,8.5)
            c.drawString(col2_x,y_r,m["name"])
            c.setFillColor(HexColor(COLOR_TEXT)); c.drawRightString(col2_x+col_w,y_r,f"{m['score']:.2f}")
            draw_progress_bar(c,col2_x,y_r-8,col_w,m["score"],color,h=6); y_r-=22

    bot_y=min(y_l,y_r)-10
    c.setFillColor(HexColor(COLOR_ACCENT)); c.setFont(FONT_BOLD,10)
    c.drawString(60,bot_y,"КАК ЧИТАТЬ ОЦЕНКУ")
    legend=("Каждая метрика — безразмерное соотношение двух расстояний на лице. Норма — медиана "
            "по полу (Farkas et al.). Оценка: 10=совпадение с нормой, чем больше отклонение — "
            "тем ниже балл. σ (сигма): ±1σ = ~68% людей, ±2σ = ~95%. Низкий балл — "
            "статистическое отклонение, а не приговор.")
    draw_wrapped(c,legend,60,bot_y-18,max_chars=110,lh=12,size=8.5,color=COLOR_TEXT_SOFT)
    draw_footer(c,W,2)


# ================== PDF: METRIC PAGE ==================
def draw_metric_page(c, W, H, image_bytes, m, idx, extra, gender, page_num):
    draw_page_bg(c, W, H)
    color = METRIC_COLORS[idx-1]

    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_BOLD,38)
    c.drawString(60, H-95, f"{idx:02d}")
    c.setFillColor(HexColor(color)); c.setFont(FONT_BOLD,22)
    c.drawString(60, H-135, m["name"])
    draw_hline(c, 60, W-60, H-155)

    overlay_img = overlay_for_metric(image_bytes, m["name"], color)
    overlay_img.thumbnail((230,270))
    img_w,img_h = overlay_img.size
    img_x,img_y = 70, H-180-img_h
    c.drawImage(ImageReader(overlay_img), img_x, img_y,
                width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')

    rx=340; iy=H-195
    c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR,10)
    c.drawString(rx, iy, "Балл метрики")
    c.setFillColor(HexColor(color)); c.setFont(FONT_BOLD,38)
    c.drawString(rx, iy-42, f"{m['score']:.2f}")
    sw = c.stringWidth(f"{m['score']:.2f}", FONT_BOLD, 38)
    c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR,11)
    c.drawString(rx+sw+6, iy-42, "/ 10")
    draw_progress_bar(c, rx, iy-60, 190, m["score"], color, h=8)

    c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR,10)
    c.drawString(rx, iy-85, "Твой показатель:")
    c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD,14)
    c.drawString(rx, iy-102, f"{m['value']:.4f}")
    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_REGULAR,8)
    fy=iy-117
    for line in wrap_text(m["formula"],46)[:2]:
        c.drawString(rx,fy,line); fy-=10
    c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR,10)
    c.drawString(rx,fy-8,"Норма:")
    c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD,14)
    c.drawString(rx,fy-25,f"{m['norm']:.4f}")
    draw_dots(c,rx,fy-50,closeness_dots(m["score"]),color)
    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_REGULAR,9)
    c.drawString(rx+70,fy-53,"Близость к норме")

    title,p1,p2,p3,infl = generate_metric_text(m,extra,gender)
    ty = img_y-30
    draw_hline(c,60,W-60,ty+10)
    c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD,12)
    for line in wrap_text(title,78):
        c.drawString(60,ty,line); ty-=16
    ty-=8
    ty=draw_wrapped(c,p1,60,ty,max_chars=98,lh=13,size=9.5,color=COLOR_TEXT); ty-=10
    if p2: ty=draw_wrapped(c,p2,60,ty,max_chars=98,lh=13,size=9.5,color=COLOR_TEXT); ty-=10
    if p3: ty=draw_wrapped(c,p3,60,ty,max_chars=98,lh=13,size=9.5,color=COLOR_TEXT_SOFT); ty-=12

    if ty>110:
        draw_hline(c,60,W-60,ty+4); ty-=18
        draw_left_accent(c,60,ty+12,max(60,ty-50),color)
        c.setFillColor(HexColor(color)); c.setFont(FONT_BOLD,10)
        c.drawString(80,ty,"ВЛИЯНИЕ"); ty-=16
        draw_wrapped(c,infl,80,ty,max_chars=92,lh=13,size=9.5,color=COLOR_TEXT)
    draw_footer(c,W,page_num)


# ================== PDF: RECOMMENDATIONS ==================
def draw_recommendations(c, W, H, data, gender):
    draw_page_bg(c, W, H)
    c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD,28)
    c.drawString(60,H-80,"Рекомендации")
    c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR,11)
    c.drawString(60,H-100,"Персональные советы по улучшению")
    draw_hline(c,60,W-60,H-115)

    strengths,improvements = generate_recommendations(data,gender)

    c.setFillColor(HexColor(COLOR_ACCENT)); c.setFont(FONT_BOLD,13)
    c.drawString(60,H-140,"Что уже отлично")
    y=H-165
    for i,(name,text) in enumerate(strengths):
        color=METRIC_COLORS[i%len(METRIC_COLORS)]
        block_top=y+12
        c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD,10)
        c.drawString(85,y,name)
        ty=draw_wrapped(c,text,85,y-14,max_chars=92,lh=12,size=9,color=COLOR_TEXT)
        draw_left_accent(c,65,block_top,ty+4,color)
        y=ty-10

    y-=10; draw_hline(c,60,W-60,y); y-=20
    c.setFillColor(HexColor(COLOR_TITLE)); c.setFont(FONT_BOLD,13)
    c.drawString(60,y,"Что можно улучшить"); y-=25
    for i,(name,text) in enumerate(improvements):
        color=METRIC_COLORS[(i+3)%len(METRIC_COLORS)]
        block_top=y+12
        c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD,10)
        c.drawString(85,y,name)
        ty=draw_wrapped(c,text,85,y-14,max_chars=92,lh=12,size=9,color=COLOR_TEXT)
        draw_left_accent(c,65,block_top,ty+4,color)
        y=ty-10
        if y<70: break

    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_REGULAR,9)
    c.drawString(60,50,f"Дата разбора: {data['date']}")
    draw_footer(c,W,23)


# ================== PDF MAIN ==================
def create_pdf_report(image_bytes, data, gender, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    W, H = A4
    draw_cover(c,W,H,image_bytes,data); c.showPage()
    draw_profile_page(c,W,H,data); c.showPage()
    page_num=3
    for idx,m in enumerate(data["metrics"],start=1):
        draw_metric_page(c,W,H,image_bytes,m,idx,data["extra"],gender,page_num)
        page_num+=1; c.showPage()
    draw_recommendations(c,W,H,data,gender)
    c.save()


# ================== KEYBOARDS ==================
main_kb = ReplyKeyboardMarkup(keyboard=[
    [KeyboardButton(text="💠 Хочу получить свой разбор")],
    [KeyboardButton(text="Что это?"), KeyboardButton(text="Техподдержка")],
], resize_keyboard=True)

gender_kb = ReplyKeyboardMarkup(keyboard=[
    [KeyboardButton(text="👨 Мужской"), KeyboardButton(text="👩 Женский")],
    [KeyboardButton(text="◀️ Назад в меню")],
], resize_keyboard=True)

cancel_kb = ReplyKeyboardMarkup(keyboard=[
    [KeyboardButton(text="◀️ Назад в меню")],
], resize_keyboard=True)


# ================== HANDLERS ==================
@dp.message(F.text.in_({"/start","/help","◀️ Назад в меню"}))
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    total, today, user_count = get_counters(message.from_user.id)
    await message.answer(
        "📍 <b>Главное меню</b>\n\n"
        f"<b>{BOT_NAME}</b> математически измеряет, насколько гармонично "
        "черты твоего лица сочетаются друг с другом.\n\n"
        f"Твой баланс: <b>{user_count} {'разбор' if user_count==1 else 'разборов'}</b>\n"
        f"Сегодня пользователи провели: <b>{today} {'разбор' if today==1 else 'разборов'}</b>",
        parse_mode="HTML", reply_markup=main_kb)

@dp.message(F.text=="Что это?")
async def what_is_this(message: Message):
    await message.answer(
        "📍 <b>Что такое Heim Face</b>\n\n"
        f"<b>{BOT_NAME}</b> — сервис математической оценки гармонии пропорций лица.\n\n"
        "🔬 Алгоритм определяет ключевые точки лица, рассчитывает <b>20 антропометрических метрик</b> "
        "и сравнивает их с нормами по твоему полу.\n\n"
        "📄 <b>В PDF-отчёт входит:</b>\n"
        "• Итоговая оценка и <b>tier-уровень</b>\n"
        "• Профиль метрик с радар-чартом\n"
        "• 20 страниц с разбором каждой метрики\n"
        "• Визуализация измерений на твоём фото\n"
        "• Сильные стороны и зоны потенциала\n"
        "• Персональные рекомендации",
        parse_mode="HTML")

@dp.message(F.text=="Техподдержка")
async def support(message: Message):
    await message.answer(
        "📍 <b>Техподдержка</b>\n\nПо всем вопросам напишите администратору проекта.",
        parse_mode="HTML")

@dp.message(F.text=="💠 Хочу получить свой разбор")
async def get_report(message: Message, state: FSMContext):
    await state.set_state(AnalysisStates.waiting_for_gender)
    await message.answer(
        "📍 <b>Выбор тарифа</b>\n\n"
        f"⚜️ <b>Полный разбор</b> — 1 PDF-отчёт\n"
        f"💰 <b>Цена</b> — {PRICE_TEXT}\n\n"
        "📚 <b>Что входит:</b>\n"
        "• Персональный PDF-отчёт на 23 страницы\n"
        "• Разбор 20 ключевых метрик лица\n"
        "• Сравнение с нормативами для твоего пола\n"
        "• Tier-уровень (Sub3 → Gigachad / Goddess)\n"
        "• Визуализация измерений на фото\n"
        "• Рекомендации по улучшению\n\n"
        "💡 <b>Тестовый режим — оплата не подключена.</b>\n\n"
        "👤 <b>Выбери пол</b> для корректных антропометрических норм:",
        parse_mode="HTML", reply_markup=gender_kb)

@dp.message(AnalysisStates.waiting_for_gender, F.text.in_({"👨 Мужской","👩 Женский"}))
async def choose_gender(message: Message, state: FSMContext):
    gender = "male" if "Мужской" in message.text else "female"
    user_gender[message.from_user.id] = gender
    await state.set_state(AnalysisStates.waiting_for_photo)
    gw = "мужской" if gender=="male" else "женский"
    await message.answer(
        f"✅ Пол выбран: <b>{gw}</b>\n\n"
        "📸 <b>Отправь фото лица.</b>\n\n"
        "Требования:\n"
        "• Лицо строго анфас\n"
        "• Нейтральное выражение, рот закрыт\n"
        "• Хорошее равномерное освещение\n"
        "• Без очков, маски, головного убора\n"
        "• Волосы не закрывают лоб и брови\n"
        "• Чёткое фото, без размытия",
        parse_mode="HTML", reply_markup=cancel_kb)

@dp.message(AnalysisStates.waiting_for_gender)
async def wrong_gender(message: Message):
    await message.answer("Выбери пол кнопкой ниже 👇", reply_markup=gender_kb)

async def process_image(message: Message, image_bytes: bytes, state: FSMContext):
    gender = user_gender.get(message.from_user.id, "male")
    await message.answer("⏳ <b>Анализирую лицо...</b>\n\nЭто займёт до 30 секунд.", parse_mode="HTML")

    data, error = analyze_face(image_bytes, gender)
    if error:
        await message.answer(f"❌ {error}\n\nПопробуй другое фото.", reply_markup=cancel_kb)
        return

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_path = tmp.name
        create_pdf_report(image_bytes, data, gender, pdf_path)

        counters = increment_counter(message.from_user.id)

        tier_line = f"{data['tier_short']}  ·  {data['tier_full']}"
        await message.answer(
            "✅ <b>Разбор завершён!</b>\n\n"
            f"Оценка: <b>{data['score']:.2f} / 10</b>\n"
            f"Уровень: <b>{data['level']}</b>\n"
            f"Tier: <b>{tier_line}</b>\n\n"
            "Полный отчёт ниже ↓",
            parse_mode="HTML")

        await message.answer_document(
            FSInputFile(pdf_path, filename=f"Heim Face Report.pdf"),
            reply_markup=main_kb)
        await state.clear()
        Path(pdf_path).unlink(missing_ok=True)

    except Exception as e:
        logger.exception("PDF generation failed")
        await message.answer(
            f"❌ Ошибка при создании отчёта: {str(e)[:200]}\n\nПопробуй ещё раз.",
            reply_markup=cancel_kb)

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
        await message.answer("📎 Это не изображение. Отправь фото.")
        return
    file = await bot.get_file(doc.file_id)
    buf  = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    await process_image(message, buf.getvalue(), state)

@dp.message(AnalysisStates.waiting_for_photo)
async def wrong_state_photo(message: Message):
    await message.answer("📸 Жду фото лица.", reply_markup=cancel_kb)

@dp.message(F.photo)
async def handle_photo_no_state(message: Message):
    await message.answer(
        "👋 Сначала нажми <b>«💠 Хочу получить свой разбор»</b> и выбери пол.",
        parse_mode="HTML", reply_markup=main_kb)

@dp.message(F.document)
async def handle_doc_no_state(message: Message):
    await message.answer(
        "👋 Сначала нажми <b>«💠 Хочу получить свой разбор»</b> и выбери пол.",
        parse_mode="HTML", reply_markup=main_kb)

@dp.message()
async def fallback(message: Message):
    await message.answer(
        "📸 Нажми <b>«💠 Хочу получить свой разбор»</b> чтобы начать.",
        parse_mode="HTML", reply_markup=main_kb)


# ================== ENTRY POINT ==================
async def main():
    logger.info(f"{BOT_NAME} starting...")
    threading.Thread(target=run_web, daemon=True).start()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

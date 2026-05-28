import os

# ================== BRAND ==================
BOT_NAME = "Heim Face"
BOT_USERNAME = "@heim_face_bot"
TOTAL_PAGES = 23
ADMIN_IDS = [7108631309]

# ================== PRODUCTS ==================
PRODUCT_FACE_REPORT = "face_report"
PRODUCT_PREMIUM_PLAN = "premium_plan"

FACE_REPORT_PRICE_RUB = 990
PREMIUM_PLAN_PRICE_RUB = 1490

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        return default


FACE_REPORT_PRICE_STARS = _env_int("FACE_REPORT_PRICE_STARS", 1190)
PREMIUM_PLAN_PRICE_STARS = _env_int("PREMIUM_PLAN_PRICE_STARS", 1790)

# ================== PAYMENT PREP ==================
YOOKASSA_SHOP_ID = os.getenv("YOOKASSA_SHOP_ID", "").strip()
YOOKASSA_SECRET_KEY = os.getenv("YOOKASSA_SECRET_KEY", "").strip()
YOOKASSA_ENABLED = bool(YOOKASSA_SHOP_ID and YOOKASSA_SECRET_KEY)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
YOOKASSA_RETURN_URL = os.getenv(
    "YOOKASSA_RETURN_URL",
    PUBLIC_BASE_URL or f"https://t.me/{BOT_USERNAME.lstrip('@')}",
).strip()
YOOKASSA_WEBHOOK_PATH = "/yookassa/webhook"

TELEGRAM_STARS_CURRENCY = "XTR"

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
    "#D4AF37", "#E8A87C", "#D4A5C5", "#A687C9", "#C9A582",
    "#D4AF37", "#E8A87C", "#D4A5C5", "#A687C9", "#C9A582",
    "#D4AF37", "#E8A87C", "#D4A5C5", "#A687C9", "#C9A582",
    "#D4AF37", "#E8A87C", "#D4A5C5", "#A687C9", "#C9A582",
]


# ================== TIER SYSTEM ==================
# (low, high, abbreviation, full name, color)
TIERS_MALE = [
    (0.0,  3.0,  "Sub3",     "Sub3",                   "#8B0000"),
    (3.0,  4.5,  "Sub4",     "Sub4",                   "#A0522D"),
    (4.5,  5.5,  "Normie-",  "Lower Normie",           "#6B6B6B"),
    (5.5,  6.2,  "Normie",   "Normie",                 "#888888"),
    (6.2,  6.8,  "Normie+",  "Upper Normie",           "#999999"),
    (6.8,  7.3,  "HTN-",     "Lower High Tier Normie", "#4A9B8E"),
    (7.3,  7.8,  "HTN",      "High Tier Normie",       "#3AAFA0"),
    (7.8,  8.3,  "HTN+",     "Upper High Tier Normie", "#2DC5B4"),
    (8.3,  8.8,  "Chad-",    "Lower Chad",             "#C9A84C"),
    (8.8,  9.2,  "Chad",     "Chad",                   "#D4AF37"),
    (9.2,  9.6,  "Chad+",    "High Chad",              "#E8C84A"),
    (9.6, 10.1,  "Gigachad", "Gigachad",               "#FFD700"),
]

TIERS_FEMALE = [
    (0.0,  3.0,  "Sub3",       "Subpar",          "#8B0000"),
    (3.0,  4.5,  "Sub4",       "Below Average",   "#A0522D"),
    (4.5,  5.5,  "Average-",   "Lower Average",   "#6B6B6B"),
    (5.5,  6.2,  "Average",    "Average",         "#888888"),
    (6.2,  6.8,  "Average+",   "Upper Average",   "#999999"),
    (6.8,  7.3,  "Pretty-",    "Lower Pretty",    "#C87CA0"),
    (7.3,  7.8,  "Pretty",     "Pretty",          "#D4809A"),
    (7.8,  8.3,  "Pretty+",    "Upper Pretty",    "#E08FAF"),
    (8.3,  8.8,  "Beautiful-", "Lower Beautiful", "#C9A84C"),
    (8.8,  9.2,  "Beautiful",  "Beautiful",       "#D4AF37"),
    (9.2,  9.6,  "Beautiful+", "Upper Beautiful", "#E8C84A"),
    (9.6, 10.1,  "Goddess",    "Goddess",         "#FFD700"),
]


def get_tier(score: float, gender: str) -> dict:
    tiers = TIERS_MALE if gender == "male" else TIERS_FEMALE
    for lo, hi, abbr, name, color in tiers:
        if lo <= score < hi:
            return {"abbr": abbr, "name": name, "color": color}
    last = tiers[-1]
    return {"abbr": last[2], "name": last[3], "color": last[4]}


# ================== LANDMARK INDICES (MediaPipe FaceMesh) ==================
IDX = {
    "face_left": 234, "face_right": 454,
    "chin": 152, "forehead": 10,
    "nose_bridge": 168, "nose_base": 2, "nose_tip": 4,
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


# ================== NORMS (median + sigma per gender) ==================
NORMS_MALE = {
    "Симметрия лица":           {"norm": 0.970, "sigma": 0.055, "formula": "Зеркальность точек / центральная ось"},
    "Пропорции лица":           {"norm": 0.890, "sigma": 0.055, "formula": "Высота лица / ширина скул"},
    "Вертикальный баланс":      {"norm": 0.730, "sigma": 0.070, "formula": "Средняя треть / нижняя треть"},
    "Баланс скул и челюсти":    {"norm": 1.355, "sigma": 0.080, "formula": "Ширина скул / ширина челюсти"},
    "Размер глаз":              {"norm": 0.223, "sigma": 0.018, "formula": "Ширина глаза / ширина лица"},
    "Расстояние между глазами": {"norm": 0.268, "sigma": 0.020, "formula": "Расстояние между глазами / ширина лица"},
    "Наклон глаз":              {"norm": 0.040, "sigma": 0.030, "formula": "Наклон уголков / ширина глаза"},
    "Ширина носа":              {"norm": 0.233, "sigma": 0.018, "formula": "Ширина крыльев носа / ширина лица"},
    "Ширина рта":               {"norm": 0.402, "sigma": 0.030, "formula": "Ширина рта / ширина скул"},
    "Длина носа":               {"norm": 0.421, "sigma": 0.035, "formula": "Длина носа / высота лица"},
    "Длина подбородка":         {"norm": 0.286, "sigma": 0.030, "formula": "Нижняя губа - подбородок / высота лица"},
    "Контур подбородка":        {"norm": 0.632, "sigma": 0.045, "formula": "Угол сужения подбородка"},
    "Нос к ширине рта":         {"norm": 0.575, "sigma": 0.050, "formula": "Ширина носа / ширина рта"},
    "Биокулярная ширина":       {"norm": 0.711, "sigma": 0.045, "formula": "Внешние углы глаз / ширина лица"},
    "Ширина лба":               {"norm": 0.916, "sigma": 0.055, "formula": "Ширина лба / ширина лица"},
    "Полнота губ":              {"norm": 0.339, "sigma": 0.055, "formula": "Высота губ / ширина рта"},
    "Пропорции губ":            {"norm": 0.634, "sigma": 0.090, "formula": "Верхняя губа / нижняя губа"},
    "Челюсть к ширине рта":     {"norm": 1.841, "sigma": 0.140, "formula": "Ширина челюсти / ширина рта"},
    "Форма глаз":               {"norm": 0.350, "sigma": 0.045, "formula": "Высота глаза / ширина глаза"},
    "Высота бровей":            {"norm": 0.377, "sigma": 0.070, "formula": "Расстояние брови до века / ширина глаза"},
}

NORMS_FEMALE = {
    "Симметрия лица":           {"norm": 0.972, "sigma": 0.050, "formula": "Зеркальность точек / центральная ось"},
    "Пропорции лица":           {"norm": 0.920, "sigma": 0.055, "formula": "Высота лица / ширина скул"},
    "Вертикальный баланс":      {"norm": 0.760, "sigma": 0.070, "formula": "Средняя треть / нижняя треть"},
    "Баланс скул и челюсти":    {"norm": 1.420, "sigma": 0.080, "formula": "Ширина скул / ширина челюсти"},
    "Размер глаз":              {"norm": 0.232, "sigma": 0.018, "formula": "Ширина глаза / ширина лица"},
    "Расстояние между глазами": {"norm": 0.265, "sigma": 0.020, "formula": "Расстояние между глазами / ширина лица"},
    "Наклон глаз":              {"norm": 0.055, "sigma": 0.030, "formula": "Наклон уголков / ширина глаза"},
    "Ширина носа":              {"norm": 0.215, "sigma": 0.018, "formula": "Ширина крыльев носа / ширина лица"},
    "Ширина рта":               {"norm": 0.395, "sigma": 0.030, "formula": "Ширина рта / ширина скул"},
    "Длина носа":               {"norm": 0.405, "sigma": 0.035, "formula": "Длина носа / высота лица"},
    "Длина подбородка":         {"norm": 0.265, "sigma": 0.030, "formula": "Нижняя губа - подбородок / высота лица"},
    "Контур подбородка":        {"norm": 0.595, "sigma": 0.045, "formula": "Угол сужения подбородка"},
    "Нос к ширине рта":         {"norm": 0.545, "sigma": 0.050, "formula": "Ширина носа / ширина рта"},
    "Биокулярная ширина":       {"norm": 0.708, "sigma": 0.045, "formula": "Внешние углы глаз / ширина лица"},
    "Ширина лба":               {"norm": 0.905, "sigma": 0.055, "formula": "Ширина лба / ширина лица"},
    "Полнота губ":              {"norm": 0.395, "sigma": 0.055, "formula": "Высота губ / ширина рта"},
    "Пропорции губ":            {"norm": 0.665, "sigma": 0.090, "formula": "Верхняя губа / нижняя губа"},
    "Челюсть к ширине рта":     {"norm": 1.785, "sigma": 0.140, "formula": "Ширина челюсти / ширина рта"},
    "Форма глаз":               {"norm": 0.385, "sigma": 0.045, "formula": "Высота глаза / ширина глаза"},
    "Высота бровей":            {"norm": 0.420, "sigma": 0.070, "formula": "Расстояние брови до века / ширина глаза"},
}


def get_norms(gender):
    return NORMS_FEMALE if gender == "female" else NORMS_MALE

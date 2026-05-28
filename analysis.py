import math
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

from config import IDX, get_norms, get_tier

# ================== MEDIAPIPE ==================
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
)


# ================== HELPERS ==================
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calc_score(value, norm, sigma):
    z = abs(value - norm) / sigma
    return round(max(0, min(10, 10 - z * 2.2)), 2)


def calc_z(value, norm, sigma):
    return (value - norm) / sigma


def get_level(score):
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
    if score >= 6.0: return 55
    return 70


# ================== MAIN ANALYSIS ==================
def analyze_face(image_bytes: bytes, gender: str):
    """Возвращает (data_dict, None) при успехе или (None, error_str) при ошибке."""
    np_arr  = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, "Не удалось декодировать изображение."

    h, w    = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None, "Лицо не обнаружено. Отправь чёткое фото строго анфас при хорошем освещении."

    lm = results.multi_face_landmarks[0].landmark

    def pt(name):
        i = IDX[name]
        return lm[i].x * w, lm[i].y * h

    # --- базовые расстояния ---
    face_w      = dist(pt("face_left"),      pt("face_right"))
    face_h      = dist(pt("nose_bridge"),    pt("chin"))
    cheek_w     = dist(pt("cheek_left"),     pt("cheek_right"))
    jaw_w       = dist(pt("jaw_left"),       pt("jaw_right"))
    jaw_w_lower = dist(pt("jaw_left_lower"), pt("jaw_right_lower"))
    nose_w      = dist(pt("nose_left"),      pt("nose_right"))
    nose_len    = dist(pt("nose_bridge"),    pt("nose_base"))
    mouth_w     = dist(pt("mouth_left"),     pt("mouth_right"))

    left_eye_w  = dist(pt("left_eye_outer"),  pt("left_eye_inner"))
    right_eye_w = dist(pt("right_eye_inner"), pt("right_eye_outer"))
    eye_w       = (left_eye_w + right_eye_w) / 2

    left_eye_h  = dist(pt("left_eye_top"),  pt("left_eye_bottom"))
    right_eye_h = dist(pt("right_eye_top"), pt("right_eye_bottom"))
    eye_h       = (left_eye_h + right_eye_h) / 2

    eye_inner_dist = dist(pt("left_eye_inner"), pt("right_eye_inner"))
    biocular_w     = dist(pt("left_eye_outer"), pt("right_eye_outer"))
    forehead_w     = dist(pt("forehead_left"),  pt("forehead_right"))

    upper_lip_h  = dist(pt("upper_lip_top"), pt("upper_lip"))
    lower_lip_h  = dist(pt("lower_lip"),     pt("lower_lip_bottom"))
    lips_h       = upper_lip_h + lower_lip_h
    chin_len     = dist(pt("lower_lip"),     pt("chin"))
    middle_third = dist(pt("nose_bridge"),   pt("nose_base"))
    lower_third  = dist(pt("nose_base"),     pt("chin"))

    # --- симметрия ---
    midline_x = (pt("forehead")[0] + pt("chin")[0]) / 2

    def axis_dev(name):
        return (pt(name)[0] - midline_x) / face_w

    sym_inner = abs(abs(axis_dev("left_eye_inner")) - abs(axis_dev("right_eye_inner")))
    sym_outer = abs(abs(axis_dev("left_eye_outer")) - abs(axis_dev("right_eye_outer")))
    sym_mouth = abs(abs(axis_dev("mouth_left"))     - abs(axis_dev("mouth_right")))
    sym_nose  = abs(abs(axis_dev("nose_left"))      - abs(axis_dev("nose_right")))
    symmetry  = max(0, 1 - (sym_inner + sym_outer + sym_mouth + sym_nose) / 4 * 5)

    eye_tilt     = abs(pt("right_eye_outer")[1] - pt("left_eye_outer")[1]) / max(biocular_w, 1)
    chin_contour = jaw_w_lower / max(jaw_w, 1)
    brow_height  = (dist(pt("left_brow_mid"),  pt("left_eye_top")) +
                    dist(pt("right_brow_mid"), pt("right_eye_top"))) / 2 / max(eye_w, 1)

    values = {
        "Симметрия лица":           symmetry,
        "Пропорции лица":           face_h / face_w,
        "Вертикальный баланс":      middle_third / lower_third,
        "Баланс скул и челюсти":    cheek_w / jaw_w,
        "Размер глаз":              eye_w / face_w,
        "Расстояние между глазами": eye_inner_dist / face_w,
        "Наклон глаз":              eye_tilt,
        "Ширина носа":              nose_w / face_w,
        "Ширина рта":               mouth_w / cheek_w,
        "Длина носа":               nose_len / face_h,
        "Длина подбородка":         chin_len / face_h,
        "Контур подбородка":        chin_contour,
        "Нос к ширине рта":         nose_w / mouth_w,
        "Биокулярная ширина":       biocular_w / face_w,
        "Ширина лба":               forehead_w / face_w,
        "Полнота губ":              lips_h / mouth_w,
        "Пропорции губ":            upper_lip_h / max(lower_lip_h, 1),
        "Челюсть к ширине рта":     jaw_w / mouth_w,
        "Форма глаз":               eye_h / eye_w,
        "Высота бровей":            brow_height,
    }

    norms   = get_norms(gender)
    metrics = []
    for name, value in values.items():
        norm  = norms[name]["norm"]
        sigma = norms[name]["sigma"]
        metrics.append({
            "name":    name,
            "value":   round(value, 4),
            "norm":    norm,
            "sigma":   sigma,
            "score":   calc_score(value, norm, sigma),
            "z":       calc_z(value, norm, sigma),
            "formula": norms[name]["formula"],
        })

    total_score    = round(sum(m["score"] for m in metrics) / len(metrics), 2)
    sorted_metrics = sorted(metrics, key=lambda x: x["score"], reverse=True)

    extra = {
        "sym_inner_eye_pct": sym_inner * 100,
        "sym_outer_eye_pct": sym_outer * 100,
        "sym_mouth_pct":     sym_mouth * 100,
        "sym_nose_pct":      sym_nose  * 100,
        "nose_w_ratio":   nose_w  / face_w,
        "mouth_w_ratio":  mouth_w / face_w,
        "jaw_w_ratio":    jaw_w   / face_w,
        "upper_lip_ratio": upper_lip_h / max(face_h, 1),
        "lower_lip_ratio": lower_lip_h / max(face_h, 1),
    }

    return {
        "score":       total_score,
        "level":       get_level(total_score),
        "top_percent": get_top_percent(total_score),
        "tier":        get_tier(total_score, gender),
        "metrics":     metrics,
        "strengths":   sorted_metrics[:3],
        "weak":        sorted_metrics[-3:][::-1],
        "extra":       extra,
        "gender":      gender,
        "date":        datetime.now().strftime("%d.%m.%Y"),
    }, None

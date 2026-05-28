import io
import os
import tempfile

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from config import (
    BOT_NAME, BOT_USERNAME, TOTAL_PAGES, IDX,
    COLOR_BG, COLOR_TITLE, COLOR_ACCENT, COLOR_TEXT, COLOR_TEXT_SOFT,
    COLOR_TEXT_MUTED, COLOR_BAR_BG, COLOR_LINE, METRIC_COLORS,
)
from analysis import face_mesh
from texts import generate_metric_text, generate_recommendations


# ================== FONTS ==================
def setup_fonts():
    candidates = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
        ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
    ]
    for reg, bold in candidates:
        if os.path.exists(reg) and os.path.exists(bold):
            try:
                pdfmetrics.registerFont(TTFont("MainFont", reg))
                pdfmetrics.registerFont(TTFont("MainFontBold", bold))
                return "MainFont", "MainFontBold"
            except Exception:
                continue
    return "Helvetica", "Helvetica-Bold"

FONT_REGULAR, FONT_BOLD = setup_fonts()


# ================== TEXT HELPERS ==================
def wrap_text(text, max_chars=74):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur + " " + w) <= max_chars:
            cur += (" " + w) if cur else w
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def closeness_dots(score):
    if score >= 9:   return 5
    if score >= 7.5: return 4
    if score >= 6:   return 3
    if score >= 4:   return 2
    return 1


# ================== OVERLAY ==================
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def base_image_for_overlay(image_bytes, max_size=900):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((max_size, max_size))
    return img

def _lm_pt(lm, name, w, h):
    i = IDX[name]
    return int(lm[i].x * w), int(lm[i].y * h)

def _pt_img(d, p, color, r=5):
    x, y = p
    d.ellipse((x-r, y-r, x+r, y+r), fill=color)

def _line_img(d, p1, p2, color, width=3):
    d.line([p1, p2], fill=color, width=width)

def overlay_for_metric(image_bytes, metric_name, color_hex):
    img = base_image_for_overlay(image_bytes)
    w, h = img.size
    d = ImageDraw.Draw(img)
    color  = hex_to_rgb(color_hex)
    accent = hex_to_rgb(COLOR_ACCENT)

    rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return img
    lm = res.multi_face_landmarks[0].landmark

    def P(n): return _lm_pt(lm, n, w, h)

    draws = {
        "Симметрия лица": lambda: [
            _line_img(d, P("forehead"), P("chin"), color, 3),
            *[_pt_img(d, P(n), accent, 5) for n in
              ["left_eye_inner","right_eye_inner","left_eye_outer","right_eye_outer",
               "mouth_left","mouth_right","nose_left","nose_right"]]],
        "Пропорции лица": lambda: [
            _line_img(d, P("nose_bridge"), P("chin"), color, 3),
            _line_img(d, P("face_left"), P("face_right"), color, 3)],
        "Вертикальный баланс": lambda: [
            _line_img(d, P("nose_bridge"), P("nose_base"), color, 3),
            _line_img(d, P("nose_base"), P("chin"), color, 3),
            *[_pt_img(d, P(n), accent, 5) for n in ["nose_bridge","nose_base","chin"]]],
        "Баланс скул и челюсти": lambda: [
            _line_img(d, P("cheek_left"), P("cheek_right"), color, 3),
            _line_img(d, P("jaw_left"), P("jaw_right"), color, 3)],
        "Размер глаз": lambda: [
            _line_img(d, P("left_eye_outer"), P("left_eye_inner"), color, 3),
            _line_img(d, P("right_eye_inner"), P("right_eye_outer"), color, 3)],
        "Расстояние между глазами": lambda: [
            _line_img(d, P("left_eye_inner"), P("right_eye_inner"), color, 3),
            *[_pt_img(d, P(n), accent, 5) for n in ["left_eye_inner","right_eye_inner"]]],
        "Наклон глаз": lambda: [
            _line_img(d, P("left_eye_inner"), P("left_eye_outer"), color, 3),
            _line_img(d, P("right_eye_inner"), P("right_eye_outer"), color, 3)],
        "Ширина носа": lambda: [
            _line_img(d, P("nose_left"), P("nose_right"), color, 3),
            *[_pt_img(d, P(n), accent, 5) for n in ["nose_left","nose_right"]]],
        "Ширина рта": lambda: [_line_img(d, P("mouth_left"), P("mouth_right"), color, 3)],
        "Длина носа": lambda: [_line_img(d, P("nose_bridge"), P("nose_base"), color, 3)],
        "Длина подбородка": lambda: [
            _line_img(d, P("lower_lip"), P("chin"), color, 3),
            *[_pt_img(d, P(n), accent, 5) for n in ["lower_lip","chin"]]],
        "Контур подбородка": lambda: [
            _line_img(d, P("jaw_left"), P("chin"), color, 3),
            _line_img(d, P("jaw_right"), P("chin"), color, 3),
            _line_img(d, P("jaw_left_lower"), P("jaw_right_lower"), color, 3)],
        "Нос к ширине рта": lambda: [
            _line_img(d, P("nose_left"), P("nose_right"), color, 3),
            _line_img(d, P("mouth_left"), P("mouth_right"), color, 3)],
        "Биокулярная ширина": lambda: [_line_img(d, P("left_eye_outer"), P("right_eye_outer"), color, 3)],
        "Ширина лба": lambda: [_line_img(d, P("forehead_left"), P("forehead_right"), color, 3)],
        "Полнота губ": lambda: [_line_img(d, P("upper_lip_top"), P("lower_lip_bottom"), color, 3)],
        "Пропорции губ": lambda: [
            _line_img(d, P("upper_lip_top"), P("upper_lip"), color, 3),
            _line_img(d, P("lower_lip"), P("lower_lip_bottom"), color, 3)],
        "Челюсть к ширине рта": lambda: [
            _line_img(d, P("jaw_left"), P("jaw_right"), color, 3),
            _line_img(d, P("mouth_left"), P("mouth_right"), color, 3)],
        "Форма глаз": lambda: [
            _line_img(d, P("left_eye_top"), P("left_eye_bottom"), color, 3),
            _line_img(d, P("right_eye_top"), P("right_eye_bottom"), color, 3),
            _line_img(d, P("left_eye_outer"), P("left_eye_inner"), color, 2),
            _line_img(d, P("right_eye_inner"), P("right_eye_outer"), color, 2)],
        "Высота бровей": lambda: [
            _line_img(d, P("left_brow_mid"), P("left_eye_top"), color, 3),
            _line_img(d, P("right_brow_mid"), P("right_eye_top"), color, 3)],
    }
    fn = draws.get(metric_name)
    if fn:
        fn()
    return img


# ================== CHARTS ==================
def create_radar_chart(metrics, output_path):
    labels = [m["name"] for m in metrics]
    scores = [m["score"] for m in metrics]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    sp = scores + [scores[0]]
    ap = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(ap, sp, color=COLOR_ACCENT, linewidth=2)
    ax.fill(ap, sp, color=COLOR_ACCENT, alpha=0.25)
    for i, (a, ch) in enumerate(zip(angles, METRIC_COLORS)):
        ax.scatter([a], [scores[i]], color=ch, s=60, zorder=5,
                   edgecolors=COLOR_BG, linewidths=1.5)
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

def create_distribution_chart(score, output_path):
    fig, ax = plt.subplots(figsize=(6, 2))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    x = np.linspace(0, 10, 200)
    y = np.exp(-((x - 6.5) ** 2) / (2 * 1.4 ** 2))
    ax.fill_between(x, y, color=COLOR_BAR_BG, alpha=0.8)
    ax.plot(x, y, color=COLOR_LINE, linewidth=1)
    ax.axvline(score, color=COLOR_ACCENT, linewidth=2.5)
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.1); ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, facecolor=COLOR_BG, dpi=150,
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ================== PDF PRIMITIVES ==================
def _bg(c, w, h):
    c.setFillColor(HexColor(COLOR_BG))
    c.rect(0, 0, w, h, fill=1, stroke=0)

def _footer(c, w, page_num):
    c.setFillColor(HexColor(COLOR_TEXT_MUTED))
    c.setFont(FONT_REGULAR, 8)
    c.drawCentredString(w / 2, 22, f"Telegram: {BOT_USERNAME}   {page_num} / {TOTAL_PAGES}")

def _wrapped(c, text, x, y, max_chars=82, lh=14, font=None, size=10, color=COLOR_TEXT):
    c.setFillColor(HexColor(color))
    c.setFont(font or FONT_REGULAR, size)
    for line in wrap_text(text, max_chars):
        c.drawString(x, y, line)
        y -= lh
    return y

def _bar(c, x, y, width, score, color_hex, height=10):
    c.setFillColor(HexColor(COLOR_BAR_BG))
    c.roundRect(x, y, width, height, height / 2, fill=1, stroke=0)
    fw = width * (score / 10)
    if fw > 1:
        c.setFillColor(HexColor(color_hex))
        c.roundRect(x, y, fw, height, height / 2, fill=1, stroke=0)

def _dots(c, x, y, filled, color_hex, total=5, size=5, gap=12):
    for i in range(total):
        c.setFillColor(HexColor(color_hex if i < filled else COLOR_BAR_BG))
        c.circle(x + i * gap, y, size, fill=1, stroke=0)

def _accent_line(c, x, y_top, y_bottom, color_hex, width=3):
    c.setFillColor(HexColor(color_hex))
    c.rect(x, y_bottom, width, y_top - y_bottom, fill=1, stroke=0)

def _hline(c, x1, x2, y, color_hex=COLOR_LINE, width=0.5):
    c.setStrokeColor(HexColor(color_hex))
    c.setLineWidth(width)
    c.line(x1, y, x2, y)


# ================== PAGE: COVER ==================
def _cover(c, w, h, image_bytes, data):
    _bg(c, w, h)
    tier = data["tier"]

    c.setFillColor(HexColor(COLOR_TITLE))
    c.setFont(FONT_BOLD, 44)
    c.drawCentredString(w / 2, h - 100, BOT_NAME)
    c.setFillColor(HexColor(COLOR_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 11)
    c.drawCentredString(w / 2, h - 122, f"Telegram: {BOT_USERNAME}")
    c.setFillColor(HexColor(COLOR_TEXT))
    c.setFont(FONT_REGULAR, 12)
    c.drawCentredString(w / 2, h - 142, "Математический разбор пропорций лица")
    _hline(c, 60, w - 60, h - 165)

    # --- Фото (компактное, фикс. высота) ---
    img = base_image_for_overlay(image_bytes)
    img.thumbnail((200, 230))
    iw, ih = img.size
    ix = (w - iw) / 2
    iy = h - 185 - ih
    c.drawImage(ImageReader(img), ix, iy, width=iw, height=ih,
                preserveAspectRatio=True, mask="auto")

    # --- Оценка ---
    sy = iy - 14
    c.setFillColor(HexColor(COLOR_ACCENT))
    c.setFont(FONT_BOLD, 54)
    c.drawCentredString(w / 2, sy - 50, f"{data['score']:.2f}")
    c.setFillColor(HexColor(COLOR_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 12)
    c.drawCentredString(w / 2, sy - 64, "из 10")

    # --- Прогресс-бар ---
    bw = 280
    by = sy - 90
    _bar(c, (w - bw) / 2, by, bw, data["score"], tier["color"], height=12)

    # --- TIER ---
    ty = by - 30
    tstr = f"{tier['abbr']}    {tier['name']}"
    c.setFillColor(HexColor(tier["color"]))
    c.setFont(FONT_BOLD, 18)
    tw = c.stringWidth(tstr, FONT_BOLD, 18)
    c.drawString((w - tw) / 2, ty, tstr)

    # --- Топ % и уровень ---
    c.setFillColor(HexColor(COLOR_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 11)
    c.drawCentredString(w / 2, ty - 20, f"Ты входишь в топ {data['top_percent']}% по геометрии лица!")

    c.setFillColor(HexColor(COLOR_TITLE))
    c.setFont(FONT_BOLD, 13)
    c.drawCentredString(w / 2, ty - 42, f"Уровень: {data['level']}")

    c.setFillColor(HexColor(COLOR_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 10)
    c.drawCentredString(w / 2, ty - 60,
                        "Сильные стороны: " + ", ".join(m["name"].lower() for m in data["strengths"]))

    # --- Distribution chart ---
    dp = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    create_distribution_chart(data["score"], dp)
    dist_y = ty - 150
    c.drawImage(ImageReader(dp), w / 2 - 120, dist_y, width=240, height=75,
                preserveAspectRatio=True, mask="auto")

    # --- Блок впечатления (внизу, фикс.) ---
    iy2 = 150
    _accent_line(c, 60, iy2 + 18, iy2 - 66, tier["color"])
    c.setFillColor(HexColor(COLOR_TITLE))
    c.setFont(FONT_BOLD, 11)
    c.drawString(80, iy2, "ОБЩЕЕ ВПЕЧАТЛЕНИЕ")
    impression = (
        "Лицо проанализировано по ключевым геометрическим точкам с расчётом 20 "
        "антропометрических метрик. Каждая метрика сравнивается с медианными значениями "
        "и сигма-отклонением для твоего пола. Итоговая оценка отражает совокупную близость "
        "пропорций к статистическим нормам гармонии.")
    _wrapped(c, impression, 80, iy2 - 18, max_chars=86, lh=14, size=10)
    _footer(c, w, 1)


# ================== PAGE: PROFILE ==================
def _profile(c, w, h, data):
    _bg(c, w, h)
    c.setFillColor(HexColor(COLOR_TEXT))
    c.setFont(FONT_BOLD, 28)
    c.drawString(60, h - 80, "Профиль метрик")
    _hline(c, 60, w - 60, h - 100)

    rp = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    create_radar_chart(data["metrics"], rp)
    c.drawImage(ImageReader(rp), w / 2 - 155, h - 420, width=310, height=310,
                preserveAspectRatio=True, mask="auto")
    _hline(c, 60, w - 60, h - 435)

    c.setFillColor(HexColor(COLOR_ACCENT))
    c.setFont(FONT_BOLD, 12)
    c.drawString(60, h - 455, "Топ-3 сильных метрики")
    c.drawString(w / 2 + 10, h - 455, "Топ-3 зоны потенциала")

    y = h - 475
    for m in data["strengths"]:
        col = METRIC_COLORS[data["metrics"].index(m)]
        c.setFillColor(HexColor(col)); c.circle(70, y + 4, 3, fill=1, stroke=0)
        c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_REGULAR, 10)
        c.drawString(82, y, f"{m['name']}  -  {m['score']:.2f}")
        y -= 17

    y = h - 475
    for m in data["weak"]:
        col = METRIC_COLORS[data["metrics"].index(m)]
        c.setFillColor(HexColor(col)); c.circle(w / 2 + 20, y + 4, 3, fill=1, stroke=0)
        c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_REGULAR, 10)
        c.drawString(w / 2 + 32, y, f"{m['name']}  -  {m['score']:.2f}")
        y -= 17

    _hline(c, 60, w - 60, h - 535)
    c.setFillColor(HexColor(COLOR_TEXT))
    c.setFont(FONT_BOLD, 13)
    c.drawString(60, h - 555, "Вклад каждой метрики")

    c1, c2 = 60, w / 2 + 10
    cw = (w / 2) - 80
    yl = yr = h - 578
    for i, m in enumerate(data["metrics"]):
        col = METRIC_COLORS[i]
        cx = c1 if i < 10 else c2
        cy = yl if i < 10 else yr
        c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR, 8.5)
        c.drawString(cx, cy, m["name"])
        c.setFillColor(HexColor(COLOR_TEXT))
        c.drawRightString(cx + cw, cy, f"{m['score']:.2f}")
        _bar(c, cx, cy - 8, cw, m["score"], col, height=6)
        if i < 10: yl -= 19
        else:      yr -= 19

    by = min(yl, yr) - 8
    c.setFillColor(HexColor(COLOR_ACCENT))
    c.setFont(FONT_BOLD, 10)
    c.drawString(60, by, "КАК ЧИТАТЬ ОЦЕНКУ")
    legend = (
        "Каждая метрика — соотношение двух расстояний на лице (безразмерная пропорция). "
        "Норма — медианное значение по антропометрическим данным (Farkas et al.) для твоего пола. "
        "Оценка показывает близость к норме: 10 = совпадение. Сигма — мера естественного разброса: "
        "в +-1σ попадают ~68% людей. Низкая оценка не означает «некрасиво» — это статистическое отклонение.")
    _wrapped(c, legend, 60, by - 16, max_chars=110, lh=11, size=8.5, color=COLOR_TEXT_SOFT)
    _footer(c, w, 2)


# ================== PAGE: METRIC ==================
def _metric_page(c, w, h, image_bytes, m, idx, extra, gender, page_num):
    _bg(c, w, h)
    color = METRIC_COLORS[idx - 1]

    num = f"{idx:02d} / 20"
    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_BOLD, 13)
    nw = c.stringWidth(num, FONT_BOLD, 13)
    c.drawString(w - 60 - nw, h - 70, num)

    c.setFillColor(HexColor(color)); c.setFont(FONT_BOLD, 24)
    c.drawString(60, h - 75, m["name"])
    _hline(c, 60, w - 60, h - 95)

    ov = overlay_for_metric(image_bytes, m["name"], color)
    ov.thumbnail((220, 260))
    iw, ih = ov.size
    iy = h - 115 - ih
    c.drawImage(ImageReader(ov), 65, iy, width=iw, height=ih,
                preserveAspectRatio=True, mask="auto")

    rx = 320
    info_y = h - 115
    c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR, 10)
    c.drawString(rx, info_y, "Балл метрики")
    c.setFillColor(HexColor(color)); c.setFont(FONT_BOLD, 42)
    c.drawString(rx, info_y - 46, f"{m['score']:.2f}")
    sw = c.stringWidth(f"{m['score']:.2f}", FONT_BOLD, 42)
    c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR, 12)
    c.drawString(rx + sw + 6, info_y - 46, "/ 10")
    _bar(c, rx, info_y - 64, 200, m["score"], color, height=8)

    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_BOLD, 8)
    c.drawString(rx, info_y - 90, "ВАШ ПОКАЗАТЕЛЬ")
    c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD, 16)
    c.drawString(rx, info_y - 108, f"{m['value']:.4f}")

    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_REGULAR, 8)
    for i, line in enumerate(wrap_text(m["formula"], 42)[:2]):
        c.drawString(rx, info_y - 122 - i * 10, line)

    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_BOLD, 8)
    c.drawString(rx, info_y - 148, "НОРМА")
    c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD, 16)
    c.drawString(rx, info_y - 166, f"{m['norm']:.4f}")

    _dots(c, rx, info_y - 192, closeness_dots(m["score"]), color)
    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_REGULAR, 9)
    c.drawString(rx + 72, info_y - 196, "Близость к норме")

    title, p1, p2, p3, infl = generate_metric_text(m, extra, gender)
    ty = iy - 20
    _hline(c, 60, w - 60, ty + 8)
    c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD, 12)
    for line in wrap_text(title, 78):
        c.drawString(60, ty, line); ty -= 16
    ty -= 6
    ty = _wrapped(c, p1, 60, ty, max_chars=92, lh=13, size=9.5); ty -= 8
    if p2: ty = _wrapped(c, p2, 60, ty, max_chars=92, lh=13, size=9.5); ty -= 8
    if p3: ty = _wrapped(c, p3, 60, ty, max_chars=92, lh=13, size=9.5, color=COLOR_TEXT_SOFT); ty -= 10

    if ty > 110:
        _hline(c, 60, w - 60, ty + 4); ty -= 16
        _accent_line(c, 60, ty + 12, max(60, ty - 50), color)
        c.setFillColor(HexColor(color)); c.setFont(FONT_BOLD, 10)
        c.drawString(80, ty, "ВЛИЯНИЕ"); ty -= 14
        _wrapped(c, infl, 80, ty, max_chars=92, lh=13, size=9.5)
    _footer(c, w, page_num)


# ================== PAGE: RECOMMENDATIONS ==================
def _recommendations(c, w, h, data, gender):
    _bg(c, w, h)
    tier = data["tier"]
    c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD, 28)
    c.drawString(60, h - 80, "Рекомендации")
    c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR, 11)
    c.drawString(60, h - 100, "Персональные советы по улучшению")
    _hline(c, 60, w - 60, h - 115)

    c.setFillColor(HexColor(tier["color"])); c.setFont(FONT_BOLD, 12)
    c.drawCentredString(w / 2, h - 135,
                        f"Твой уровень: {tier['abbr']} · {tier['name']}  -  {data['score']:.2f} / 10")
    _hline(c, 60, w - 60, h - 150, color_hex=tier["color"], width=0.8)

    strengths, improvements = generate_recommendations(data, gender)

    c.setFillColor(HexColor(COLOR_ACCENT)); c.setFont(FONT_BOLD, 13)
    c.drawString(60, h - 170, "Что уже отлично")
    y = h - 192
    for i, (name, text) in enumerate(strengths):
        col = METRIC_COLORS[i % len(METRIC_COLORS)]
        top = y + 12
        c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD, 10)
        c.drawString(85, y, name)
        ty = _wrapped(c, text, 85, y - 14, max_chars=92, lh=12, size=9)
        _accent_line(c, 65, top, ty + 4, col)
        y = ty - 10

    y -= 8
    _hline(c, 60, w - 60, y); y -= 20
    c.setFillColor(HexColor(COLOR_TITLE)); c.setFont(FONT_BOLD, 13)
    c.drawString(60, y, "Что можно улучшить"); y -= 22
    for i, (name, text) in enumerate(improvements):
        if y < 70:
            break
        col = METRIC_COLORS[(i + 3) % len(METRIC_COLORS)]
        top = y + 12
        c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD, 10)
        c.drawString(85, y, name)
        ty = _wrapped(c, text, 85, y - 14, max_chars=92, lh=12, size=9)
        _accent_line(c, 65, top, ty + 4, col)
        y = ty - 10

    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_REGULAR, 9)
    c.drawString(60, 50, f"Дата разбора: {data['date']}")
    _footer(c, w, TOTAL_PAGES)


# ================== MAIN BUILDER ==================
def create_pdf_report(image_bytes, data, gender, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    w, h = A4

    _cover(c, w, h, image_bytes, data);  c.showPage()
    _profile(c, w, h, data);             c.showPage()

    page = 3
    for idx, m in enumerate(data["metrics"], start=1):
        _metric_page(c, w, h, image_bytes, m, idx, data["extra"], gender, page)
        page += 1
        c.showPage()

    _recommendations(c, w, h, data, gender)
    c.save()

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
from texts import generate_metric_text, generate_recommendations, generate_premium_focus_text


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


def wrap_text_width(c, text, max_width, font_name, font_size):
    lines = []
    for paragraph in str(text or "").splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue

        cur = ""
        for word in words:
            candidate = f"{cur} {word}".strip()
            if c.stringWidth(candidate, font_name, font_size) <= max_width:
                cur = candidate
                continue

            if cur:
                lines.append(cur)
                cur = ""

            if c.stringWidth(word, font_name, font_size) <= max_width:
                cur = word
            else:
                chunk = ""
                for ch in word:
                    candidate = chunk + ch
                    if c.stringWidth(candidate, font_name, font_size) <= max_width:
                        chunk = candidate
                    else:
                        if chunk:
                            lines.append(chunk)
                        chunk = ch
                cur = chunk

        if cur:
            lines.append(cur)
    return lines


def closeness_dots(score):
    if score >= 9:   return 5
    if score >= 7.5: return 4
    if score >= 6:   return 3
    if score >= 4:   return 2
    return 1


def display_metric_score(score):
    return max(2.5, float(score))


def score_status(score):
    if score < 2.5:
        return "зона высокого потенциала"
    if score < 5.0:
        return "зона роста"
    if score < 7.0:
        return "нейтральная зона"
    return "сильная зона"


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

def _ellipsize_to_width(c, text, max_width, font_name, size):
    suffix = "..."
    if c.stringWidth(text, font_name, size) <= max_width:
        return text
    while text and c.stringWidth(text + suffix, font_name, size) > max_width:
        text = text[:-1].rstrip()
    return (text + suffix) if text else suffix


def _wrapped(
    c, text, x, y, max_chars=82, lh=14, font=None, size=10, color=COLOR_TEXT,
    max_width=None, min_y=58, max_lines=None,
):
    font_name = font or FONT_REGULAR
    c.setFillColor(HexColor(color))
    c.setFont(font_name, size)
    lines = (wrap_text_width(c, text, max_width, font_name, size)
             if max_width else wrap_text(text, max_chars))
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = _ellipsize_to_width(c, lines[-1], max_width or 10_000, font_name, size)

    if min_y is not None:
        available = max(0, int((y - min_y) // lh) + 1)
        if len(lines) > available:
            lines = lines[:available]
            if lines:
                lines[-1] = _ellipsize_to_width(c, lines[-1], max_width or 10_000, font_name, size)

    for line in lines:
        if min_y is not None and y < min_y:
            break
        c.drawString(x, y, line)
        y -= lh
    return y

def _bar(c, x, y, width, score, color_hex, height=10):
    score = max(0, min(10, float(score)))
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
    strength_line = "Сильные стороны: " + ", ".join(m["name"].lower() for m in data["strengths"])
    c.drawCentredString(w / 2, ty - 60,
                        _ellipsize_to_width(c, strength_line, w - 120, FONT_REGULAR, 10))

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
    _wrapped(c, impression, 80, iy2 - 18, max_width=w - 160, lh=14, size=10, min_y=66)
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
        ds = display_metric_score(m["score"])
        c.setFillColor(HexColor(col)); c.circle(70, y + 4, 3, fill=1, stroke=0)
        c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_REGULAR, 10)
        c.drawString(82, y, f"{m['name']}  -  {ds:.2f}")
        y -= 17

    y = h - 475
    for m in data["weak"]:
        col = METRIC_COLORS[data["metrics"].index(m)]
        ds = display_metric_score(m["score"])
        c.setFillColor(HexColor(col)); c.circle(w / 2 + 20, y + 4, 3, fill=1, stroke=0)
        c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_REGULAR, 10)
        c.drawString(w / 2 + 32, y, f"{m['name']}  -  {ds:.2f}")
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
        ds = display_metric_score(m["score"])
        cx = c1 if i < 10 else c2
        cy = yl if i < 10 else yr
        c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR, 8.5)
        c.drawString(cx, cy, m["name"])
        c.setFillColor(HexColor(COLOR_TEXT))
        c.drawRightString(cx + cw, cy, f"{ds:.2f}")
        _bar(c, cx, cy - 8, cw, ds, col, height=6)
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
    _wrapped(c, legend, 60, by - 16, max_width=w - 120, lh=11, size=8.5, color=COLOR_TEXT_SOFT, min_y=58)
    _footer(c, w, 2)


# ================== PAGE: METRIC ==================
def _metric_page(c, w, h, image_bytes, m, idx, extra, gender, page_num):
    _bg(c, w, h)
    color = METRIC_COLORS[idx - 1]
    display_score = display_metric_score(m["score"])
    status = score_status(m["score"])
    margin = 58
    content_w = w - margin * 2

    num = f"{idx:02d} / 20"
    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_BOLD, 13)
    nw = c.stringWidth(num, FONT_BOLD, 13)
    c.drawString(w - margin - nw, h - 70, num)

    _wrapped(c, m["name"], margin, h - 70, font=FONT_BOLD, size=22,
             color=color, max_width=content_w - 90, lh=25, min_y=h - 125, max_lines=2)
    _hline(c, margin, w - margin, h - 126)

    ov = overlay_for_metric(image_bytes, m["name"], color)
    ov.thumbnail((220, 245))
    iw, ih = ov.size
    ix = margin + (230 - iw) / 2
    iy = h - 150 - ih
    c.setFillColor(HexColor(COLOR_BAR_BG))
    c.roundRect(margin, h - 405, 230, 255, 8, fill=1, stroke=0)
    c.drawImage(ImageReader(ov), ix, iy, width=iw, height=ih,
                preserveAspectRatio=True, mask="auto")

    rx = 320
    info_y = h - 155
    c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR, 10)
    c.drawString(rx, info_y, "Балл метрики")
    c.setFillColor(HexColor(color)); c.setFont(FONT_BOLD, 42)
    c.drawString(rx, info_y - 46, f"{display_score:.2f}")
    sw = c.stringWidth(f"{display_score:.2f}", FONT_BOLD, 42)
    c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR, 12)
    c.drawString(rx + sw + 6, info_y - 46, "/ 10")
    _bar(c, rx, info_y - 64, 200, display_score, color, height=8)
    c.setFillColor(HexColor(COLOR_TEXT_SOFT)); c.setFont(FONT_REGULAR, 9)
    c.drawString(rx, info_y - 82, status)
    if m["score"] < 2.5:
        c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_REGULAR, 8)
        c.drawString(rx, info_y - 96, "Технический score сохранён внутри расчёта.")

    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_BOLD, 8)
    c.drawString(rx, info_y - 122, "ВАШ ПОКАЗАТЕЛЬ")
    c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD, 16)
    c.drawString(rx, info_y - 140, f"{m['value']:.4f}")

    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_REGULAR, 8)
    for i, line in enumerate(wrap_text_width(c, m["formula"], 205, FONT_REGULAR, 8)[:2]):
        c.drawString(rx, info_y - 154 - i * 10, line)

    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_BOLD, 8)
    c.drawString(rx, info_y - 184, "НОРМА")
    c.setFillColor(HexColor(COLOR_TEXT)); c.setFont(FONT_BOLD, 16)
    c.drawString(rx, info_y - 202, f"{m['norm']:.4f}")

    _dots(c, rx, info_y - 228, closeness_dots(display_score), color)
    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_REGULAR, 9)
    c.drawString(rx + 72, info_y - 232, "Близость к норме")

    title, p1, p2, p3, infl = generate_metric_text(m, extra, gender)
    ty = h - 435
    _hline(c, margin, w - margin, ty + 14)
    c.setFillColor(HexColor(color)); c.setFont(FONT_BOLD, 9)
    c.drawString(margin, ty + 1, "КЛЮЧЕВОЙ ВЫВОД")
    ty -= 20
    ty = _wrapped(c, title, margin, ty, font=FONT_BOLD, size=12.2, color=COLOR_TEXT,
                  max_width=content_w, lh=15, min_y=110, max_lines=3)
    ty -= 10
    ty = _wrapped(c, p1, margin, ty, max_width=content_w, lh=13.2, size=9.4,
                  min_y=110, max_lines=5)
    ty -= 7
    if p2 and ty > 145:
        ty = _wrapped(c, p2, margin, ty, max_width=content_w, lh=13.2, size=9.4,
                      min_y=110, max_lines=4)
        ty -= 7
    if p3 and ty > 132:
        ty = _wrapped(c, p3, margin, ty, max_width=content_w, lh=12.2, size=8.6,
                      color=COLOR_TEXT_SOFT, min_y=106, max_lines=2)
        ty -= 8

    if ty > 102:
        _hline(c, margin, w - margin, ty + 4); ty -= 15
        _accent_line(c, margin, ty + 12, max(62, ty - 36), color)
        c.setFillColor(HexColor(color)); c.setFont(FONT_BOLD, 10)
        c.drawString(margin + 20, ty, "ВЛИЯНИЕ НА ВОСПРИЯТИЕ"); ty -= 14
        _wrapped(c, infl, margin + 20, ty, max_width=content_w - 24, lh=12.6,
                 size=9.1, min_y=58, max_lines=3)
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
        ty = _wrapped(c, text, 85, y - 14, max_width=w - 145, lh=12, size=9, min_y=58, max_lines=3)
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
        ty = _wrapped(c, text, 85, y - 14, max_width=w - 145, lh=12, size=9, min_y=58, max_lines=3)
        _accent_line(c, 65, top, ty + 4, col)
        y = ty - 10

    c.setFillColor(HexColor(COLOR_TEXT_MUTED)); c.setFont(FONT_REGULAR, 9)
    c.drawString(60, 50, f"Дата разбора: {data['date']}")
    _footer(c, w, TOTAL_PAGES)


# ================== PREMIUM PDF: LOOKSMAXXING PLAN ==================
LUX_BG = "#0B0B0D"
LUX_PANEL = "#17161A"
LUX_PANEL_SOFT = "#201D24"
LUX_GOLD = "#D4AF37"
LUX_GOLD_SOFT = "#E8D5A0"
LUX_PURPLE = "#9B7BC7"
LUX_TEXT = "#F4EEE2"
LUX_TEXT_SOFT = "#BDB2A1"
LUX_TEXT_MUTED = "#756C62"
LUX_LINE = "#3B3430"


def _lux_bg(c, w, h):
    c.setFillColor(HexColor(LUX_BG))
    c.rect(0, 0, w, h, fill=1, stroke=0)
    c.setFillColor(HexColor("#121014"))
    c.rect(0, h - 150, w, 150, fill=1, stroke=0)
    c.setFillColor(HexColor(LUX_GOLD))
    c.rect(0, h - 153, w, 1.1, fill=1, stroke=0)
    c.setFillColor(HexColor(LUX_PURPLE))
    c.rect(0, h - 156, w * 0.42, 1.1, fill=1, stroke=0)


def _lux_footer(c, w, page_num, total_pages):
    c.setFillColor(HexColor(LUX_TEXT_MUTED))
    c.setFont(FONT_REGULAR, 8)
    c.drawCentredString(w / 2, 24, f"Heim Face Premium Plan · {BOT_USERNAME} · {page_num} / {total_pages}")


def _lux_header(c, w, h, kicker, title, page_num, total_pages):
    _lux_bg(c, w, h)
    c.setFillColor(HexColor(LUX_GOLD))
    c.setFont(FONT_BOLD, 9)
    c.drawString(58, h - 72, kicker.upper())
    c.setFillColor(HexColor(LUX_TEXT))
    c.setFont(FONT_BOLD, 26)
    y = h - 100
    for line in wrap_text_width(c, title, w - 116, FONT_BOLD, 26)[:2]:
        c.drawString(58, y, line)
        y -= 29
    _lux_footer(c, w, page_num, total_pages)


def _lux_text(c, text, x, y, max_chars=78, lh=15, size=10, color=LUX_TEXT_SOFT, font=None,
              max_width=None, min_y=58, max_lines=None):
    return _wrapped(
        c, text, x, y, max_chars=max_chars, lh=lh, font=font or FONT_REGULAR,
        size=size, color=color, max_width=max_width, min_y=min_y, max_lines=max_lines,
    )


def _card_lines(c, text, max_width, font_name, font_size, max_lines):
    lines = wrap_text_width(c, text, max_width, font_name, font_size)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = _ellipsize_to_width(c, lines[-1], max_width, font_name, font_size)
    return lines


def _lux_card(c, x, y, width, height, title, text, accent=LUX_GOLD, body_size=9.8, max_lines=None):
    title_size = 10.8
    body_lh = body_size + 4
    pad_x = 18
    title_y = y - 23
    body_y = y - 46
    bottom = y - height + 16
    available_lines = max(1, int((body_y - bottom) // body_lh) + 1)
    if max_lines is None:
        max_lines = available_lines
    else:
        max_lines = min(max_lines, available_lines)

    c.setFillColor(HexColor(LUX_PANEL))
    c.roundRect(x, y - height, width, height, 8, fill=1, stroke=0)
    c.setFillColor(HexColor(accent))
    c.rect(x, y - height, 3, height, fill=1, stroke=0)
    c.setFillColor(HexColor(LUX_TEXT))
    c.setFont(FONT_BOLD, title_size)
    safe_title = _ellipsize_to_width(c, str(title), width - 2 * pad_x, FONT_BOLD, title_size)
    c.drawString(x + pad_x, title_y, safe_title)

    c.setFillColor(HexColor(LUX_TEXT_SOFT))
    c.setFont(FONT_REGULAR, body_size)
    lines = _card_lines(c, text, width - 2 * pad_x, FONT_REGULAR, body_size, max_lines)
    ty = body_y
    for line in lines:
        if ty < bottom:
            break
        c.drawString(x + pad_x, ty, line)
        ty -= body_lh


def _lux_bullets(c, items, x, y, max_chars=80, max_width=455, min_y=62):
    for title, text in items:
        if y < min_y + 28:
            break
        c.setFillColor(HexColor(LUX_GOLD))
        c.circle(x, y + 4, 3.2, fill=1, stroke=0)
        c.setFillColor(HexColor(LUX_TEXT))
        c.setFont(FONT_BOLD, 10.8)
        c.drawString(x + 16, y, _ellipsize_to_width(c, title, max_width, FONT_BOLD, 10.8))
        y = _lux_text(c, text, x + 16, y - 16, max_chars=max_chars, max_width=max_width,
                      lh=14, size=9.8, min_y=min_y, max_lines=3)
        y -= 14
    return y


def _lux_metric_names(metrics):
    return ", ".join(m["name"].lower() for m in metrics)


def _lux_group_label(metrics, default_label):
    names = _metric_names(metrics)
    if names & NOSE_METRICS:
        return "центральная зона"
    return default_label


def _lux_strengths_label(data):
    return _lux_group_label(data["strengths"], "топ-сильные зоны")


def _lux_weak_label(data):
    return _lux_group_label(data["weak"], "ключевые зоны роста")


def _metric_names(metrics):
    return {m["name"] for m in metrics}


def _has_metric(metrics, *names):
    present = _metric_names(metrics)
    return any(name in present for name in names)


EYE_METRICS = {"Размер глаз", "Расстояние между глазами", "Наклон глаз", "Биокулярная ширина", "Форма глаз", "Высота бровей"}
LIP_METRICS = {"Ширина рта", "Полнота губ", "Пропорции губ"}
JAW_METRICS = {"Баланс скул и челюсти", "Длина подбородка", "Контур подбородка", "Челюсть к ширине рта"}
NOSE_METRICS = {"Ширина носа", "Длина носа", "Нос к ширине рта"}
FRAME_METRICS = {"Пропорции лица", "Вертикальный баланс", "Ширина лба"}


def _metrics_in_group(metrics, group):
    return [m for m in metrics if m["name"] in group]


def _best_in_group(data, group):
    matches = _metrics_in_group(data["strengths"], group)
    return matches[0] if matches else None


def _weak_in_group(data, group):
    matches = _metrics_in_group(data["weak"], group)
    return matches[0] if matches else None


def _limiting_metric(data):
    return sorted(data["metrics"], key=lambda m: m["score"])[0]


def _potential_range(score, weak_count=3):
    lift = 0.38 + min(0.32, weak_count * 0.08)
    low = min(9.4, score + lift)
    high = min(9.7, low + 0.28)
    return round(low, 1), round(high, 1)


def _metric_group_name(metric_name):
    if metric_name in EYE_METRICS:
        return "глазная зона"
    if metric_name in LIP_METRICS:
        return "рот и губы"
    if metric_name in JAW_METRICS:
        return "нижняя треть"
    if metric_name in NOSE_METRICS:
        return "центральная зона"
    if metric_name in FRAME_METRICS:
        return "рамка лица"
    if metric_name == "Симметрия лица":
        return "симметрия"
    return "общий баланс"


def _primary_strength_sentence(data):
    m = data["strengths"][0]
    group = _metric_group_name(m["name"])
    return f"Главный актив: {m['name'].lower()} ({display_metric_score(m['score']):.2f}/10), зона: {group}."


def _primary_limit_sentence(data):
    m = _limiting_metric(data)
    group = _metric_group_name(m["name"])
    return f"Главный ограничивающий фактор: {m['name'].lower()} ({score_status(m['score'])}), зона: {group}."


def _compact_limit_sentence(data):
    m = _limiting_metric(data)
    status = "зона роста" if m["score"] < 5 else score_status(m["score"])
    return f"{m['name']} · {status}"


def _zone_strategy(metric_name, gender, strong=False):
    if metric_name in EYE_METRICS:
        if strong:
            return "Делай взгляд главным акцентом: чистая нижняя линия брови, мягкий свет чуть выше камеры, спокойное выражение без широкой улыбки."
        return "Собери взгляд через grooming бровей: убрать лишнее снизу, уложить волоски вверх-наружу, не закрывать глаза волосами и избегать верхнего жёсткого света."
    if metric_name in LIP_METRICS:
        if strong:
            return "Используй губы и рот как харизматичный акцент: увлажнение, мягкая полуулыбка, расслабленная нижняя треть, без сжатых губ."
        return "Не сжимай рот на фото, проверь увлажнение губ и контур. Лучше нейтральное выражение с лёгкой мягкостью, чем широкая напряжённая улыбка."
    if metric_name in JAW_METRICS:
        if gender == "male":
            return "Нижнюю треть усиливают щетина 2-5 мм, чистая линия шеи, камера чуть выше глаз и свет сверху-сбоку."
        return "Нижнюю треть смягчают вытянутая шея, лёгкий контур под скулой, чистая линия волос у лица и спокойный акцент на глазах."
    if metric_name in NOSE_METRICS:
        return "Центральную зону балансируют дистанция камеры, мягкий фронтальный свет, ракурс 10-15 градусов, чистые брови и спокойный контраст одежды."
    if metric_name in FRAME_METRICS:
        return "Рамку лица корректируют стрижка, объём и открытая верхняя треть: меньше случайной пышности, больше контролируемой формы."
    if metric_name == "Симметрия лица":
        if strong:
            return "Симметрию стоит показывать фронтальным мягким светом, ровной посадкой головы и аккуратной линией волос/бровей."
        return "Симметрию визуально собирают ровная укладка, одинаковая плотность бровей, фронтальный свет и отсутствие сильного наклона головы."
    return "Главный принцип: убрать визуальный шум, повторять рабочий свет и держать grooming максимально чистым."


def _metric_tip(name, gender, strong=False):
    if name in {"Ширина носа", "Нос к ширине рта", "Длина носа"}:
        return (
            "Центральная зона",
            _zone_strategy(name, gender, strong),
        )
    if name in {"Высота бровей", "Наклон глаз", "Форма глаз", "Размер глаз", "Расстояние между глазами", "Биокулярная ширина"}:
        if strong and name == "Форма глаз":
            return (
                "Взгляд как акцент",
                _zone_strategy(name, gender, strong),
            )
        return (
            "Брови и взгляд",
            _zone_strategy(name, gender, strong),
        )
    if name in {"Ширина рта", "Полнота губ", "Пропорции губ"}:
        return (
            "Рот и губы",
            _zone_strategy(name, gender, strong),
        )
    if name in {"Длина подбородка", "Контур подбородка", "Челюсть к ширине рта", "Баланс скул и челюсти"}:
        return (
            "Нижняя треть",
            _zone_strategy(name, gender, strong),
        )
    if name in {"Пропорции лица", "Вертикальный баланс", "Ширина лба"}:
        return (
            "Рамка лица",
            _zone_strategy(name, gender, strong),
        )
    if name == "Симметрия лица":
        return (
            "Симметрия",
            _zone_strategy(name, gender, strong),
        )
    return (
        "Визуальная настройка",
        "Работай через чистый grooming, спокойный свет, аккуратную одежду и повторяемые фото-условия. Это даёт эффект дороже, чем агрессивная коррекция.",
    )


def _strength_items(data, gender):
    items = []
    for m in data["strengths"]:
        title, tip = _metric_tip(m["name"], gender, strong=True)
        items.append((f"{m['name']} · {display_metric_score(m['score']):.2f}/10", tip))
    return items


def _growth_items(data, gender):
    items = []
    for m in data["weak"]:
        title, tip = _metric_tip(m["name"], gender, strong=False)
        items.append((f"{m['name']} · {score_status(m['score'])}", tip))
    return items


def _hair_plan(data, gender):
    weak = data["weak"]
    strengths = data["strengths"]
    eye_strength = _best_in_group(data, EYE_METRICS)
    jaw_weak = _weak_in_group(data, JAW_METRICS)
    frame_weak = _weak_in_group(data, FRAME_METRICS)
    nose_weak = _weak_in_group(data, NOSE_METRICS)

    if frame_weak and frame_weak["name"] == "Пропорции лица":
        recommended = "Текстурированный crop, side part с умеренным объёмом, medium taper, мягкие слои у лица."
        avoid = "Слишком высокий pompadour, тяжёлая ровная чёлка, экстремально короткие бока без баланса сверху."
        why = f"Потому что {frame_weak['name'].lower()} сейчас ограничивает рамку лица: стрижка должна управлять высотой и шириной, а не усиливать перекос."
    elif jaw_weak:
        recommended = "Средняя длина с чистыми висками, taper fade без агрессии, лёгкий объём сверху, аккуратная щетина для связки с нижней третью."
        avoid = "Гладко прилизанные волосы назад, очень длинные пряди вдоль щёк, резкий undercut без объёма у темени."
        why = f"Потому что зона «{jaw_weak['name']}» просит визуально собрать нижний контур и не утяжелять боковые линии."
    elif eye_strength:
        recommended = "Открытый лоб, лёгкая текстура, side part или curtain flow без перекрытия бровей."
        avoid = "Длинная чёлка ниже бровей, мокрая укладка на лицо, хаотичный объём у глаз."
        why = f"Потому что сильная зона «{eye_strength['name']}» должна быть видна: волосы не должны закрывать взгляд."
    elif nose_weak:
        recommended = "Мягкий объём сверху, чистые виски, форма без резких центральных линий и без пряди, падающей на нос."
        avoid = "Центральный пробор с жёсткими вертикалями, плоская укладка, сверхкороткая стрижка без рамки."
        why = f"Потому что «{nose_weak['name']}» лучше балансировать рамкой лица, светом и боковым объёмом, а не подчёркивать центр."
    else:
        recommended = "Классический taper, side part, мягкая текстура сверху, чистый контур висков и затылка."
        avoid = "Случайная длина без формы, пересушенная укладка, слишком много стайлинга и тяжёлая чёлка."
        why = "Потому что топ-сильные зоны лучше работают в чистой и спокойной рамке."

    return [
        ("Рекомендуемые стрижки", recommended),
        ("Нежелательные стрижки", avoid),
        ("Почему именно так", why),
        ("Точка контроля", "После стрижки сделай фото анфас и 3/4 при дневном свете. Если взгляд/нижняя треть стали читаться хуже — форма выбрана неудачно."),
    ]


def _brow_plan(data, gender):
    weak_brow = _weak_in_group(data, {"Высота бровей", "Наклон глаз", "Расстояние между глазами"})
    strong_eye = _best_in_group(data, EYE_METRICS)
    if weak_brow:
        remove = "Убрать лишние волоски под нижней линией и у переносицы, особенно всё, что визуально опускает хвостик."
        keep = "Не трогать верхнюю линию слишком агрессивно: она держит плотность и статус взгляда."
        style = "Укладывать вверх и наружу прозрачным гелем; хвостик направлять слегка к виску, без резкой графики."
        why = f"Потому что зона «{weak_brow['name']}» сейчас влияет на открытость и собранность взгляда."
    elif strong_eye:
        remove = "Убрать только визуальный шум: отдельные волоски снизу, у переносицы и выбившиеся по хвостику."
        keep = "Сохранить натуральную толщину и мягкую асимметрию: сильная зона «глаза» не требует тяжёлой коррекции."
        style = "Лёгкая укладка вверх, затем наружу. На фото бровь должна открывать глаз, а не становиться главным объектом."
        why = f"Потому что «{strong_eye['name']}» уже работает как актив, бровь должна его подсветить."
    else:
        remove = "Убрать нижний хаос и волоски у переносицы."
        keep = "Не делать брови тонкими и слишком тёмными."
        style = "Натуральная укладка по направлению роста плюс лёгкий подъём хвостика."
        why = "Потому что аккуратная бровь делает верхнюю треть дороже даже без яркого акцента на глазах."
    return [("Что убрать", remove), ("Что не трогать", keep), ("Как укладывать", style), ("Почему", why)]


def _photo_plan(data):
    weak = data["weak"]
    strengths = data["strengths"]
    nose_weak = _weak_in_group(data, NOSE_METRICS)
    jaw_weak = _weak_in_group(data, JAW_METRICS)
    eye_strength = _best_in_group(data, EYE_METRICS)
    lip_strength = _best_in_group(data, LIP_METRICS)

    best_angle = "3/4 на 10-15 градусов, камера чуть выше уровня глаз"
    worst_angle = "сверхблизко, 0.5x, камера снизу"
    if jaw_weak:
        best_angle = "камера чуть выше глаз, подбородок слегка вперёд-вниз, шея длинная"
        worst_angle = "камера снизу и втянутая шея: это прячет линию челюсти"
    if nose_weak:
        best_angle = "3/4 на 10 градусов с дистанцией от камеры; объектив без широкого угла"
        worst_angle = "фронтальный сверхкрупный план и 0.5x: центр лица станет визуально тяжелее"

    best_light = "мягкий фронтальный свет из окна, источник чуть выше глаз"
    worst_light = "нижний свет, точечный верхний свет, жёсткая лампа прямо над лицом"
    if eye_strength:
        best_light = f"мягкий верхне-фронтальный свет: он подчёркивает сильную зону «{eye_strength['name']}»"
    if lip_strength:
        best_angle += "; выражение — расслабленная полуулыбка"

    return [
        ("Лучший ракурс", best_angle),
        ("Худший ракурс", worst_angle),
        ("Лучший свет", best_light),
        ("Худший свет", worst_light),
        ("Персональный акцент", f"В кадре сначала считываются топ-сильные зоны; {_lux_group_label(weak, 'ключевые зоны роста')} не должны становиться центром внимания."),
    ]


def _limiter_plan(data, gender):
    limiter = _limiting_metric(data)
    return [
        ("Метрика", f"{limiter['name']} · {score_status(limiter['score'])}."),
        ("Что ограничивает", f"Влияет на {_metric_group_name(limiter['name'])}: эту зону лучше не выводить в центр кадра."),
        ("Как смягчать", _zone_strategy(limiter["name"], gender, strong=False)),
        ("Чего не делать", "Не компенсировать агрессивно. Сначала работают свет, ракурс, grooming, причёска и чистая подача."),
    ]


def _potential_plan(data, gender):
    low, high = _potential_range(data["score"], len(data["weak"]))
    limiter = _limiting_metric(data)
    return [
        ("Текущий score", f"{data['score']:.2f}/10 · tier {data['tier']['abbr']} · {data['tier']['name']}."),
        ("Прогноз после внедрения", f"{data['score']:.2f} → {low:.1f}-{high:.1f}. Это визуальный прогноз, а не пересчёт исходных метрик."),
        ("Что даёт прирост", f"Главный рычаг — «{limiter['name']}», затем топ-сильные зоны и чистая фото-подача."),
        ("Условие", "Система на 30 дней: волосы, брови, кожа, depuff-утро и одинаковые фото-условия."),
    ]


def _lux_section_page(c, w, h, page_num, total_pages, kicker, title, intro, items):
    _lux_header(c, w, h, kicker, title, page_num, total_pages)
    y = h - 170
    y = _lux_text(c, intro, 58, y, max_width=w - 116, lh=15, size=10.4, color=LUX_TEXT_SOFT, min_y=62, max_lines=4)
    y -= 20
    _lux_bullets(c, items, 64, y, max_chars=88, max_width=w - 144, min_y=62)


def _lux_cover(c, w, h, image_bytes, data, page_num, total_pages):
    _lux_bg(c, w, h)
    tier = data["tier"]
    limiter = _limiting_metric(data)

    c.setFillColor(HexColor(LUX_GOLD))
    c.setFont(FONT_BOLD, 10)
    c.drawString(58, h - 76, "HEIM FACE · PREMIUM PLAN")
    c.setFillColor(HexColor(LUX_TEXT))
    c.setFont(FONT_BOLD, 36)
    c.drawString(58, h - 118, "Луксмаксинг-план")
    c.setFillColor(HexColor(LUX_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 11)
    c.drawString(58, h - 140, "Персональная стратегия внешнего потенциала")

    img = base_image_for_overlay(image_bytes)
    img.thumbnail((255, 330))
    iw, ih = img.size
    ix = w - iw - 58
    iy = h - 475
    c.setFillColor(HexColor(LUX_PANEL))
    c.roundRect(ix - 10, iy - 10, iw + 20, ih + 20, 10, fill=1, stroke=0)
    c.drawImage(ImageReader(img), ix, iy, width=iw, height=ih, preserveAspectRatio=True, mask="auto")

    c.setFillColor(HexColor(LUX_TEXT_MUTED))
    c.setFont(FONT_BOLD, 8)
    c.drawString(58, h - 205, "ПРОФИЛЬ")
    c.setFillColor(HexColor(tier["color"]))
    c.setFont(FONT_BOLD, 56)
    c.drawString(58, h - 267, f"{data['score']:.2f}")
    c.setFillColor(HexColor(LUX_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 13)
    c.drawString(205, h - 261, "/ 10")
    _bar(c, 58, h - 292, 235, data["score"], tier["color"], height=9)

    c.setFillColor(HexColor(LUX_GOLD_SOFT))
    c.setFont(FONT_BOLD, 15)
    c.drawString(58, h - 325, f"{tier['abbr']} · {tier['name']}")
    c.setFillColor(HexColor(LUX_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 10)
    c.drawString(58, h - 345, f"Дата: {data['date']}")

    _lux_card(c, 58, 250, 225, 82, "Главный актив", _primary_strength_sentence(data), accent=LUX_GOLD, max_lines=2)
    _lux_card(c, 312, 250, 225, 82, "Главный лимитер", _compact_limit_sentence(data), accent=LUX_PURPLE, body_size=10.4, max_lines=2)
    _lux_card(
        c, 58, 145, w - 116, 74,
        "Топ сильных зон",
        "топ-сильные зоны · главный визуальный актив для света, ракурса и grooming",
        accent=tier["color"],
        body_size=10.2,
        max_lines=2,
    )
    _lux_footer(c, w, page_num, total_pages)


def _lux_score_page(c, w, h, data, page_num, total_pages):
    _lux_header(c, w, h, "Executive summary", "Короткий итог по внешнему потенциалу", page_num, total_pages)
    tier = data["tier"]
    focus_text, best_text, growth_text = generate_premium_focus_text(data)
    y = h - 185
    c.setFillColor(HexColor(tier["color"]))
    c.setFont(FONT_BOLD, 64)
    c.drawString(58, y, f"{data['score']:.2f}")
    c.setFillColor(HexColor(LUX_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 14)
    c.drawString(222, y + 8, "/ 10")
    _bar(c, 58, y - 30, 360, data["score"], tier["color"], height=11)

    y -= 70
    _lux_text(
        c,
        f"База уже рассчитана в основном отчёте: 20 антропометрических метрик, симметрия, пропорции и итоговый tier. {focus_text}",
        58, y, max_width=w - 116, lh=15, size=10.2,
    )

    y -= 95
    _lux_card(c, 58, y, 225, 96, "Сильный визуальный сигнал", best_text)
    _lux_card(c, 312, y, 225, 96, "Главная зона роста", growth_text)

    y -= 135
    _lux_text(
        c,
        "Цель плана — не менять лицо радикально, а поднять воспринимаемый уровень за счёт ухоженности, чистого силуэта, правильного света, контроля отёчности и деталей, которые усиливают геометрию.",
        58, y, max_width=w - 116, lh=15, size=10.2, color=LUX_TEXT,
    )


def _lux_strengths_page(c, w, h, data, page_num, total_pages):
    items = _strength_items(data, data.get("gender", "male"))
    intro = (
        f"Твой tier: {data['tier']['abbr']} · {data['tier']['name']}. "
        "Сильные стороны — это элементы, которые нужно сделать видимыми: светом, ракурсом, аккуратной рамкой лица и чистым grooming."
    )
    _lux_section_page(c, w, h, page_num, total_pages, "Strengths", "Сильные стороны", intro, items)


def _lux_growth_page(c, w, h, data, page_num, total_pages):
    items = _growth_items(data, data.get("gender", "male"))
    intro = "Зоны роста — это точки с наибольшим потенциалом визуального улучшения. Они не означают проблему; это просто места, где аккуратная настройка даст максимальный эффект."
    _lux_section_page(c, w, h, page_num, total_pages, "Growth map", "Зоны роста", intro, items)


def _lux_limiter_page(c, w, h, data, gender, page_num, total_pages):
    _lux_header(c, w, h, "Main limiter", "Главный ограничивающий фактор", page_num, total_pages)
    limiter = _limiting_metric(data)
    group = _metric_group_name(limiter["name"])
    visual_score = display_metric_score(limiter["score"])
    y = h - 175

    c.setFillColor(HexColor(LUX_PURPLE))
    c.setFont(FONT_BOLD, 10)
    c.drawString(58, y, "САМАЯ СЛАБАЯ МЕТРИКА В ПРОФИЛЕ")
    y -= 32
    c.setFillColor(HexColor(LUX_TEXT))
    c.setFont(FONT_BOLD, 28)
    for line in wrap_text_width(c, limiter["name"], w - 116, FONT_BOLD, 28)[:2]:
        c.drawString(58, y, line)
        y -= 32

    y -= 4
    c.setFillColor(HexColor(LUX_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 10.5)
    c.drawString(58, y, f"Зона: {group} · визуально: {score_status(limiter['score'])}")
    y -= 28
    _bar(c, 58, y, 300, visual_score, LUX_PURPLE, height=10)
    c.setFillColor(HexColor(LUX_TEXT_MUTED))
    c.setFont(FONT_REGULAR, 8.5)
    c.drawString(58, y - 17, "В PDF низкие метрики показываются мягко; технический score остаётся внутри расчёта.")

    cards = _limiter_plan(data, gender)
    _lux_card(c, 58, 410, 225, 110, cards[0][0], cards[0][1], accent=LUX_PURPLE, body_size=10.2, max_lines=3)
    _lux_card(c, 312, 410, 225, 110, cards[1][0], cards[1][1], accent=LUX_GOLD, body_size=10.2, max_lines=3)
    _lux_card(c, 58, 276, 225, 116, cards[2][0], cards[2][1], accent=LUX_GOLD, body_size=9.9, max_lines=4)
    _lux_card(c, 312, 276, 225, 116, cards[3][0], cards[3][1], accent=LUX_PURPLE, body_size=10.0, max_lines=4)

    _lux_text(
        c,
        "Смысл этой страницы — не зафиксировать недостаток, а выбрать самый выгодный рычаг. В следующих разделах стрижка, брови, фото и стиль настроены так, чтобы этот фактор не становился центром внимания.",
        58, 118, max_width=w - 116, lh=14, size=9.7, color=LUX_TEXT_SOFT,
    )


def _lux_potential_page(c, w, h, data, gender, page_num, total_pages):
    _lux_header(c, w, h, "Potential", "Потенциал после внедрения", page_num, total_pages)
    low, high = _potential_range(data["score"], len(data["weak"]))
    limiter = _limiting_metric(data)
    tier = data["tier"]
    y = h - 185

    c.setFillColor(HexColor(LUX_TEXT_MUTED))
    c.setFont(FONT_BOLD, 9)
    c.drawString(58, y, "ТЕКУЩИЙ УРОВЕНЬ")
    c.drawString(332, y, "ПРОГНОЗИРУЕМЫЙ ДИАПАЗОН")
    y -= 52
    c.setFillColor(HexColor(tier["color"]))
    c.setFont(FONT_BOLD, 42)
    c.drawString(58, y, f"{data['score']:.2f}")
    c.setFillColor(HexColor(LUX_GOLD))
    c.setFont(FONT_BOLD, 35)
    c.drawString(238, y + 2, "→")
    c.setFillColor(HexColor(LUX_TEXT))
    c.setFont(FONT_BOLD, 42)
    c.drawString(332, y, f"{low:.1f}-{high:.1f}")
    c.setFillColor(HexColor(LUX_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 10)
    c.drawString(58, y - 24, f"tier {tier['abbr']} · {tier['name']}")
    c.drawString(332, y - 24, "визуальный диапазон после 30 дней")

    y -= 58
    _bar(c, 58, y, 200, data["score"], tier["color"], height=10)
    _bar(c, 332, y, 200, high, LUX_GOLD, height=10)

    y -= 58
    intro = (
        f"Прогноз строится на текущем score {data['score']:.2f}/10, tier {tier['abbr']} "
        f"и главном лимитере «{limiter['name']}». Это не медицинская оценка и не обещание изменения геометрии: "
        "речь о том, как лицо может считываться после настройки подачи."
    )
    y = _lux_text(c, intro, 58, y, max_width=w - 116, lh=15, size=10.2, color=LUX_TEXT_SOFT)
    y -= 24

    cards = _potential_plan(data, gender)
    _lux_card(c, 58, y, 225, 98, cards[2][0], cards[2][1], accent=LUX_GOLD, body_size=10.0, max_lines=3)
    _lux_card(c, 312, y, 225, 98, cards[3][0], cards[3][1], accent=LUX_PURPLE, body_size=10.0, max_lines=3)
    y -= 126
    _lux_card(c, 58, y, 225, 98, "Персональные рычаги", f"{limiter['name']} · {_lux_weak_label(data)} · топ-сильные зоны.", accent=LUX_PURPLE, body_size=10.0, max_lines=3)
    _lux_card(c, 312, y, 225, 98, "Как мерить прогресс", "Фото анфас и 3/4 в одинаковом свете. Сравнивай контур, взгляд, кожу и первое впечатление.", accent=LUX_GOLD, body_size=10.0, max_lines=3)


def _lux_hair_page(c, w, h, data, gender, page_num, total_pages):
    items = _hair_plan(data, gender)
    intro = (
        f"Стрижка выбирается не по моде, а по метрикам. {_primary_strength_sentence(data)} "
        f"{_primary_limit_sentence(data)} Поэтому форма волос должна одновременно показать актив и не усилить лимитер."
    )
    _lux_section_page(c, w, h, page_num, total_pages, "Hair", "Причёска", intro, items)


def _lux_brows_page(c, w, h, data, gender, page_num, total_pages):
    items = _brow_plan(data, gender)
    eye_strength = _best_in_group(data, EYE_METRICS)
    brow_weak = _weak_in_group(data, {"Высота бровей", "Наклон глаз", "Расстояние между глазами"})
    marker = f"Сильный ориентир: {eye_strength['name']}." if eye_strength else f"Зона контроля: {brow_weak['name']}." if brow_weak else "Цель: чистая верхняя треть."
    intro = f"Брови управляют выражением лица. {marker} Раздел ниже — конкретный grooming-протокол, а не общий совет."
    _lux_section_page(c, w, h, page_num, total_pages, "Brows", "Брови", intro, items)


def _lux_skin_page(c, w, h, data, gender, page_num, total_pages):
    score = data["score"]
    limiter = _limiting_metric(data)
    eye_weak = _weak_in_group(data, EYE_METRICS)
    nose_weak = _weak_in_group(data, NOSE_METRICS)
    jaw_weak = _weak_in_group(data, JAW_METRICS)
    eye_strength = _best_in_group(data, EYE_METRICS)
    lip_strength = _best_in_group(data, LIP_METRICS)
    intensity = "держи уход простым и стабильным" if score >= 7 else "начни с минимальной, но регулярной базы"
    photo_focus = (
        f"область под глазами и брови, потому что зона «{eye_weak['name']}» чувствительна к теням"
        if eye_weak else
        f"центральная зона, потому что «{nose_weak['name']}» быстро усиливается бликом и широким углом"
        if nose_weak else
        f"нижняя треть, потому что зона «{jaw_weak['name']}» зависит от чистого контура"
        if jaw_weak else
        f"акцент на сильную зону «{(eye_strength or lip_strength or data['strengths'][0])['name']}»"
    )
    items = [
        ("База утром", f"Мягкое очищение, лёгкое увлажнение и SPF. При score {score:.2f}/10 задача — не терять визуальный уровень из-за усталой текстуры: {intensity}."),
        ("База вечером", f"Восстановление барьера без пересушивания. Если кожа раздражена, лимитер «{limiter['name']}» будет считываться заметнее, потому что внимание уйдёт в визуальный шум."),
        ("Тон и текстура", f"Цель — спокойная матово-сатиновая кожа. Главный фото-фокус: {photo_focus}."),
        ("Перед съёмкой", "Умыться, увлажнить кожу, убрать блеск с Т-зоны и проверить губы. Это помогает топ-сильным зонам считываться первыми."),
    ]
    intro = f"Кожа — фон для геометрии. В твоём профиле она должна поддержать топ-сильные зоны и не усиливать {_lux_weak_label(data)}."
    _lux_section_page(c, w, h, page_num, total_pages, "Skin", "Кожа", intro, items)


def _lux_beard_page(c, w, h, data, gender, page_num, total_pages):
    weak = data["weak"]
    limiter = _limiting_metric(data)
    jaw_weak = _weak_in_group(data, JAW_METRICS)
    jaw_strength = _best_in_group(data, JAW_METRICS)
    lip_strength = _best_in_group(data, LIP_METRICS)
    lower_focus = (
        f"Нижняя треть входит в зоны роста через «{jaw_weak['name']}», поэтому главный фокус — чистый контур, шея и свет."
        if jaw_weak else
        f"Нижняя треть уже может работать как актив через «{jaw_strength['name']}»: важно сохранить чистую форму."
        if jaw_strength else
        f"Главный лимитер — «{limiter['name']}», поэтому нижняя треть должна быть аккуратной и не забирать лишнее внимание."
    )
    if gender == "male":
        items = [
            ("Щетина 2-5 мм", f"{lower_focus} Короткая ровная щетина добавляет плотность челюсти и визуально собирает подбородок."),
            ("Линия шеи", f"Не поднимай линию бороды слишком высоко: при лимитере «{limiter['name']}» грязная шея быстро утяжеляет весь кадр."),
            ("Усы и рот", f"{'Не перекрывай тяжёлыми усами сильную зону «' + lip_strength['name'] + '».' if lip_strength else 'Держи линию губ читаемой: рот не должен теряться под щетиной.'}"),
            ("Плотность", "Если рост неравномерный, лучше чистое бритьё. Твоя цель — усилить топ-сильные зоны, а не показать случайную текстуру."),
        ]
    else:
        items = [
            ("Нижняя треть", f"{lower_focus} Для женского образа важны гладкость кожи, отсутствие визуального шума и мягкий контур."),
            ("Контур", f"Лёгкая тень под скулой и по нижней линии уместна, если она не выводит в центр «{limiter['name']}»."),
            ("Губы", f"{'Увлажнённые губы и мягкая полуулыбка подчеркнут сильную зону «' + lip_strength['name'] + '».' if lip_strength else 'Увлажнённые губы и чистый контур делают нижнюю треть аккуратнее на крупном плане.'}"),
            ("Минимализм", "Один главный акцент выглядит дороже. В твоём случае логичнее вести внимание к топ-сильным зонам."),
        ]
    intro = f"Нижняя треть влияет на статус и ухоженность. Раздел настроен под твой профиль: топ-сильные зоны и лимитер «{limiter['name']}»."
    _lux_section_page(c, w, h, page_num, total_pages, "Lower third", "Щетина/борода", intro, items)


def _lux_depuff_page(c, w, h, data, page_num, total_pages):
    eye_weak = _weak_in_group(data, EYE_METRICS)
    jaw_weak = _weak_in_group(data, JAW_METRICS)
    focus = "под глазами и верхняя треть" if eye_weak else "нижняя треть и линия челюсти" if jaw_weak else "скулы, глаза и общий контур"
    items = [
        ("Фокус зоны", f"У тебя depuff особенно важен для зоны: {focus}. Это помогает не маскировать сильные метрики и не усиливать лимитер."),
        ("Сон", "7-8 часов сна и стабильное время подъёма. Главный эффект даёт регулярность, потому что лицо начинает выглядеть чётче ещё до фото."),
        ("Соль и вода", "Вечером уменьши солёное и алкоголь, утром выпей воду до кофе. Это простая база для более собранной нижней и верхней трети."),
        ("Утренний протокол", "Холодный компресс 2-3 минуты, мягкий лимфодренаж от центра лица к ушам, затем лёгкое увлажнение."),
        ("Бонусный PDF", "Отдельный файл по снижению отёчности отправляется вместе с Premium Plan: используй его как чек-лист перед фото."),
    ]
    intro = "Отёчность может скрывать топ-сильные зоны. Поэтому протокол ниже привязан к твоему профилю, а не является общим советом."
    _lux_section_page(c, w, h, page_num, total_pages, "Depuff", "Снижение отёчности", intro, items)


def _lux_photo_page(c, w, h, data, page_num, total_pages):
    items = _photo_plan(data)
    intro = (
        f"Фото должно показывать топ-сильные зоны и не выводить в центр {_lux_weak_label(data)}. "
        "Поэтому ракурс и свет ниже привязаны к твоим метрикам."
    )
    _lux_section_page(c, w, h, page_num, total_pages, "Photo", "Фото и позирование", intro, items)


def _lux_style_page(c, w, h, data, gender, page_num, total_pages):
    tier = data["tier"]
    limiter = _limiting_metric(data)
    eye_strength = _best_in_group(data, EYE_METRICS)
    lip_strength = _best_in_group(data, LIP_METRICS)
    accent = "акцент на глаза" if eye_strength else "акцент на губы/улыбку" if lip_strength else "акцент на чистую рамку лица"
    items = [
        ("Палитра", f"Для tier {tier['abbr']} используй чёрный, графит, молочный, глубокий синий. Золото или мягкий фиолетовый — небольшой акцент, чтобы поддержать premium-вид."),
        ("Контраст", f"Главный лимитер — {limiter['name'].lower()}, поэтому не ставь рядом с лицом крупные принты и шумные воротники: они перетянут внимание."),
        ("Вырез и ворот", "Открытая шея визуально улучшает нижнюю треть. Если низ лица в зонах роста — избегай высокого ворота и шарфов у подбородка."),
        ("Аксессуары", f"Оставь один акцент: {accent}. Несколько акцентов рядом с лицом дробят внимание и делают образ дешевле."),
    ]
    intro = f"Одежда должна работать на топ-сильные зоны и не усиливать {_lux_weak_label(data)}."
    _lux_section_page(c, w, h, page_num, total_pages, "Style", "Одежда, цвета и визуальная подача", intro, items)


def _lux_mistakes_page(c, w, h, data, page_num, total_pages):
    weak = data["weak"]
    strengths = data["strengths"]
    limiter = _limiting_metric(data)
    nose_weak = _weak_in_group(data, NOSE_METRICS)
    jaw_weak = _weak_in_group(data, JAW_METRICS)
    brow_weak = _weak_in_group(data, {"Высота бровей", "Наклон глаз", "Расстояние между глазами"})
    lip_strength = _best_in_group(data, LIP_METRICS)
    eye_strength = _best_in_group(data, EYE_METRICS)
    camera_text = (
        f"0.5x и сверхкрупный план усилят «{nose_weak['name']}». Держи дистанцию и 3/4 на 10 градусов."
        if nose_weak else
        f"Камера снизу прячет линию челюсти, особенно при зоне «{jaw_weak['name']}». Лучше чуть выше глаз."
        if jaw_weak else
        f"Слишком близкая камера может вывести в центр лимитер «{limiter['name']}». Держи объектив дальше от лица."
    )
    brow_text = (
        f"Тяжёлая прядь на лбу ухудшит «{brow_weak['name']}». Открой брови и верхнюю треть."
        if brow_weak else
        f"Не закрывай актив «{eye_strength['name']}»: волосы и брови должны подсвечивать взгляд."
        if eye_strength else
        "Волосы на лице и хаотичные брови добавляют шум, даже если верхняя треть не является слабой зоной."
    )
    mouth_text = (
        f"Сжатые губы скрывают сильную зону «{lip_strength['name']}». Лучше мягкая полуулыбка и расслабленная нижняя треть."
        if lip_strength else
        "Сжатые губы ухудшают нижнюю треть и делают выражение напряжённым. Держи рот расслабленным."
    )
    items = [
        ("Сверхблизкая камера", camera_text),
        ("Жёсткий верхний свет", f"Он даёт тени под глазами и делает кожу менее ровной. При лимитере «{limiter['name']}» лучше мягкий свет из окна."),
        ("Закрытые брови", brow_text),
        ("Напряжённый рот", mouth_text),
        ("Случайный фон", "Шумный фон спорит с топ-сильными зонами. Чем чище кадр, тем дороже считывается лицо."),
    ]
    if eye_strength:
        items.append(("Потерянный взгляд", f"Не смотри пусто мимо камеры: «{eye_strength['name']}» может быть главным активом кадра. Держи фокус и лёгкое напряжение взгляда."))
    intro = f"Ошибки ниже подобраны под {_lux_group_label(weak, 'ключевые зоны роста')} и топ-сильные зоны."
    _lux_section_page(c, w, h, page_num, total_pages, "Photo mistakes", "Ошибки, которые ухудшают лицо на фото", intro, items)


def _lux_7day_page(c, w, h, data, page_num, total_pages):
    limiter = _limiting_metric(data)
    items = [
        ("День 1", f"Сделай фото анфас и 3/4. Отметь, как в кадре выглядит лимитер: {limiter['name'].lower()}."),
        ("День 2", "Приведи волосы к форме из раздела «Причёска». Цель — показать топ-сильные зоны."),
        ("День 3", "Настрой базовый уход: очищение, увлажнение, SPF утром; мягкое очищение и восстановление вечером."),
        ("День 4", "Проведи depuff-утро и сравни контур лица с фото дня 1 при таком же свете."),
        ("День 5", "Проверь нижнюю треть: щетина/бритьё/контур должны выглядеть намеренно, особенно если низ лица в зонах роста."),
        ("День 6", "Собери 2 рабочих фото-сценария из персонального фото-раздела: лучший свет и лучший ракурс."),
        ("День 7", f"Сравни фото и оставь только решения, которые усилили активы и не вывели в центр {limiter['name'].lower()}."),
    ]
    intro = "Семидневный план построен вокруг твоей самой слабой метрики и топ-3 сильных сторон, поэтому он не повторяет универсальный чек-лист."
    _lux_section_page(c, w, h, page_num, total_pages, "7 days", "План на 7 дней", intro, items)


def _lux_30day_page(c, w, h, data, page_num, total_pages):
    limiter = _limiting_metric(data)
    low, high = _potential_range(data["score"], len(data["weak"]))
    items = [
        ("Неделя 1", f"Убери визуальный шум: кожа, волосы, брови, отёчность. Фокус — не усиливать {limiter['name'].lower()}."),
        ("Неделя 2", "Подбери форму причёски и нижней трети. Сделай 10 тестовых фото с разным светом и дистанцией."),
        ("Неделя 3", "Усиль топ-сильные зоны. Ракурсы и grooming строятся вокруг них."),
        ("Неделя 4", f"Зафиксируй личный стандарт. Целевой визуальный диапазон после внедрения: {low:.1f}-{high:.1f}/10."),
        ("После 30 дней", "Повтори фото в тех же условиях. Сравни не ощущения, а видимые изменения: кожа, контур, взгляд, симметрия кадра."),
    ]
    intro = f"Тридцатидневный план переводит текущие {data['score']:.2f}/10 в более сильную визуальную подачу без изменения исходной геометрии."
    _lux_section_page(c, w, h, page_num, total_pages, "30 days", "План на 30 дней", intro, items)


def _lux_morning_page(c, w, h, data, page_num, total_pages):
    eye_strength = _best_in_group(data, EYE_METRICS)
    limiter = _limiting_metric(data)
    items = [
        ("Вода и лицо", "Стакан воды до кофе, умывание прохладной водой, 2-3 минуты холодного компресса на зоны отёчности."),
        ("Кожа", "Лёгкое увлажнение, SPF, контроль блеска на Т-зоне. Кожа должна выглядеть спокойной, не перегруженной."),
        ("Волосы", f"Проверь, не закрывают ли волосы актив: {eye_strength['name'] if eye_strength else data['strengths'][0]['name']}."),
        ("Брови", "Расчеши вверх и наружу, убери явные выбившиеся волоски. На фото проверь симметрию хвостиков."),
        ("Нижняя треть", "Бритьё или щетина должны выглядеть намеренно. Проверь шею, линию челюсти, губы и сухость кожи вокруг рта."),
        ("Фото-контроль", f"Сделай тестовый кадр. Если лимитер «{limiter['name']}» стал заметнее — меняй свет/ракурс до отправки фото."),
    ]
    intro = f"Утренний чек-лист привязан к твоему профилю: актив — {data['strengths'][0]['name']}, лимитер — {limiter['name']}."
    _lux_section_page(c, w, h, page_num, total_pages, "Morning checklist", "Чек-лист утренней подготовки", intro, items)


def _lux_final_page(c, w, h, data, page_num, total_pages):
    _lux_header(c, w, h, "Next steps", "Что делать дальше", page_num, total_pages)
    y = h - 170
    summary = (
        f"Твой текущий ориентир: {data['score']:.2f}/10, tier {data['tier']['abbr']} · {data['tier']['name']}. "
        f"Фокус: топ-сильные зоны и {_lux_weak_label(data)}."
    )
    y = _lux_text(c, summary, 58, y, max_width=w - 116, lh=15, size=10.5, color=LUX_TEXT)
    y -= 22
    items = [
        ("1. Сохрани базовые фото", "Анфас и 3/4 при дневном свете. Они нужны, чтобы сравнивать прогресс без самообмана."),
        ("2. Выбери три действия", "Одно по волосам, одно по коже, одно по фото. Не пытайся менять всё сразу."),
        ("3. Повтори через 30 дней", "Сделай фото в тех же условиях и сравни: взгляд, кожа, контур, отёчность, общий силуэт."),
        ("4. Используй основной PDF", "Основной отчёт показывает цифры. Premium Plan показывает, как превратить цифры в визуальные решения."),
    ]
    _lux_bullets(c, items, 64, y, max_width=w - 144, min_y=96)
    c.setFillColor(HexColor(LUX_GOLD))
    c.setFont(FONT_BOLD, 13)
    c.drawCentredString(w / 2, 82, "Heim Face · спокойная геометрия, чистая подача, системный прогресс")


def create_looksmaxxing_pdf(image_bytes, analysis_data, gender, output_path):
    analysis_data = dict(analysis_data)
    analysis_data["gender"] = gender
    c = canvas.Canvas(output_path, pagesize=A4)
    w, h = A4
    total_pages = 18

    _lux_cover(c, w, h, image_bytes, analysis_data, 1, total_pages); c.showPage()
    _lux_score_page(c, w, h, analysis_data, 2, total_pages); c.showPage()
    _lux_strengths_page(c, w, h, analysis_data, 3, total_pages); c.showPage()
    _lux_growth_page(c, w, h, analysis_data, 4, total_pages); c.showPage()
    _lux_limiter_page(c, w, h, analysis_data, gender, 5, total_pages); c.showPage()
    _lux_potential_page(c, w, h, analysis_data, gender, 6, total_pages); c.showPage()
    _lux_hair_page(c, w, h, analysis_data, gender, 7, total_pages); c.showPage()
    _lux_brows_page(c, w, h, analysis_data, gender, 8, total_pages); c.showPage()
    _lux_skin_page(c, w, h, analysis_data, gender, 9, total_pages); c.showPage()
    _lux_beard_page(c, w, h, analysis_data, gender, 10, total_pages); c.showPage()
    _lux_depuff_page(c, w, h, analysis_data, 11, total_pages); c.showPage()
    _lux_photo_page(c, w, h, analysis_data, 12, total_pages); c.showPage()
    _lux_style_page(c, w, h, analysis_data, gender, 13, total_pages); c.showPage()
    _lux_mistakes_page(c, w, h, analysis_data, 14, total_pages); c.showPage()
    _lux_7day_page(c, w, h, analysis_data, 15, total_pages); c.showPage()
    _lux_30day_page(c, w, h, analysis_data, 16, total_pages); c.showPage()
    _lux_morning_page(c, w, h, analysis_data, 17, total_pages); c.showPage()
    _lux_final_page(c, w, h, analysis_data, 18, total_pages)
    c.save()


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

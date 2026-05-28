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


def _lux_card(c, x, y, width, height, title, text, accent=LUX_GOLD):
    c.setFillColor(HexColor(LUX_PANEL))
    c.roundRect(x, y - height, width, height, 8, fill=1, stroke=0)
    c.setFillColor(HexColor(accent))
    c.rect(x, y - height, 3, height, fill=1, stroke=0)
    c.setFillColor(HexColor(LUX_TEXT))
    c.setFont(FONT_BOLD, 11)
    c.drawString(x + 18, y - 23, title)
    _lux_text(c, text, x + 18, y - 43, max_width=width - 34, lh=13, size=9.2, min_y=y - height + 14)


def _lux_bullets(c, items, x, y, max_chars=80, max_width=455, min_y=62):
    for title, text in items:
        if y < min_y + 28:
            break
        c.setFillColor(HexColor(LUX_GOLD))
        c.circle(x, y + 4, 3.2, fill=1, stroke=0)
        c.setFillColor(HexColor(LUX_TEXT))
        c.setFont(FONT_BOLD, 10.5)
        c.drawString(x + 16, y, title)
        y = _lux_text(c, text, x + 16, y - 16, max_chars=max_chars, max_width=max_width,
                      lh=13, size=9.2, min_y=min_y, max_lines=4)
        y -= 13
    return y


def _lux_metric_names(metrics):
    return ", ".join(m["name"].lower() for m in metrics)


def _metric_names(metrics):
    return {m["name"] for m in metrics}


def _has_metric(metrics, *names):
    present = _metric_names(metrics)
    return any(name in present for name in names)


def _metric_tip(name, gender, strong=False):
    if name in {"Ширина носа", "Нос к ширине рта", "Длина носа"}:
        return (
            "Центральная зона",
            "Держи мягкий фронтальный свет, избегай нижнего света и сверхблизкой камеры. Чистая линия волос, аккуратные брови и умеренный контраст одежды помогают сбалансировать центр лица без радикальных решений.",
        )
    if name in {"Высота бровей", "Наклон глаз", "Форма глаз", "Размер глаз", "Расстояние между глазами", "Биокулярная ширина"}:
        if strong and name == "Форма глаз":
            return (
                "Взгляд как акцент",
                "Подчёркивай форму глаз чистой линией бровей, мягким боковым светом и ракурсом 3/4. На фото держи лёгкое напряжение нижнего века: взгляд становится собраннее.",
            )
        return (
            "Брови и взгляд",
            "Убери лишние волоски по нижней линии, уложи брови вверх и наружу, проверь хвостик брови на фото. Для кадра используй свет чуть выше уровня глаз и спокойное выражение.",
        )
    if name in {"Ширина рта", "Полнота губ", "Пропорции губ"}:
        return (
            "Рот и губы",
            "Перед фото увлажни губы, не сжимай рот и держи мягкое нейтральное выражение. Если эта зона сильная, лёгкая полуулыбка даст больше харизмы, чем широкая улыбка.",
        )
    if name in {"Длина подбородка", "Контур подбородка", "Челюсть к ширине рта", "Баланс скул и челюсти"}:
        if gender == "male":
            return (
                "Нижняя треть",
                "Проверь щетину 2-5 мм, чистую линию шеи и свет сверху-сбоку. Это визуально собирает челюсть и делает нижний контур чётче.",
            )
        return (
            "Нижняя треть",
            "Используй мягкий контур по нижней линии и не перегружай макияж губ. На фото слегка вытягивай шею и подавай подбородок вперёд-вниз.",
        )
    if name in {"Пропорции лица", "Вертикальный баланс", "Ширина лба"}:
        return (
            "Рамка лица",
            "Причёска должна балансировать вертикаль и ширину: меньше случайного объёма, больше контролируемой формы. Открывай сильные зоны и не закрывай брови тяжёлой прядью.",
        )
    if name == "Симметрия лица":
        return (
            "Симметрия",
            "Держи пробор, брови и бороду/контур максимально аккуратными. На фото выбирай фронтальный свет и не наклоняй голову сильнее 5-7 градусов.",
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


def _lux_section_page(c, w, h, page_num, total_pages, kicker, title, intro, items):
    _lux_header(c, w, h, kicker, title, page_num, total_pages)
    y = h - 170
    y = _lux_text(c, intro, 58, y, max_width=w - 116, lh=15, size=10.2, color=LUX_TEXT_SOFT, min_y=62)
    y -= 20
    _lux_bullets(c, items, 64, y, max_chars=88, max_width=w - 144, min_y=62)


def _lux_cover(c, w, h, image_bytes, data, page_num, total_pages):
    _lux_bg(c, w, h)
    tier = data["tier"]

    c.setFillColor(HexColor(LUX_GOLD))
    c.setFont(FONT_BOLD, 10)
    c.drawString(58, h - 76, "HEIM FACE · PREMIUM PLAN")
    c.setFillColor(HexColor(LUX_TEXT))
    c.setFont(FONT_BOLD, 34)
    c.drawString(58, h - 118, "Луксмаксинг-план")
    c.setFillColor(HexColor(LUX_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 11)
    c.drawString(58, h - 140, "Персональная стратегия внешнего потенциала")

    img = base_image_for_overlay(image_bytes)
    img.thumbnail((215, 270))
    iw, ih = img.size
    ix = w - iw - 58
    iy = h - 420
    c.setFillColor(HexColor(LUX_PANEL))
    c.roundRect(ix - 10, iy - 10, iw + 20, ih + 20, 10, fill=1, stroke=0)
    c.drawImage(ImageReader(img), ix, iy, width=iw, height=ih, preserveAspectRatio=True, mask="auto")

    c.setFillColor(HexColor(LUX_TEXT_MUTED))
    c.setFont(FONT_BOLD, 8)
    c.drawString(58, h - 230, "ОЦЕНКА ВНЕШНЕГО ПОТЕНЦИАЛА")
    c.setFillColor(HexColor(tier["color"]))
    c.setFont(FONT_BOLD, 58)
    c.drawString(58, h - 292, f"{data['score']:.2f}")
    c.setFillColor(HexColor(LUX_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 13)
    c.drawString(205, h - 286, "/ 10")
    _bar(c, 58, h - 316, 245, data["score"], tier["color"], height=9)

    c.setFillColor(HexColor(LUX_GOLD_SOFT))
    c.setFont(FONT_BOLD, 15)
    c.drawString(58, h - 355, f"{tier['abbr']} · {tier['name']}")
    c.setFillColor(HexColor(LUX_TEXT_SOFT))
    c.setFont(FONT_REGULAR, 10)
    c.drawString(58, h - 375, f"Дата: {data['date']}")

    _lux_card(
        c, 58, 215, w - 116, 92,
        "Фокус плана",
        "Этот PDF не заменяет основной разбор лица. Он переводит метрики в практичные шаги: причёска, кожа, брови, нижняя треть, отёчность, фото и режим.",
        accent=LUX_GOLD,
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


def _lux_hair_page(c, w, h, data, gender, page_num, total_pages):
    male = gender == "male"
    weak = data["weak"]
    strengths = data["strengths"]
    shape_tip = "Сделай форму более вертикальной и чистой по бокам." if _has_metric(weak, "Пропорции лица", "Ширина лба") else "Сохраняй форму, которая открывает сильные зоны лица."
    eye_tip = "Не закрывай брови и верхнюю треть тяжёлой прядью: у тебя взгляд может быть сильным визуальным акцентом." if _has_metric(strengths, "Форма глаз", "Размер глаз", "Наклон глаз") else "Открытая верхняя треть делает лицо чище и дороже."
    items = [
        ("Форма", f"{shape_tip} Если лицо кажется вытянутым — меньше высоты сверху; если широким — больше вертикали и чистые боковые линии."),
        ("Объём", f"Контролируемый объём лучше случайной пышности. {eye_tip}"),
        ("Контур", "Чистая линия висков и затылка делает лицо дороже визуально. Обновление формы каждые 3-5 недель поддерживает премиальный вид."),
        ("Стиль", "Держи образ минималистичным: графит, чёрный, молочный, глубокие холодные оттенки. Слишком шумная укладка конкурирует с геометрией лица."),
    ]
    if not male:
        items[2] = ("Контур", "Слои у лица, мягкая рамка и аккуратный объём помогают балансировать лоб, скулы и нижнюю треть без тяжёлого визуального эффекта.")
    intro = "Причёска — главный инструмент рамки лица. Она может усилить скулы, вытянуть силуэт, смягчить нижнюю треть или, наоборот, сделать образ собраннее."
    _lux_section_page(c, w, h, page_num, total_pages, "Hair", "Причёска", intro, items)


def _lux_brows_page(c, w, h, data, gender, page_num, total_pages):
    weak = data["weak"]
    strengths = data["strengths"]
    brow_focus = "Это одна из твоих зон роста, поэтому брови лучше сделать максимально чистыми и управляемыми." if _has_metric(weak, "Высота бровей", "Наклон глаз") else "Задача бровей — поддержать уже имеющуюся геометрию взгляда."
    eye_focus = "Так как форма/размер глаз входят в сильные стороны, не перегружай взгляд: чистая бровь и мягкий свет дадут больше, чем тяжёлый контраст." if _has_metric(strengths, "Форма глаз", "Размер глаз") else "Аккуратная линия бровей визуально собирает верхнюю треть."
    items = [
        ("Линия", f"{brow_focus} Брови должны поддерживать направление глаз, а не спорить с ним."),
        ("Плотность", "Не делай брови слишком графичными. Премиальный эффект — это чистая форма, натуральная плотность и отсутствие лишних волосков по нижней линии."),
        ("Симметрия", "Если одна бровь визуально выше, корректируй не толщиной, а нижней линией и укладкой волосков прозрачным гелем."),
        ("Взгляд", eye_focus),
    ]
    intro = "Брови управляют выражением лица. В Premium Plan их задача — сделать взгляд чище, увереннее и дороже, не превращая лицо в маску."
    _lux_section_page(c, w, h, page_num, total_pages, "Brows", "Брови", intro, items)


def _lux_skin_page(c, w, h, data, gender, page_num, total_pages):
    score = data["score"]
    intensity = "держи уход простым и стабильным" if score >= 7 else "начни с минимальной, но регулярной базы"
    items = [
        ("База утром", f"Мягкое очищение, лёгкое увлажнение и SPF. При твоей общей оценке {score:.2f}/10 особенно важно не терять баллы из-за усталой текстуры: {intensity}."),
        ("База вечером", "Очищение без пересушивания и восстановление барьера. Если кожа реагирует раздражением, уменьши активы и оставь стабильную простую схему."),
        ("Тон и текстура", "Цель — не идеальная фарфоровость, а спокойная матово-сатиновая кожа без жирного блеска, шелушений и сильной красноты."),
        ("Разбор фото", "Перед съёмкой: умыться, увлажнить кожу, убрать блеск с Т-зоны, проверить область под глазами и губы."),
    ]
    intro = "Кожа — это фон для всей геометрии лица. Даже сильные пропорции выглядят слабее, если фон уставший, пересушенный или блестящий."
    _lux_section_page(c, w, h, page_num, total_pages, "Skin", "Кожа", intro, items)


def _lux_beard_page(c, w, h, data, gender, page_num, total_pages):
    weak = data["weak"]
    lower_focus = "Нижняя треть входит в зоны роста, поэтому главный фокус — чистый контур, шея и свет." if _has_metric(weak, "Длина подбородка", "Контур подбородка", "Челюсть к ширине рта") else "Нижняя треть не требует агрессивной коррекции: достаточно поддерживать чистый контур."
    if gender == "male":
        items = [
            ("Щетина 2-5 мм", f"{lower_focus} Короткая ровная щетина добавляет плотность челюсти и визуально собирает подбородок."),
            ("Линия шеи", "Не поднимай линию бороды слишком высоко. Чистая шея и аккуратный нижний край делают контур дороже."),
            ("Усы и рот", "Если ширина рта входит в сильные стороны, не перекрывай её тяжёлыми усами. Линия губ должна оставаться читаемой."),
            ("Плотность", "При редком росте лучше чистое бритьё, чем неравномерная борода. Премиальность всегда в аккуратности."),
        ]
    else:
        items = [
            ("Нижняя треть", f"{lower_focus} Для женского образа важны гладкость кожи, отсутствие визуального шума и мягкий контур."),
            ("Контур", "Лёгкое контурирование под скулой и по нижней линии помогает собрать овал без тяжёлого макияжа."),
            ("Губы", "Увлажнённые губы и чистый контур делают нижнюю треть аккуратнее, особенно на фото крупным планом."),
            ("Минимализм", "Избегай перегруженных акцентов сразу на губах, бровях и глазах. Один главный акцент выглядит дороже."),
        ]
    intro = "Нижняя треть сильно влияет на ощущение зрелости, статуса и ухоженности. Здесь важны чистые границы и отсутствие случайности."
    _lux_section_page(c, w, h, page_num, total_pages, "Lower third", "Щетина/борода", intro, items)


def _lux_depuff_page(c, w, h, data, page_num, total_pages):
    items = [
        ("Сон", "7-8 часов сна и стабильное время подъёма заметно уменьшают отёчность лица. Главный эффект даёт регулярность, не разовые лайфхаки."),
        ("Соль и вода", "Вечером уменьши солёное и алкоголь, утром выпей воду до кофе. Это простая база для более чёткого лица."),
        ("Утренний протокол", "Холодный компресс 2-3 минуты, мягкий лимфодренаж от центра лица к ушам, затем лёгкое увлажнение."),
        ("Бонусный PDF", "Отдельный файл по снижению отёчности отправляется вместе с Premium Plan: сохрани его как быстрый чек-лист на утро."),
    ]
    intro = "Отёчность может скрывать скулы, глаза и линию челюсти. Работа с ней часто даёт быстрый визуальный прирост без изменения самой геометрии."
    _lux_section_page(c, w, h, page_num, total_pages, "Depuff", "Снижение отёчности", intro, items)


def _lux_photo_page(c, w, h, data, page_num, total_pages):
    weak = data["weak"]
    strengths = data["strengths"]
    nose_note = "Так как центральная зона входит в зоны роста, избегай 0.5x, близкой камеры и нижнего света." if _has_metric(weak, "Ширина носа", "Нос к ширине рта", "Длина носа") else "Держи камеру на комфортной дистанции, чтобы не искажать центральную зону."
    eye_note = "Если взгляд входит в сильные стороны, делай его главным акцентом: камера чуть выше глаз и мягкий верхне-фронтальный свет." if _has_metric(strengths, "Форма глаз", "Размер глаз", "Наклон глаз") else "Свет чуть выше уровня глаз делает верхнюю треть спокойнее."
    items = [
        ("Свет", f"Лучший вариант — мягкий фронтальный свет из окна. {eye_note}"),
        ("Камера", f"Держи камеру чуть выше уровня глаз, без сильного широкоугольного искажения. {nose_note}"),
        ("Поза", "Шея длинная, подбородок слегка вперёд и вниз, плечи расслаблены. Это собирает нижнюю треть и улучшает контур."),
        ("Выражение", "Нейтральное лицо плюс лёгкое напряжение глаз. Сильная улыбка меняет пропорции, а пустой взгляд снижает выразительность."),
    ]
    intro = "Фото должно показывать сильные стороны, а не случайно искажать лицо. Хороший свет и дистанция часто важнее фильтров."
    _lux_section_page(c, w, h, page_num, total_pages, "Photo", "Фото и позирование", intro, items)


def _lux_style_page(c, w, h, data, gender, page_num, total_pages):
    tier = data["tier"]
    items = [
        ("Палитра", "Рабочая база Heim Face: чёрный, графит, молочный, глубокий синий, тёмный зелёный. Золото или мягкий фиолетовый — как небольшой акцент, не как главный цвет."),
        ("Контраст", f"Для уровня {tier['abbr']} лучше работает чистый контраст без визуального шума: однотонный верх, спокойная фактура, без крупных принтов рядом с лицом."),
        ("Вырез и ворот", "Открытая шея визуально улучшает нижнюю треть. Слишком высокий ворот может утяжелять подбородок и скрывать линию челюсти."),
        ("Аксессуары", "Оставь один акцент: часы, цепь, серьги или очки. Несколько акцентов рядом с лицом дробят внимание и делают образ дешевле."),
    ]
    intro = "Одежда и цвет не меняют метрики, но сильно меняют первое впечатление. Цель — создать дорогую рамку для лица и не спорить с геометрией."
    _lux_section_page(c, w, h, page_num, total_pages, "Style", "Одежда, цвета и визуальная подача", intro, items)


def _lux_mistakes_page(c, w, h, data, page_num, total_pages):
    weak = data["weak"]
    strengths = data["strengths"]
    items = [
        ("Сверхблизкая камера", "Искажает нос, рот и нижнюю треть. Особенно избегай этого, если в зонах роста есть ширина носа, нос к ширине рта или длина носа."),
        ("Жёсткий верхний свет", "Даёт тени под глазами, усиливает отёчность и делает кожу менее ровной. Лучше мягкий свет из окна или большой рассеянный источник."),
        ("Закрытые брови", "Тяжёлая прядь на лбу ухудшает верхнюю треть и взгляд. Если слабая зона — высота бровей, это особенно заметно."),
        ("Напряжённый рот", "Сжатые губы ухудшают нижнюю треть. Если сильная зона — ширина рта или губы, мягкое выражение лица покажет её лучше."),
        ("Случайный фон", "Шумный фон снижает ощущение статуса. Чем чище кадр, тем дороже выглядит лицо."),
    ]
    if _has_metric(strengths, "Форма глаз", "Размер глаз", "Наклон глаз"):
        items.append(("Потерянный взгляд", "Не смотри пусто мимо камеры: твоя глазная зона может быть главным активом кадра. Держи фокус и лёгкое напряжение взгляда."))
    intro = "Эти ошибки часто портят фото сильнее, чем реальные пропорции. Убери их — и отчёт начнёт работать в твою пользу."
    _lux_section_page(c, w, h, page_num, total_pages, "Photo mistakes", "Ошибки, которые ухудшают лицо на фото", intro, items)


def _lux_7day_page(c, w, h, data, page_num, total_pages):
    items = [
        ("День 1", "Сделай чистое фото анфас и 3/4 при дневном свете. Это контрольная точка для сравнения."),
        ("День 2", "Приведи волосы и брови к аккуратной форме. Убери всё, что закрывает сильные зоны лица."),
        ("День 3", "Настрой базовый уход: очищение, увлажнение, SPF утром; мягкое очищение и восстановление вечером."),
        ("День 4", "Проведи depuff-утро: вода, холод, лёгкий массаж, меньше соли вечером до этого."),
        ("День 5", "Проверь нижнюю треть: щетина/бритьё/контур должны выглядеть намеренно, а не случайно."),
        ("День 6", "Собери 2 рабочих фото-сценария: дневной свет и вечерний мягкий свет."),
        ("День 7", "Сравни фото с первым днём и оставь только то, что реально улучшило лицо."),
    ]
    intro = "Семидневный план нужен для быстрых визуальных улучшений и настройки рутины без перегруза."
    _lux_section_page(c, w, h, page_num, total_pages, "7 days", "План на 7 дней", intro, items)


def _lux_30day_page(c, w, h, data, page_num, total_pages):
    items = [
        ("Неделя 1", "Стабилизируй сон, воду, кожу и базовую аккуратность волос/бровей. Цель — убрать визуальный шум."),
        ("Неделя 2", "Подбери форму причёски и нижней трети. Сделай 10 тестовых фото с разным светом и дистанцией."),
        ("Неделя 3", "Усиль сильные стороны из отчёта: выстраивай укладку, брови и ракурсы вокруг топ-3 метрик."),
        ("Неделя 4", "Зафиксируй личный стандарт: уход, стрижка, depuff-протокол, 2 лучших ракурса и одежда в премиальной палитре."),
        ("После 30 дней", "Повтори фото в тех же условиях. Сравни не ощущения, а видимые изменения: кожа, контур, взгляд, симметрия кадра."),
    ]
    intro = "Тридцатидневный план переводит разовые улучшения в устойчивый внешний стандарт."
    _lux_section_page(c, w, h, page_num, total_pages, "30 days", "План на 30 дней", intro, items)


def _lux_morning_page(c, w, h, data, page_num, total_pages):
    items = [
        ("Вода и лицо", "Стакан воды до кофе, умывание прохладной водой, 2-3 минуты холодного компресса на зоны отёчности."),
        ("Кожа", "Лёгкое увлажнение, SPF, контроль блеска на Т-зоне. Кожа должна выглядеть спокойной, не перегруженной."),
        ("Волосы", "Проверь форму у висков, объём сверху и то, не закрывают ли волосы брови или сильную глазную зону."),
        ("Брови", "Расчеши вверх и наружу, убери явные выбившиеся волоски. На фото проверь симметрию хвостиков."),
        ("Нижняя треть", "Бритьё или щетина должны выглядеть намеренно. Проверь шею, линию челюсти, губы и сухость кожи вокруг рта."),
        ("Фото-контроль", "Сделай один тестовый кадр при дневном свете. Если лицо выглядит плоским — повернись на 10-15 градусов и подними камеру чуть выше."),
    ]
    intro = "Этот чек-лист нужен для дней, когда важно выглядеть максимально собранно: съёмка, встреча, свидание, деловой день."
    _lux_section_page(c, w, h, page_num, total_pages, "Morning checklist", "Чек-лист утренней подготовки", intro, items)


def _lux_final_page(c, w, h, data, page_num, total_pages):
    _lux_header(c, w, h, "Next steps", "Что делать дальше", page_num, total_pages)
    y = h - 170
    summary = (
        f"Твой текущий ориентир: {data['score']:.2f}/10, tier {data['tier']['abbr']} · {data['tier']['name']}. "
        f"Сильные стороны: {_lux_metric_names(data['strengths'])}. Зоны роста: {_lux_metric_names(data['weak'])}."
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
    c = canvas.Canvas(output_path, pagesize=A4)
    w, h = A4
    total_pages = 16

    _lux_cover(c, w, h, image_bytes, analysis_data, 1, total_pages); c.showPage()
    _lux_score_page(c, w, h, analysis_data, 2, total_pages); c.showPage()
    _lux_strengths_page(c, w, h, analysis_data, 3, total_pages); c.showPage()
    _lux_growth_page(c, w, h, analysis_data, 4, total_pages); c.showPage()
    _lux_hair_page(c, w, h, analysis_data, gender, 5, total_pages); c.showPage()
    _lux_brows_page(c, w, h, analysis_data, gender, 6, total_pages); c.showPage()
    _lux_skin_page(c, w, h, analysis_data, gender, 7, total_pages); c.showPage()
    _lux_beard_page(c, w, h, analysis_data, gender, 8, total_pages); c.showPage()
    _lux_depuff_page(c, w, h, analysis_data, 9, total_pages); c.showPage()
    _lux_photo_page(c, w, h, analysis_data, 10, total_pages); c.showPage()
    _lux_style_page(c, w, h, analysis_data, gender, 11, total_pages); c.showPage()
    _lux_mistakes_page(c, w, h, analysis_data, 12, total_pages); c.showPage()
    _lux_7day_page(c, w, h, analysis_data, 13, total_pages); c.showPage()
    _lux_30day_page(c, w, h, analysis_data, 14, total_pages); c.showPage()
    _lux_morning_page(c, w, h, analysis_data, 15, total_pages); c.showPage()
    _lux_final_page(c, w, h, analysis_data, 16, total_pages)
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

# -*- coding: utf-8 -*-
"""Генерация PNG-карточки результата для шаринга."""
import os

from PIL import Image, ImageDraw, ImageFont, ImageFilter

BOT_USERNAME = "@heim_face_bot"

# Палитра (тёмный фон, золото/графит/бирюза)
BG_TOP = (18, 24, 33)       # #121821
BG_BOTTOM = (10, 14, 18)    # #0A0E12
GOLD = (212, 175, 55)       # #D4AF37 — score/tier
TEAL = (45, 212, 191)       # #2DD4BF — основной акцент
TEAL_SOFT = (94, 234, 212)  # #5EEAD4 — glow
TEXT = (220, 227, 232)      # #DCE3E8
TEXT_SOFT = (124, 138, 150) # #7C8A96
LINE = (36, 50, 64)         # #243240

CARD_W, CARD_H = 1080, 1350

_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
]


def _fonts():
    for reg, bold in _FONT_CANDIDATES:
        if os.path.exists(reg) and os.path.exists(bold):
            return reg, bold
    return None, None


def _load(path, size):
    return ImageFont.truetype(path, size) if path else ImageFont.load_default()


def _center(draw, text, font, y, w, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((w - tw) / 2, y), text, font=font, fill=fill)


def create_share_card(data: dict, output_path: str) -> str:
    """Создаёт PNG-карточку результата. Возвращает путь к файлу."""
    img = Image.new("RGB", (CARD_W, CARD_H), BG_TOP)
    draw = ImageDraw.Draw(img)

    # Вертикальный градиент фона
    for y in range(CARD_H):
        t = y / CARD_H
        r = int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t)
        g = int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t)
        b = int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t)
        draw.line([(0, y), (CARD_W, y)], fill=(r, g, b))

    reg_path, bold_path = _fonts()
    f_brand = _load(bold_path, 64)
    f_score_label = _load(reg_path, 38)
    f_score = _load(bold_path, 170)
    f_section = _load(bold_path, 40)
    f_item = _load(reg_path, 44)
    f_footer = _load(bold_path, 44)
    f_subtle = _load(reg_path, 30)

    # Рамка
    draw.rectangle([40, 40, CARD_W - 40, CARD_H - 40], outline=LINE, width=2)

    # Бренд
    _center(draw, "HEIM FACE", f_brand, 110, CARD_W, GOLD)
    draw.line([(CARD_W / 2 - 120, 205), (CARD_W / 2 + 120, 205)], fill=GOLD, width=3)

    # Score (с мягким бирюзовым свечением)
    _center(draw, "FACE HARMONY SCORE", f_score_label, 290, CARD_W, TEXT_SOFT)
    score = float(data.get("score", 0))
    # glow: рисуем цифру на отдельном слое, размываем, накладываем
    glow = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    score_txt = f"{score:.2f}"
    gb = gdraw.textbbox((0, 0), score_txt, font=f_score)
    gx = (CARD_W - (gb[2] - gb[0])) / 2
    gdraw.text((gx, 350), score_txt, font=f_score, fill=(94, 234, 212, 200))
    glow = glow.filter(ImageFilter.GaussianBlur(18))
    img.paste(glow, (0, 0), glow)
    draw = ImageDraw.Draw(img)
    _center(draw, score_txt, f_score, 350, CARD_W, TEAL)
    _center(draw, "/ 10", f_score_label, 570, CARD_W, TEXT_SOFT)

    # Tier (авто-подбор размера под ширину)
    tier = data.get("tier", {})
    tier_str = f"{tier.get('abbr', '')} · {tier.get('name', '')}".strip(" ·")
    tier_size = 64
    while tier_size > 30:
        f_tier = _load(bold_path, tier_size)
        bbox = draw.textbbox((0, 0), tier_str, font=f_tier)
        if (bbox[2] - bbox[0]) <= CARD_W - 200:
            break
        tier_size -= 4
    _center(draw, tier_str, f_tier, 650, CARD_W, GOLD)

    # Сильные стороны
    y = 800
    draw.text((130, y), "TOP FACIAL STRENGTHS", font=f_section, fill=TEXT_SOFT)
    y += 80
    strengths = data.get("strengths", [])[:3]
    for m in strengths:
        draw.ellipse([135, y + 18, 155, y + 38], fill=TEAL)
        draw.text((185, y), m["name"], font=f_item, fill=TEXT)
        y += 80

    # Футер
    draw.line([(130, CARD_H - 220), (CARD_W - 130, CARD_H - 220)], fill=LINE, width=2)
    _center(draw, BOT_USERNAME, f_footer, CARD_H - 175, CARD_W, GOLD)
    _center(draw, "AI Facial Geometry Report", f_subtle, CARD_H - 110, CARD_W, TEXT_SOFT)

    img.save(output_path, "PNG")
    return output_path

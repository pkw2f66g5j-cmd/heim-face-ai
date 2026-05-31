# -*- coding: utf-8 -*-
"""Генерация PNG-карточки результата для шаринга."""
import os

from PIL import Image, ImageDraw, ImageFont

BOT_USERNAME = "@heim_face_bot"

# Палитра (тёмный фон, золото/графит/бирюза)
BG_TOP = (18, 14, 24)
BG_BOTTOM = (10, 8, 14)
GOLD = (232, 213, 160)
TEAL = (90, 200, 190)
TEXT = (240, 230, 216)
TEXT_SOFT = (150, 140, 130)
LINE = (60, 50, 66)

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

    # Рамка
    draw.rectangle([40, 40, CARD_W - 40, CARD_H - 40], outline=LINE, width=2)

    # Бренд
    _center(draw, "HEIM FACE", f_brand, 130, CARD_W, GOLD)
    draw.line([(CARD_W / 2 - 120, 230), (CARD_W / 2 + 120, 230)], fill=GOLD, width=3)

    # Score
    _center(draw, "SCORE", f_score_label, 320, CARD_W, TEXT_SOFT)
    score = float(data.get("score", 0))
    _center(draw, f"{score:.2f}", f_score, 380, CARD_W, TEAL)
    _center(draw, "/ 10", f_score_label, 600, CARD_W, TEXT_SOFT)

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
    _center(draw, tier_str, f_tier, 680, CARD_W, GOLD)

    # Сильные стороны
    y = 830
    draw.text((130, y), "СИЛЬНЫЕ СТОРОНЫ", font=f_section, fill=TEXT_SOFT)
    y += 80
    strengths = data.get("strengths", [])[:3]
    for m in strengths:
        draw.ellipse([135, y + 18, 155, y + 38], fill=TEAL)
        draw.text((185, y), m["name"], font=f_item, fill=TEXT)
        y += 80

    # Футер
    draw.line([(130, CARD_H - 200), (CARD_W - 130, CARD_H - 200)], fill=LINE, width=2)
    _center(draw, BOT_USERNAME, f_footer, CARD_H - 150, CARD_W, GOLD)

    img.save(output_path, "PNG")
    return output_path

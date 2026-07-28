# -*- coding: utf-8 -*-
"""Категоризация товаров и извлечение бренда из названия.

В исходной выгрузке нет ни колонки категории, ни колонки бренда — а без них
невозможен почти любой содержательный разрез рынка. Здесь они восстанавливаются
из названия товара набором упорядоченных правил.

Почему правила, а не кластеризация текста: правила проверяемы и объяснимы —
каждый товар можно проследить до конкретного сработавшего шаблона, а покрытие
и ошибки видно напрямую. Для витрины рынка это важнее, чем гибкость.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Не-товары
# --------------------------------------------------------------------------

#: Позиции, которые физическим товаром не являются: страховки, подписки,
#: цифровые коды. Они попадают в выдачу Amazon наравне с товарами, но у них
#: своя экономика (у 91% есть рейтинг, но цена — лишь у 15%), поэтому в
#: рыночных агрегатах они только шумят.
NON_PRODUCT_RULES: list[tuple[str, str]] = [
    ("страховка/гарантия", r"protection plan|asurion|\bwarranty\b.*\bplan\b|geek squad"),
    ("подписка/членство", r"\bmembership\b|\bsubscription\b|on-demand.*access"),
    ("подарочная карта", r"\bgift card\b|\begift\b"),
    ("установка/услуга", r"\binstallation service\b|\bsetup service\b"),
]

# --------------------------------------------------------------------------
# Категории (порядок важен: срабатывает первое правило)
# --------------------------------------------------------------------------

CATEGORY_RULES: list[tuple[str, str]] = [
    # --- высокоточные исключения, которые иначе перехватят широкие правила ---
    # 'Apple Pencil' — аксессуар к планшету, а не карандаш из канцелярии,
    # поэтому это правило обязано стоять выше офисного.
    ("Смартфоны и аксессуары", r"apple pencil|surface (slim )?pen|\bstylus\b"),
    ("Планшеты и электронные книги", r"\b(kindle|kobo|e-?reader|\bipad\b|galaxy tab|idea ?tab|fire hd)\b"),

    # --- узкие товарные классы -------------------------------------------
    ("Картриджи и расходники",
     r"\b(toner|cartridges?|drum unit|printer ribbon)\b"
     r"|\bink\b\s*(/|\s)?\s*(paper|cartridge|value|combo|for|set|tank|refill)"
     r"|\b(pgi|cli|lc|tn|dr)-?\d{2,4}(xl)?\b|durabrite"),
    ("Офис и канцелярия",
     r"\b(calculator|pencils?|markers?|shredder|laminator|laminating|laminate"
     r"|file folders?|envelopes?|labels?|binder|stapler|printer paper|photo paper"
     r"|copy paper|index cards?|whiteboard)\b|\bavery\b"),
    ("Батарейки и питание",
     r"\b(alkaline|aaa?\s+batter|batteries|battery pack|rechargeable batter"
     r"|coin cell|cr2032|\bsla\b|\bagm\b|lifepo4|\d+\s?ah\b)\b"),
    ("Хранение данных",
     r"\b(microsd|sd card|memory card|flash drive|thumb drive|usb drive"
     r"|external (hard|ssd)|nvme|ssd|hard drive|hdd|network attached storage)\b"),
    ("Комплектующие ПК",
     r"\b(ddr[45]|ram\b|motherboard|graphics card|gpu|processor|cpu\b"
     r"|power supply|psu\b|cpu cooler|heatsink|case fan|breadboard)\b"),
    ("Ноутбуки и ПК",
     r"\b(laptop|notebook computer|macbook|chromebook|desktop (computer|pc)"
     r"|all-in-one|workstation|mini pc)\b"),
    ("Мониторы", r"\b(monitor|portable display)\b"),
    ("Телевизоры и видео",
     r"\b(tv\b|television|projector|soundbar|roku|fire tv|streaming stick"
     r"|blu-ray|dvd player)\b"),
    ("Принтеры и сканеры", r"\b(printer|scanner|scanjet|label maker)\b"),
    ("Периферия",
     r"\b(keyboard|mouse\b|mousepad|webcam|docking station|dock\b|kvm\b"
     r"|graphics tablet|trackball|trackpad|numpad|keypad)\b"),
    ("Аудио",
     r"\b(headphones?|earbuds?|earphones?|headset|speakers?|microphone|mic\b"
     r"|amplifier|turntable|record player|vinyl|subwoofer|audio interface"
     r"|mp3 player|\bdac\b)\b"),
    ("Камеры и фото",
     r"\b(camera|camcorder|lens\b|tripod|gimbal|gopro|dslr|mirrorless"
     r"|photo frame|photo storage)\b"),
    ("Оптика", r"\b(telescope|binoculars?|microscope|monocular|spotting scope|riflescope)\b"),
    ("Игры и консоли",
     r"\b(xbox|playstation|ps[45]\b|nintendo|amiibo|game controller"
     r"|gaming console|switch (console|lite|oled)|video game|meta quest"
     r"|oculus|vr headset)\b"),
    ("Умный дом",
     r"\b(alexa|echo dot|smart (bulb|plug|lock|home|thermostat|display)"
     r"|doorbell|security camera|simplisafe|entry sensor|motion sensor"
     r"|alarm system)\b"),
    ("Носимые устройства",
     r"\b(smartwatch|smart watch|fitness tracker|garmin|activity tracker|fitbit)\b"),
    ("Смартфоны и аксессуары",
     r"\b(iphone|smartphone|phone case|screen protector|tablet case|phone holder)\b"),
    ("Рации и радиосвязь",
     r"\b(walkie ?talkie|two-way radio|ham radio|cb radio|baofeng|\bdmr\b"
     r"|amateur radio)\b"),
    ("Телефоны и связь",
     r"\b(cordless phone|answering machine|landline|dect|desk phone)\b"),
    ("Кабели и зарядки",
     r"\b(cable|charger|charging|adapter|power bank|usb-c|hdmi|surge protect"
     r"|extension cord|hub\b|circuit breaker|power strip)\b"),
    ("Сетевое оборудование",
     r"\b(router|modem|wi-?fi (extender|range)|mesh (wifi|system)|ethernet"
     r"|network switch|access point|\bnas\b|\bpoe\b)\b"),
    ("Авто и навигация",
     r"\b(dash ?cam|gps navigator|car (charger|mount|stereo)|backup camera"
     r"|radar detector|marine|fish ?finder|humminbird|helmet)\b"),
    ("Крепления и подставки",
     r"\b(mount|stand\b|bracket|holder|wall plate|\brack\b)\b"),
]

_COMPILED_CATEGORY = [(name, re.compile(pat, re.I)) for name, pat in CATEGORY_RULES]
_COMPILED_NON_PRODUCT = [(name, re.compile(pat, re.I)) for name, pat in NON_PRODUCT_RULES]

UNCLASSIFIED = "Прочее"


def classify_non_product(title) -> str:
    """-> тип не-товара, либо '' если это обычный товар."""
    if pd.isna(title):
        return ""
    t = str(title)
    for name, pat in _COMPILED_NON_PRODUCT:
        if pat.search(t):
            return name
    return ""


def classify_category(title) -> str:
    """-> название категории, либо 'Прочее'."""
    if pd.isna(title):
        return UNCLASSIFIED
    t = str(title)
    for name, pat in _COMPILED_CATEGORY:
        if pat.search(t):
            return name
    return UNCLASSIFIED


# --------------------------------------------------------------------------
# Бренд
# --------------------------------------------------------------------------

#: Бренды, которые записаны в несколько слов или в разном регистре.
#: Ключ — как встречается в названии (в нижнем регистре), значение — канон.
BRAND_ALIASES = {
    "hp": "HP", "hewlett": "HP", "hewlett-packard": "HP",
    "tp-link": "TP-Link", "tplink": "TP-Link",
    "sandisk": "SanDisk", "samsung": "Samsung", "sony": "Sony",
    "acer": "Acer", "asus": "ASUS", "kodak": "Kodak", "nikon": "Nikon",
    "canon": "Canon", "lenovo": "Lenovo", "dell": "Dell", "apple": "Apple",
    "logitech": "Logitech", "corsair": "Corsair", "ugreen": "UGREEN",
    "anker": "Anker", "belkin": "Belkin", "jbl": "JBL", "bose": "Bose",
    "seagate": "Seagate", "crucial": "Crucial", "kingston": "Kingston",
    "western": "Western Digital", "wd": "Western Digital",
    "amazon": "Amazon", "amazonbasics": "Amazon Basics",
    "g.skill": "G.SKILL", "gskill": "G.SKILL",
    "microsoft": "Microsoft", "intel": "Intel", "amd": "AMD",
    "texas": "Texas Instruments", "peak": "Peak Design",
    "melissa": "Melissa & Doug", "e-z": "E-Z Ink", "mighty": "Mighty Max",
    "netgear": "NETGEAR", "razer": "Razer", "epson": "Epson",
    "brother": "Brother", "garmin": "Garmin", "energizer": "Energizer",
    "duracell": "Duracell", "asurion": "Asurion", "lg": "LG",
    "lenovo/thinkpad": "Lenovo",
}

_BRAND_STRIP_RE = re.compile(r"^[\"'“”\s]+|[\"'“”,.:;]+$")
_BRAND_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-\.&]*$")


def extract_brand(title) -> str | float:
    """Первый значимый токен названия как бренд.

    На Amazon название почти всегда начинается с бренда, поэтому эвристика
    работает хорошо; спорные случаи сводятся к канону через `BRAND_ALIASES`.
    Токены, не похожие на бренд (числа, единицы измерения), отбрасываются.
    """
    if pd.isna(title):
        return np.nan
    parts = str(title).strip().split()
    if not parts:
        return np.nan
    tok = _BRAND_STRIP_RE.sub("", parts[0])
    if not tok or not _BRAND_TOKEN_RE.match(tok):
        return np.nan
    if tok.isdigit():
        return np.nan
    low = tok.lower()
    if low in BRAND_ALIASES:
        return BRAND_ALIASES[low]
    # Бренды в названиях часто написаны капсом целиком — приводим к Title-case,
    # но короткие аббревиатуры (RCA, JVC, MSI) оставляем как есть.
    if tok.isupper() and len(tok) > 4:
        return tok.capitalize()
    return tok


def coverage(df: pd.DataFrame, category_col: str = "category") -> dict:
    """Насколько полно правила покрывают ассортимент.

    Покрытие — обязательная метрика для эвристической классификации: без неё
    нельзя понять, описывает ли разрез по категориям рынок или только его часть.
    """
    total = len(df)
    unclassified = int((df[category_col] == UNCLASSIFIED).sum())
    return {
        "total": total,
        "classified": total - unclassified,
        "coverage_pct": round((total - unclassified) / total * 100, 1) if total else 0.0,
        "unclassified": unclassified,
        "n_categories": int(df[category_col].nunique()),
    }


def annotate(df: pd.DataFrame, title_col: str = "title") -> pd.DataFrame:
    """Добавляет к таблице колонки `category`, `non_product_type`, `brand`."""
    out = df.copy()
    titles = out[title_col]
    out["non_product_type"] = titles.map(classify_non_product)
    out["is_non_product"] = out["non_product_type"] != ""
    out["category"] = titles.map(classify_category)
    out.loc[out["is_non_product"], "category"] = "Услуги и подписки"
    out["brand"] = titles.map(extract_brand)
    return out

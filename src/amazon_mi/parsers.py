# -*- coding: utf-8 -*-
"""Парсеры отдельных «грязных» полей исходной выгрузки.

Общий принцип модуля: каждая функция либо возвращает распознанное значение,
либо `NaN`. Ни один парсер не «угадывает» значение по частичному совпадению —
непонятное значение всегда становится пропуском и попадает в отчёт о качестве
данных (`quality.py`), а не в аналитику.

Каждый парсер дополнительно умеет возвращать *причину* отказа — это то, из чего
строится таксономия мусора в отчёте о качестве.
"""

from __future__ import annotations

import re
from urllib.parse import unquote

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Рейтинг
# --------------------------------------------------------------------------

_RATING_RE = re.compile(r"^([\d.]+)\s+out of 5 stars$")


def parse_rating(x) -> float:
    """'4.6 out of 5 stars' -> 4.6"""
    if pd.isna(x):
        return np.nan
    m = _RATING_RE.match(str(x).strip())
    if not m:
        return np.nan
    val = float(m.group(1))
    # Рейтинг вне [0, 5] — признак битой строки, а не редкого случая
    return val if 0.0 <= val <= 5.0 else np.nan


# --------------------------------------------------------------------------
# Количество отзывов
# --------------------------------------------------------------------------


def parse_reviews(x) -> float:
    """'2,457' -> 2457, '8,65,598' (индийская группировка) -> 865598.

    Разделители разрядов просто удаляются: от конвенции группировки сами цифры
    не меняются, поэтому оба формата дают одинаково корректный результат.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "")
    if not s.isdigit():
        return np.nan
    return float(s)


# --------------------------------------------------------------------------
# Цены
# --------------------------------------------------------------------------


def parse_price_plain(x) -> float:
    """'89.68' -> 89.68 и '2,299.00' -> 2299.0.

    Разделитель тысяч здесь принципиален: в колонке `Current/discounted_price`
    1 192 значения записаны как '2,299.00'. Наивный `float(x)` на них падает,
    и все они молча исчезают из анализа — а это ровно дорогой сегмент
    (от $1 019 до $4 699), из-за чего средняя цена оказывается занижена
    примерно на 31%.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "").replace("$", "")
    try:
        val = float(s)
    except ValueError:
        return np.nan
    return val if val > 0 else np.nan


def parse_price_dollar(x) -> float:
    """'$159.00' -> 159.0 ; 'No Discount' и прочее -> NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s.startswith("$"):
        return np.nan
    return parse_price_plain(s[1:])


_VARIANT_PRICE_RE = re.compile(r"\$([\d,]+\.?\d*)")


def parse_price_variant(x) -> float:
    """'basic variant price: $162.24' -> 162.24.

    'basic variant price: 2.4GHz' или '... 16 Count (Pack of 1)' -> NaN:
    в этой колонке лежит описание варианта товара, и цена там есть не всегда.
    """
    if pd.isna(x):
        return np.nan
    m = _VARIANT_PRICE_RE.search(str(x))
    if not m:
        return np.nan
    return parse_price_plain(m.group(1))


# --------------------------------------------------------------------------
# Спрос: «куплено за последний месяц»
# --------------------------------------------------------------------------

_BOUGHT_RE = re.compile(r"^([\d,]+)(K|M)?\+\s+bought in past month$")
_MULT = {"K": 1_000, "M": 1_000_000}

#: Значения, которые скрапер положил в колонку спроса, но которые спросом не являются.
#: Классифицируем их, чтобы показать в отчёте, *чем именно* засорено поле.
JUNK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("нет предложения (buy box)", re.compile(r"^No featured offers available$")),
    ("блок «другие продавцы»", re.compile(r"^More Buying Choices$")),
    ("обрывок ценника", re.compile(r"^(Typical:|List:|List Price:|Was:|New Price:)$")),
    ("цена за единицу", re.compile(r"^\(\$")),
    ("возрастной рейтинг ESRB", re.compile(r"^ESRB Rating:")),
    ("маркетинговый текст", re.compile(r"(Alexa|small business brands)", re.I)),
]


def classify_bought_junk(x) -> str:
    """Возвращает тип мусора в поле спроса ('' — если значение валидное)."""
    if pd.isna(x):
        return "пропуск"
    s = str(x).strip()
    if _BOUGHT_RE.match(s):
        return ""
    for label, pat in JUNK_PATTERNS:
        if pat.search(s):
            return label
    return "прочий мусор"


def parse_bought_last_month(x) -> float:
    """'300+ bought in past month' -> 300 ; '6K+ bought in past month' -> 6000.

    Важно: это *нижняя граница* округлённого диапазона, который Amazon
    показывает сам, а не точное число продаж.
    """
    if pd.isna(x):
        return np.nan
    m = _BOUGHT_RE.match(str(x).strip())
    if not m:
        return np.nan
    num = float(m.group(1).replace(",", ""))
    return num * _MULT.get(m.group(2), 1)


# --------------------------------------------------------------------------
# Купоны
# --------------------------------------------------------------------------

_COUPON_PCT_RE = re.compile(r"Save\s+(\d+(?:\.\d+)?)%")
_COUPON_AMT_RE = re.compile(r"Save\s+\$([\d,]+\.?\d*)")


def parse_coupon(x) -> tuple[bool, float, float]:
    """-> (есть_купон, скидка_%, скидка_$)."""
    if pd.isna(x) or str(x).strip() == "No Coupon":
        return False, np.nan, np.nan
    s = str(x)
    pct = _COUPON_PCT_RE.search(s)
    amt = _COUPON_AMT_RE.search(s)
    return (
        True,
        float(pct.group(1)) if pct else np.nan,
        parse_price_plain(amt.group(1)) if amt else np.nan,
    )


# --------------------------------------------------------------------------
# Значки (колонка is_best_seller)
# --------------------------------------------------------------------------

_SAVE_BADGE_RE = re.compile(r"^Save\s+\d+%$")


def parse_badge(x) -> str:
    """Нормализует колонку значков.

    Колонка смешивает три разных сущности: собственно значки ('Best Seller',
    "Amazon's"), таймер акции ('Limited time deal', 'Ends in') и — в 336 строках —
    метку скидки ('Save 30%'), которая к значкам вообще не относится и в
    бинарный флаг «есть значок» попадать не должна.
    """
    if pd.isna(x):
        return "none"
    s = str(x).strip()
    if s in ("No Badge", ""):
        return "none"
    if s == "Best Seller":
        return "best_seller"
    if s.startswith("Amazon"):
        return "amazons_choice"
    if s in ("Limited time deal", "Ends in"):
        return "limited_deal"
    if _SAVE_BADGE_RE.match(s):
        return "discount_label"      # не значок — метка скидки, затесавшаяся в колонку
    return "other"


# --------------------------------------------------------------------------
# Значки устойчивого развития
# --------------------------------------------------------------------------

#: Настоящие бейджи программы Climate Pledge Friendly.
SUSTAINABILITY_BADGES = {
    "Carbon impact",
    "Energy efficiency",
    "Manufacturing practices",
    "Forestry practices",
    "Recycled materials",
    "Safer chemicals",
}

_PLUS_MORE_RE = re.compile(r"\s*\+\d+\s+more$")


def parse_sustainability(x) -> tuple[bool, str]:
    """-> (это_настоящий_эко-бейдж, нормализованное_значение).

    Колонка `sustainability_badges` смешана: из 3 408 непустых значений
    1 955 — это вообще не про экологию ('Small Business', 'Works with Alexa',
    'Alexa Built-in'). Трактовать `notna()` как «есть эко-значок» — ошибка,
    которая завышает покрытие более чем вдвое.
    """
    if pd.isna(x):
        return False, ""
    base = _PLUS_MORE_RE.sub("", str(x).strip())
    return base in SUSTAINABILITY_BADGES, base


# --------------------------------------------------------------------------
# Доставка
# --------------------------------------------------------------------------

_DELIVERY_FEE_RE = re.compile(r"^\$([\d,]+\.?\d*)\s+delivery")
_MIN_ORDER_RE = re.compile(r"on \$(\d+) of items")


def parse_delivery(x) -> tuple[bool, float, float]:
    """-> (бесплатная_доставка, порог_бесплатной_доставки_$, платная_доставка_$).

    Поле `delivery_details` в исходном проекте не разбиралось вовсе, хотя
    несёт отдельное коммерческое измерение: 7 542 строки обещают бесплатную
    доставку, 1 736 — только при заказе от определённой суммы.
    """
    if pd.isna(x):
        return False, np.nan, np.nan
    s = str(x).strip()
    fee = _DELIVERY_FEE_RE.match(s)
    if fee:
        return False, np.nan, parse_price_plain(fee.group(1))
    if "FREE delivery" in s:
        thr = _MIN_ORDER_RE.search(s)
        return True, float(thr.group(1)) if thr else 0.0, np.nan
    return False, np.nan, np.nan


def parse_fastest_delivery(x) -> bool:
    """Есть ли опция ускоренной доставки ('Or fastest delivery ...')."""
    return bool(pd.notna(x) and "fastest delivery" in str(x))


# --------------------------------------------------------------------------
# Идентификатор товара (ASIN)
# --------------------------------------------------------------------------

_ASIN_RE = re.compile(r"/dp/([A-Z0-9]{10})")


def extract_asin(url) -> str | float:
    """Достаёт ASIN из ссылки на товар, включая спонсорские редиректы.

    Органическая ссылка содержит '/dp/B0D1XD1ZV3' напрямую. Спонсорская же
    выглядит как '/sspa/click?...&url=%2F...%2Fdp%2FB0DNZB7TQG%2Fref%3D...' —
    там ASIN спрятан в URL-кодированном параметре.

    Это не мелочь: простая регулярка по '/dp/' находит ASIN в 35 114 строках,
    а с декодированием — в 40 606. Все 5 492 «потерянные» строки спонсорские,
    поэтому дедупликация без декодирования выбросила бы именно рекламные
    размещения и исказила любое сравнение «спонсорское против органического».
    """
    if pd.isna(url):
        return np.nan
    s = str(url)
    m = _ASIN_RE.search(s)
    if m:
        return m.group(1)
    m = _ASIN_RE.search(unquote(unquote(s)))    # двойное декодирование
    return m.group(1) if m else np.nan

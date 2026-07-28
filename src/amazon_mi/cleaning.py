# -*- coding: utf-8 -*-
"""Пайплайн очистки: сырой CSV -> таблица размещений -> таблица товаров.

Ключевое отличие от «обычной» очистки этого датасета — два уровня на выходе:

* `listings` — одна строка = одно размещение товара в выдаче (после парсинга
  полей). На этом уровне корректно считать всё, что относится к самой выдаче:
  как часто товар показывается, какая доля показов рекламная.
* `products` — одна строка = один товар (ASIN). На этом уровне считается вся
  рыночная аналитика.

Смешивать их нельзя. В выгрузке 40 606 строк с ASIN соответствуют всего
8 630 товарам, причём 1% товаров занимает 36.7% всех строк. Любая статистика,
посчитанная по строкам, — это статистика, взвешенная по частоте скрапинга,
а не по товарам.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import parsers as P
from . import taxonomy


def load_raw(path) -> pd.DataFrame:
    """Читает исходный CSV целиком как строки — парсинг делаем сами."""
    return pd.read_csv(path, dtype=str)


# --------------------------------------------------------------------------
# Уровень 1: размещения
# --------------------------------------------------------------------------


def build_listings(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Парсит все поля. Возвращает (таблица размещений, статистика шагов)."""
    stats: dict = {"rows_raw": len(raw)}

    df = raw.drop_duplicates().reset_index(drop=True)
    stats["exact_duplicates_dropped"] = len(raw) - len(df)

    out = pd.DataFrame(index=df.index)
    out["title"] = df["Title"]

    # --- идентификатор -----------------------------------------------------
    out["asin"] = df["product_url"].map(P.extract_asin)
    stats["rows_with_asin"] = int(out["asin"].notna().sum())
    stats["unique_asins"] = int(out["asin"].nunique())

    # --- рейтинг и отзывы --------------------------------------------------
    out["rating"] = df["Rating"].map(P.parse_rating)
    out["num_reviews"] = df["Number_of_reviews"].map(P.parse_reviews)

    # --- спрос -------------------------------------------------------------
    out["bought_last_month"] = df["bought_in_last_month"].map(P.parse_bought_last_month)
    out["_bought_junk_type"] = df["bought_in_last_month"].map(P.classify_bought_junk)
    stats["bought_parsed"] = int(out["bought_last_month"].notna().sum())
    stats["bought_junk"] = int(
        (~out["_bought_junk_type"].isin(["", "пропуск"])).sum()
    )

    # --- цены --------------------------------------------------------------
    price_current = df["Current/discounted_price"].map(P.parse_price_plain)
    price_variant = df["Price_on_variant"].map(P.parse_price_variant)
    price_listed = df["Listed_price"].map(P.parse_price_dollar)

    # Сколько цен потерял бы наивный float() без учёта разделителя тысяч
    naive = pd.to_numeric(df["Current/discounted_price"], errors="coerce")
    stats["prices_saved_by_thousand_sep"] = int(
        (price_current.notna() & naive.isna()).sum()
    )

    # Три частично заполненных источника цены сводятся с явным приоритетом:
    # фактическая цена -> цена варианта -> «листовая» цена.
    out["price"] = price_current.combine_first(price_variant).combine_first(price_listed)
    out["price_source"] = np.select(
        [price_current.notna(), price_variant.notna(), price_listed.notna()],
        ["current", "variant", "listed"],
        default="none",
    )
    out["listed_price"] = price_listed

    has_discount = (
        price_listed.notna() & price_current.notna() & (price_listed > price_current)
    )
    out["discount_pct"] = np.where(
        has_discount, (price_listed - price_current) / price_listed * 100, np.nan
    )

    # --- значки ------------------------------------------------------------
    badge = df["is_best_seller"].map(P.parse_badge)
    out["badge"] = badge
    out["is_best_seller"] = badge == "best_seller"
    out["is_amazons_choice"] = badge == "amazons_choice"
    out["has_limited_deal"] = badge == "limited_deal"
    stats["discount_labels_in_badge_column"] = int((badge == "discount_label").sum())

    # --- реклама -----------------------------------------------------------
    out["is_sponsored"] = df["is_sponsored"].eq("Sponsored")

    # --- купоны ------------------------------------------------------------
    coupon = df["is_couponed"].map(P.parse_coupon)
    out["has_coupon"] = coupon.map(lambda t: t[0])
    out["coupon_pct"] = coupon.map(lambda t: t[1])
    out["coupon_amount"] = coupon.map(lambda t: t[2])

    # --- buy box -----------------------------------------------------------
    # Пустая ячейка тут означает не «нет данных», а «нет предложения»:
    # у всех 4 571 строки с 'No featured offers available' в колонке спроса
    # buy box пуст ровно в 100% случаев.
    out["has_buy_box"] = df["buy_box_availability"].eq("Add to cart")

    # --- устойчивое развитие ----------------------------------------------
    sus = df["sustainability_badges"].map(P.parse_sustainability)
    out["has_sustainability_badge"] = sus.map(lambda t: t[0])
    out["badge_label_raw"] = sus.map(lambda t: t[1])
    out["is_small_business"] = out["badge_label_raw"].eq("Small Business")
    out["works_with_alexa"] = out["badge_label_raw"].isin(
        ["Works with Alexa", "Alexa Built-in"]
    )
    stats["fake_sustainability_badges"] = int(
        df["sustainability_badges"].notna().sum()
        - out["has_sustainability_badge"].sum()
    )

    # --- доставка ----------------------------------------------------------
    deliv = df["delivery_details"].map(P.parse_delivery)
    out["free_delivery"] = deliv.map(lambda t: t[0])
    out["free_delivery_threshold"] = deliv.map(lambda t: t[1])
    out["delivery_fee"] = deliv.map(lambda t: t[2])
    out["has_fast_delivery"] = df["delivery_details"].map(P.parse_fastest_delivery)

    # --- время сбора -------------------------------------------------------
    ts = pd.to_datetime(df["collected_at"], format="%d-%m-%Y %H:%M", errors="coerce")
    out["collected_at"] = ts
    out["collected_date"] = ts.dt.normalize()

    # --- категории и бренд -------------------------------------------------
    out = taxonomy.annotate(out, title_col="title")

    stats["rows_listings"] = len(out)
    return out, stats


# --------------------------------------------------------------------------
# Уровень 2: товары
# --------------------------------------------------------------------------


def _mode_or_nan(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if len(m) else np.nan


def build_products(listings: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Схлопывает размещения до уровня товара (ASIN).

    Правила агрегации выбраны так, чтобы повторные показы одного товара не
    искажали его характеристики:

    * цена/рейтинг/отзывы/спрос — медиана по наблюдениям (устойчива к тому,
      что в разные дни скрапер видел разные варианты товара);
    * флаги значков — `max` (значок был хотя бы раз);
    * `is_sponsored` — доля рекламных показов, а не булев флаг: товар может
      показываться и органически, и как реклама;
    * `n_listings` — сколько раз товар вообще встретился в выдаче. Это
      самостоятельная метрика видимости, а не технический артефакт.
    """
    stats: dict = {}

    df = listings[listings["asin"].notna()].copy()
    stats["rows_without_asin_dropped"] = int(listings["asin"].isna().sum())

    num_median = ["price", "listed_price", "rating", "num_reviews",
                  "bought_last_month", "discount_pct", "coupon_pct",
                  "coupon_amount", "delivery_fee", "free_delivery_threshold"]
    flags_max = ["is_best_seller", "is_amazons_choice", "has_limited_deal",
                 "has_coupon", "has_buy_box", "has_sustainability_badge",
                 "is_small_business", "works_with_alexa", "free_delivery",
                 "has_fast_delivery", "is_non_product"]

    agg: dict = {c: (c, "median") for c in num_median}
    agg |= {c: (c, "max") for c in flags_max}
    agg |= {
        "title": ("title", "first"),
        "category": ("category", _mode_or_nan),
        "brand": ("brand", _mode_or_nan),
        "non_product_type": ("non_product_type", _mode_or_nan),
        "badge_label_raw": ("badge_label_raw", _mode_or_nan),
        "sponsored_share": ("is_sponsored", "mean"),
        "n_listings": ("asin", "size"),
        "n_days_seen": ("collected_date", "nunique"),
        "first_seen": ("collected_at", "min"),
        "last_seen": ("collected_at", "max"),
        "price_spread_pct": ("price", lambda s: _spread_pct(s)),
    }

    prod = df.groupby("asin", sort=False).agg(**agg).reset_index()
    prod["is_sponsored"] = prod["sponsored_share"] > 0

    stats["n_products"] = len(prod)
    stats["mean_listings_per_product"] = round(float(prod["n_listings"].mean()), 2)
    stats["top1pct_row_share"] = round(
        float(
            prod.nlargest(max(1, len(prod) // 100), "n_listings")["n_listings"].sum()
            / prod["n_listings"].sum() * 100
        ),
        1,
    )

    # Товары без рейтинга, без отзывов и без цены — почти наверняка служебные
    # строки выдачи, а не карточки товара. Помечаем, но не удаляем.
    prod["likely_invalid"] = (
        prod["rating"].isna() & prod["num_reviews"].isna() & prod["price"].isna()
    )
    stats["likely_invalid"] = int(prod["likely_invalid"].sum())

    return prod, stats


def _spread_pct(s: pd.Series) -> float:
    """Разброс цены товара между наблюдениями, % от минимума."""
    s = s.dropna()
    if len(s) < 2 or s.min() <= 0:
        return np.nan
    return float((s.max() - s.min()) / s.min() * 100)


# --------------------------------------------------------------------------
# Аналитический срез
# --------------------------------------------------------------------------


def analytic_frame(products: pd.DataFrame) -> pd.DataFrame:
    """Подмножество товаров, пригодное для рыночного анализа.

    Исключаются служебные строки и не-товары (страховки, подписки): у них
    другая экономика, и в рыночных агрегатах они только шумят.
    """
    mask = ~products["likely_invalid"] & ~products["is_non_product"]
    return products[mask].copy()

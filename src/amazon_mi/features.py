# -*- coding: utf-8 -*-
"""Производные признаки для анализа и моделирования."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_BANDS = [0, 10, 25, 50, 100, 200, 500, np.inf]
PRICE_BAND_LABELS = ["<$10", "$10–25", "$25–50", "$50–100",
                     "$100–200", "$200–500", "$500+"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет аналитические признаки к таблице товаров."""
    out = df.copy()

    out["log_price"] = np.log10(out["price"].where(out["price"] > 0))
    out["log_reviews"] = np.log10(out["num_reviews"].fillna(0) + 1)
    out["log_demand"] = np.log10(out["bought_last_month"].where(out["bought_last_month"] > 0))

    out["price_band"] = pd.cut(out["price"], bins=PRICE_BANDS,
                               labels=PRICE_BAND_LABELS, right=False)

    # Оценка месячной выручки. `bought_last_month` — нижняя граница диапазона,
    # который показывает сам Amazon ('300+'), поэтому это заведомо оценка снизу
    # и годится для сравнения сегментов между собой, но не как абсолютная цифра.
    out["est_revenue"] = out["bought_last_month"] * out["price"]

    # Отзывов на единицу спроса: косвенная мера «зрелости» карточки —
    # много отзывов при слабом текущем спросе означает старый популярный товар.
    out["reviews_per_demand"] = np.where(
        out["bought_last_month"].gt(0),
        out["num_reviews"] / out["bought_last_month"],
        np.nan,
    )

    out["has_discount"] = out["discount_pct"].notna() & (out["discount_pct"] > 0)
    out["discount_pct_filled"] = out["discount_pct"].fillna(0.0)

    # Признаки названия
    title = out["title"].fillna("")
    out["title_len"] = title.str.len()
    out["title_words"] = title.str.split().str.len()
    out["title_has_digits"] = title.str.contains(r"\d", regex=True)
    out["title_caps_ratio"] = np.where(
        out["title_len"] > 0,
        title.map(lambda s: sum(c.isupper() for c in s)) / out["title_len"].replace(0, np.nan),
        np.nan,
    )

    return out


def empirical_bayes_rating(df: pd.DataFrame,
                           rating_col: str = "rating",
                           n_col: str = "num_reviews") -> pd.Series:
    """Сглаженный рейтинг: усадка к среднему по рынку с весом по числу отзывов.

    Сырой рейтинг несопоставим между товарами: 5.0 при трёх отзывах и 4.6 при
    35 000 — величины разной надёжности. Оценка вида

        r* = (C * m + n * r) / (C + n)

    где m — средний рейтинг рынка, C — «априорное» число отзывов, подтягивает
    малоизученные товары к рынку и почти не трогает популярные.

    C оценивается методом моментов: C = m * (1 - m) / var(r) - 1 на шкале [0, 1].
    """
    r = df[rating_col]
    n = df[n_col].fillna(0)

    valid = r.notna()
    if valid.sum() < 2:
        return r.copy()

    m = float(r[valid].mean())
    p = r[valid] / 5.0
    p_mean, p_var = float(p.mean()), float(p.var(ddof=1))

    if p_var <= 0:
        prior_n = 50.0
    else:
        prior_n = max(1.0, p_mean * (1 - p_mean) / p_var - 1)

    return (prior_n * m + n * r.fillna(m)) / (prior_n + n)


def herfindahl(shares: pd.Series) -> float:
    """Индекс Херфиндаля–Хиршмана (HHI) по долям рынка, в пунктах 0–10 000."""
    s = shares.dropna()
    total = s.sum()
    if total <= 0:
        return np.nan
    return float(((s / total * 100) ** 2).sum())

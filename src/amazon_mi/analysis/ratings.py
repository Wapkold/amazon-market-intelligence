# -*- coding: utf-8 -*-
"""Надёжность рейтингов: сглаживание по числу отзывов.

Сырая звёздная оценка несопоставима между товарами. 5.0 при трёх отзывах и
4.6 при 35 000 — это оценки принципиально разной надёжности, но в таблице они
стоят рядом, и любой рейтинг «топ по оценке» заполняется товарами с двумя
отзывами. Эмпирический Байес решает это: оценка подтягивается к среднему по
рынку тем сильнее, чем меньше отзывов (см. `features.empirical_bayes_rating`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import viz


def rating_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Насколько сырой рейтинг «завышен» в разрезе числа отзывов."""
    d = df.dropna(subset=["rating", "rating_shrunk"]).copy()
    d["review_bucket"] = pd.cut(
        d["num_reviews"].fillna(0),
        [-1, 0, 10, 50, 200, 1000, 10_000, np.inf],
        labels=["нет отзывов", "1–10", "11–50", "51–200", "201–1000",
                "1–10 тыс.", "10 тыс.+"],
    )
    g = d.groupby("review_bucket", observed=True)
    tbl = pd.DataFrame({
        "Товаров": g.size(),
        "Сырой рейтинг": g["rating"].mean().round(3),
        "Сглаженный рейтинг": g["rating_shrunk"].mean().round(3),
        "Медианный спрос": g["bought_last_month"].median(),
        "Медианная цена, $": g["price"].median().round(2),
    })
    tbl["Поправка"] = (tbl["Сглаженный рейтинг"] - tbl["Сырой рейтинг"]).round(3)
    return tbl


def top_by_rating(df: pd.DataFrame, n: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Топ товаров по сырому и по сглаженному рейтингу — чтобы увидеть разницу."""
    d = df.dropna(subset=["rating", "rating_shrunk"]).copy()
    cols = {"title": "Товар", "brand": "Бренд", "rating": "Рейтинг",
            "num_reviews": "Отзывов", "rating_shrunk": "Сглаженный",
            "price": "Цена, $"}

    raw = (d.sort_values(["rating", "num_reviews"], ascending=[False, False])
             .head(n)[list(cols)].rename(columns=cols))
    shr = (d.sort_values("rating_shrunk", ascending=False)
             .head(n)[list(cols)].rename(columns=cols))
    for t in (raw, shr):
        t["Товар"] = t["Товар"].str.slice(0, 62) + "…"
        t["Сглаженный"] = t["Сглаженный"].round(3)
    return raw, shr


def outliers(df: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Товары, чей рейтинг сильнее всего расходится со средним по их категории."""
    d = df.dropna(subset=["rating_shrunk", "category"]).copy()
    cat_mean = d.groupby("category", observed=True)["rating_shrunk"].transform("mean")
    d["отклонение"] = (d["rating_shrunk"] - cat_mean).round(3)
    d = d[d["num_reviews"].fillna(0) >= 50]

    cols = {"title": "Товар", "category": "Категория", "rating": "Рейтинг",
            "num_reviews": "Отзывов", "отклонение": "Откл. от категории"}
    best = d.nlargest(n, "отклонение")[list(cols)].rename(columns=cols)
    worst = d.nsmallest(n, "отклонение")[list(cols)].rename(columns=cols)
    for t in (best, worst):
        t["Товар"] = t["Товар"].str.slice(0, 58) + "…"
    return best, worst


def plot_shrinkage(df: pd.DataFrame, name: str = "10_rating_shrinkage"):
    d = df.dropna(subset=["rating", "rating_shrunk", "num_reviews"])
    if d.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    ax = axes[0]
    ax.scatter(d["num_reviews"] + 1, d["rating"], s=6, alpha=0.12,
               color=viz.MUTED, edgecolors="none", label="сырой рейтинг")
    ax.scatter(d["num_reviews"] + 1, d["rating_shrunk"], s=6, alpha=0.18,
               color=viz.ACCENT, edgecolors="none", label="после сглаживания")
    ax.set_xscale("log")
    ax.set_xlabel("Число отзывов + 1 (лог. шкала)")
    ax.set_ylabel("Рейтинг")
    ax.set_title("Сглаживание убирает «идеальные» оценки у товаров без отзывов")
    leg = ax.legend(loc="lower right", markerscale=3)
    for h in leg.legend_handles:
        h.set_alpha(1)
    viz.clean_axes(ax)

    ax = axes[1]
    bins = np.linspace(3.0, 5.0, 41)
    ax.hist(d["rating"], bins=bins, color=viz.MUTED, alpha=0.75, label="сырой")
    ax.hist(d["rating_shrunk"], bins=bins, color=viz.ACCENT, alpha=0.75, label="сглаженный")
    ax.set_xlabel("Рейтинг")
    ax.set_ylabel("Товаров")
    ax.set_title("Распределение рейтингов до и после")
    ax.legend()
    viz.clean_axes(ax)

    fig.tight_layout()
    return viz.save(fig, name)

# -*- coding: utf-8 -*-
"""Аудит качества данных: сколько значений реально пригодно к анализу.

Обычная проверка `df.isnull().sum()` показывает только заполненность. Здесь
считается более честная метрика — *валидность*: доля значений, которые не
просто присутствуют, но и распознаются парсером. Разница между этими двумя
величинами и есть скрытый мусор скрапинга.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import parsers as P
from . import viz


def column_scorecard(raw: pd.DataFrame) -> pd.DataFrame:
    """Таблица «заполнено / валидно / мусор» по каждому проблемному полю."""
    n = len(raw)
    checks: list[tuple[str, pd.Series, pd.Series]] = [
        ("Rating", raw["Rating"], raw["Rating"].map(P.parse_rating)),
        ("Number_of_reviews", raw["Number_of_reviews"], raw["Number_of_reviews"].map(P.parse_reviews)),
        ("bought_in_last_month", raw["bought_in_last_month"], raw["bought_in_last_month"].map(P.parse_bought_last_month)),
        ("Current/discounted_price", raw["Current/discounted_price"], raw["Current/discounted_price"].map(P.parse_price_plain)),
        ("Price_on_variant", raw["Price_on_variant"], raw["Price_on_variant"].map(P.parse_price_variant)),
        ("Listed_price", raw["Listed_price"], raw["Listed_price"].map(P.parse_price_dollar)),
        ("product_url (ASIN)", raw["product_url"], raw["product_url"].map(P.extract_asin)),
        ("delivery_details", raw["delivery_details"], raw["delivery_details"].map(lambda x: np.nan if pd.isna(x) else (1.0 if P.parse_delivery(x)[0] or P.parse_fastest_delivery(x) or "Delivery" in str(x) else np.nan))),
    ]

    rows = []
    for name, orig, parsed in checks:
        filled = int(orig.notna().sum())
        valid = int(parsed.notna().sum())
        rows.append({
            "Поле": name,
            "Заполнено": filled,
            "Заполнено, %": round(filled / n * 100, 1),
            "Валидно": valid,
            "Валидно, %": round(valid / n * 100, 1),
            "Мусор (заполнено, но не распознано)": filled - valid,
        })

    # Значковые колонки: валидность = доля значений, которые действительно
    # являются тем, чем колонка называется.
    badge = raw["is_best_seller"].map(P.parse_badge)
    rows.append({
        "Поле": "is_best_seller (значки)",
        "Заполнено": n,
        "Заполнено, %": 100.0,
        "Валидно": int((badge != "discount_label").sum()),
        "Валидно, %": round((badge != "discount_label").mean() * 100, 1),
        "Мусор (заполнено, но не распознано)": int((badge == "discount_label").sum()),
    })

    sus_filled = int(raw["sustainability_badges"].notna().sum())
    sus_valid = int(raw["sustainability_badges"].map(lambda x: P.parse_sustainability(x)[0]).sum())
    rows.append({
        "Поле": "sustainability_badges",
        "Заполнено": sus_filled,
        "Заполнено, %": round(sus_filled / n * 100, 1),
        "Валидно": sus_valid,
        "Валидно, %": round(sus_valid / n * 100, 1),
        "Мусор (заполнено, но не распознано)": sus_filled - sus_valid,
    })

    return pd.DataFrame(rows).sort_values("Мусор (заполнено, но не распознано)", ascending=False)


def bought_junk_taxonomy(raw: pd.DataFrame) -> pd.DataFrame:
    """Из чего состоит мусор в колонке спроса.

    Это самое загрязнённое поле датасета, и загрязнение неслучайно: скрапер
    подставлял сюда текст соседнего элемента вёрстки. Разбор по типам
    показывает, что 4 571 значение — это на самом деле информативный сигнал
    «у товара нет buy box», а не отсутствие данных.
    """
    types = raw["bought_in_last_month"].map(P.classify_bought_junk)
    junk = types[~types.isin(["", "пропуск"])]
    out = junk.value_counts().rename_axis("Тип мусора").reset_index(name="Строк")
    out["Доля непустых, %"] = (
        out["Строк"] / raw["bought_in_last_month"].notna().sum() * 100
    ).round(1)
    return out


def entity_resolution_impact(listings: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Сравнение ключевых метрик до и после схлопывания до уровня товара.

    Главная таблица проекта: показывает, что построчный расчёт по этому
    датасету — не «примерно то же самое», а систематическое искажение.
    """
    lst = listings[listings["asin"].notna()]
    prod = products

    def pct(x):
        return round(float(x) * 100, 2)

    rows = [
        ("Наблюдений", len(lst), len(prod)),
        ("Доля спонсорских, %", pct(lst["is_sponsored"].mean()), pct(prod["is_sponsored"].mean())),
        ("Средняя цена, $", round(float(lst["price"].mean()), 2), round(float(prod["price"].mean()), 2)),
        ("Медианная цена, $", round(float(lst["price"].median()), 2), round(float(prod["price"].median()), 2)),
        ("Среднее число отзывов", round(float(lst["num_reviews"].mean()), 0), round(float(prod["num_reviews"].mean()), 0)),
        ("Средний рейтинг", round(float(lst["rating"].mean()), 3), round(float(prod["rating"].mean()), 3)),
        ("Доля с купоном, %", pct(lst["has_coupon"].mean()), pct(prod["has_coupon"].mean())),
        ("Доля Best Seller, %", pct(lst["is_best_seller"].mean()), pct(prod["is_best_seller"].mean())),
        ("Доля с buy box, %", pct(lst["has_buy_box"].mean()), pct(prod["has_buy_box"].mean())),
    ]

    df = pd.DataFrame(rows, columns=["Метрика", "По строкам выдачи", "По товарам"])
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Расхождение, %"] = (
            (df["По строкам выдачи"] - df["По товарам"]) / df["По товарам"] * 100
        ).replace([np.inf, -np.inf], np.nan).round(1)
    return df


def duplication_profile(products: pd.DataFrame) -> pd.DataFrame:
    """Распределение числа показов на товар — насколько неравномерен скрапинг."""
    n = products["n_listings"]
    total = n.sum()
    buckets = [(1, 1, "1 показ"), (2, 3, "2–3"), (4, 10, "4–10"),
               (11, 50, "11–50"), (51, 200, "51–200"), (201, 10**9, "200+")]
    rows = []
    for lo, hi, label in buckets:
        m = (n >= lo) & (n <= hi)
        rows.append({
            "Показов в выдаче": label,
            "Товаров": int(m.sum()),
            "Товаров, %": round(m.mean() * 100, 1),
            "Строк": int(n[m].sum()),
            "Строк, %": round(n[m].sum() / total * 100, 1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Графики
# --------------------------------------------------------------------------


def plot_scorecard(tbl: pd.DataFrame, n_rows: int, name: str = "01_data_quality"):
    """Заполненность против валидности: где заполненность вводит в заблуждение."""
    t = tbl.sort_values("Валидно, %").copy()
    y = np.arange(len(t))

    fig, ax = plt.subplots(figsize=(10.5, 0.5 * len(t) + 2.8))
    ax.barh(y, t["Заполнено, %"], height=0.66, color=viz.GRID,
            label="заполнено (непустая ячейка)")
    ax.barh(y, t["Валидно, %"], height=0.66, color=viz.ACCENT,
            label="валидно (значение распознано)")

    for i, (filled, valid) in enumerate(zip(t["Заполнено, %"], t["Валидно, %"])):
        gap = filled - valid
        if gap > 1.0:
            ax.text(filled + 1, i, f"−{gap:.0f} п.п. мусора",
                    va="center", fontsize=8.5, color=viz.ACCENT_4)

    ax.set_yticks(y)
    ax.set_yticklabels(t["Поле"], fontsize=9)
    ax.set_xlabel("% строк")
    ax.set_xlim(0, 118)
    ax.set_title(f"Заполненность ≠ пригодность к анализу\n{n_rows:,} строк исходной выгрузки"
                 .replace(",", " "))
    ax.legend(loc="lower right")
    viz.clean_axes(ax)
    return viz.save(fig, name)


def plot_entity_impact(impact: pd.DataFrame, dup: pd.DataFrame,
                       name: str = "02_entity_resolution"):
    """Центральный график проекта: во что обходится счёт по строкам."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # --- слева: расхождение метрик ---------------------------------------
    ax = axes[0]
    m = impact[impact["Метрика"] != "Наблюдений"].copy()
    m = m.dropna(subset=["Расхождение, %"])
    m = m.reindex(m["Расхождение, %"].abs().sort_values().index)
    y = np.arange(len(m))
    colors = [viz.ACCENT_4 if abs(v) > 20 else viz.ACCENT for v in m["Расхождение, %"]]
    ax.barh(y, m["Расхождение, %"], color=colors, height=0.66)
    for i, v in enumerate(m["Расхождение, %"]):
        ax.text(v + (2 if v >= 0 else -2), i, f"{v:+.0f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=viz.INK)
    ax.axvline(0, color=viz.INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(m["Метрика"], fontsize=9)
    ax.set_xlabel("Ошибка построчного счёта относительно счёта по товарам, %")
    ax.set_title("Насколько врёт статистика «по строкам выдачи»")
    pad = max(abs(m["Расхождение, %"]).max() * 0.35, 12)
    ax.set_xlim(m["Расхождение, %"].min() - pad, m["Расхождение, %"].max() + pad)
    viz.clean_axes(ax)

    # --- справа: кто занимает строки --------------------------------------
    ax = axes[1]
    x = np.arange(len(dup))
    ax.bar(x - 0.19, dup["Товаров, %"], width=0.38, color=viz.MUTED,
           label="доля товаров")
    ax.bar(x + 0.19, dup["Строк, %"], width=0.38, color=viz.ACCENT,
           label="доля строк выдачи")
    ax.set_xticks(x)
    ax.set_xticklabels(dup["Показов в выдаче"], fontsize=9)
    ax.set_xlabel("Сколько раз товар встретился в выдаче")
    ax.set_ylabel("%")
    ax.set_title("Немногие товары занимают большую часть строк")
    ax.legend()
    viz.clean_axes(ax)

    fig.tight_layout()
    return viz.save(fig, name)

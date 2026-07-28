# -*- coding: utf-8 -*-
"""Структура рынка: категории, концентрация, распределение выручки."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config import MIN_CATEGORY_SIZE
from ..features import herfindahl
from .. import viz


def category_economics(df: pd.DataFrame) -> pd.DataFrame:
    """Экономика категорий: размер, цена, спрос, оценка выручки."""
    g = df.groupby("category", observed=True)
    tbl = pd.DataFrame({
        "Товаров": g.size(),
        "Медианная цена, $": g["price"].median().round(2),
        "Медианный рейтинг": g["rating"].median().round(2),
        "Медианный спрос, шт/мес": g["bought_last_month"].median(),
        "Оценка выручки, $": g["est_revenue"].sum(),
        "Доля спонсорских, %": (g["is_sponsored"].mean() * 100).round(1),
        "Доля с купоном, %": (g["has_coupon"].mean() * 100).round(1),
        "Медианная скидка, %": g["discount_pct"].median().round(1),
    })
    tbl["Доля выручки, %"] = (tbl["Оценка выручки, $"] / tbl["Оценка выручки, $"].sum() * 100).round(1)
    tbl["Средний чек к рынку"] = (
        tbl["Медианная цена, $"] / df["price"].median()
    ).round(2)
    return tbl.sort_values("Оценка выручки, $", ascending=False)


def revenue_concentration(df: pd.DataFrame) -> dict:
    """Насколько выручка сосредоточена в верхушке ассортимента (Парето)."""
    rev = df["est_revenue"].dropna().sort_values(ascending=False)
    if rev.empty:
        return {}
    total = rev.sum()
    cum = rev.cumsum() / total
    n = len(rev)
    out = {"n_products_with_revenue": n, "total_est_revenue": float(total)}
    for q in (0.01, 0.05, 0.10, 0.20, 0.50):
        k = max(1, int(round(n * q)))
        out[f"top_{int(q*100)}pct_revenue_share"] = round(float(cum.iloc[k - 1] * 100), 1)
    # Сколько товаров даёт половину выручки
    out["products_for_50pct_revenue"] = int((cum < 0.5).sum() + 1)
    out["share_of_products_for_50pct"] = round(out["products_for_50pct_revenue"] / n * 100, 1)
    return out


def brand_concentration(df: pd.DataFrame, min_products: int = MIN_CATEGORY_SIZE) -> pd.DataFrame:
    """HHI по брендам внутри каждой категории.

    HHI считается по оценке выручки: <1500 — конкурентный рынок,
    1500–2500 — умеренно концентрированный, >2500 — концентрированный
    (пороги антимонопольных регуляторов США).
    """
    rows = []
    for cat, sub in df.groupby("category", observed=True):
        if len(sub) < min_products:
            continue
        by_brand = sub.groupby("brand", observed=True)["est_revenue"].sum()
        by_brand = by_brand[by_brand > 0]
        if len(by_brand) < 3:
            continue
        top = by_brand.sort_values(ascending=False)
        rows.append({
            "Категория": cat,
            "Товаров": len(sub),
            "Брендов": int(sub["brand"].nunique()),
            "HHI": round(herfindahl(by_brand), 0),
            "Доля топ-1 бренда, %": round(float(top.iloc[0] / top.sum() * 100), 1),
            "Доля топ-3, %": round(float(top.head(3).sum() / top.sum() * 100), 1),
            "Лидер": top.index[0],
        })
    out = pd.DataFrame(rows).sort_values("HHI", ascending=False)
    if not out.empty:
        out["Тип рынка"] = pd.cut(
            out["HHI"], [-np.inf, 1500, 2500, np.inf],
            labels=["конкурентный", "умеренно концентр.", "концентрированный"],
        )
    return out


def top_brands(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    g = df.dropna(subset=["brand"]).groupby("brand", observed=True)
    tbl = pd.DataFrame({
        "Товаров": g.size(),
        "Оценка выручки, $": g["est_revenue"].sum(),
        "Медианная цена, $": g["price"].median().round(2),
        "Медианный рейтинг": g["rating"].median().round(2),
        "Показов в выдаче": g["n_listings"].sum(),
    })
    tbl["Доля выручки, %"] = (tbl["Оценка выручки, $"] / tbl["Оценка выручки, $"].sum() * 100).round(2)
    return tbl.sort_values("Оценка выручки, $", ascending=False).head(n)


def price_band_profile(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("price_band", observed=True)
    tbl = pd.DataFrame({
        "Товаров": g.size(),
        "Медианный рейтинг": g["rating"].median().round(2),
        "Медианный спрос": g["bought_last_month"].median(),
        "Оценка выручки, $": g["est_revenue"].sum(),
        "Доля с купоном, %": (g["has_coupon"].mean() * 100).round(1),
        "Доля спонсорских, %": (g["is_sponsored"].mean() * 100).round(1),
    })
    tbl["Доля выручки, %"] = (tbl["Оценка выручки, $"] / tbl["Оценка выручки, $"].sum() * 100).round(1)
    return tbl


# --------------------------------------------------------------------------
# Графики
# --------------------------------------------------------------------------


def plot_category_economics(tbl: pd.DataFrame, name: str = "03_category_economics"):
    top = tbl.head(12).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)

    ax = axes[0]
    ax.barh(top.index, top["Доля выручки, %"], color=viz.ACCENT, height=0.68)
    ax.set_title("Доля в оценке выручки, %")
    ax.set_xlabel("%")
    for y, v in zip(range(len(top)), top["Доля выручки, %"]):
        ax.text(v + 0.4, y, f"{v:.1f}", va="center", fontsize=9, color=viz.INK)
    viz.clean_axes(ax)

    ax = axes[1]
    ax.barh(top.index, top["Медианная цена, $"], color=viz.ACCENT_2, height=0.68)
    ax.set_title("Медианная цена, $")
    ax.set_xlabel("$")
    for y, v in zip(range(len(top)), top["Медианная цена, $"]):
        ax.text(v * 1.02, y, f"{v:,.0f}", va="center", fontsize=9, color=viz.INK)
    ax.set_xscale("log")
    viz.clean_axes(ax)

    fig.suptitle("Экономика категорий: где деньги и по какой цене",
                 x=0.005, ha="left", fontsize=14, fontweight="600")
    fig.tight_layout()
    return viz.save(fig, name)


def plot_pareto(df: pd.DataFrame, conc: dict, name: str = "04_revenue_pareto"):
    rev = df["est_revenue"].dropna().sort_values(ascending=False)
    cum = rev.cumsum() / rev.sum() * 100
    x = np.arange(1, len(rev) + 1) / len(rev) * 100

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(x, cum, color=viz.ACCENT, lw=2.2)
    ax.fill_between(x, cum, color=viz.ACCENT, alpha=0.08)
    ax.plot([0, 100], [0, 100], color=viz.MUTED, ls="--", lw=1,
            label="равномерное распределение")

    for q in (1, 5, 10, 20):
        key = f"top_{q}pct_revenue_share"
        if key in conc:
            ax.scatter([q], [conc[key]], color=viz.INK, s=26, zorder=5)
            ax.annotate(f"топ-{q}% товаров → {conc[key]:.0f}% выручки",
                        (q, conc[key]), textcoords="offset points",
                        xytext=(10, -4), fontsize=9, color=viz.INK)

    ax.set_xlabel("Доля товаров (от самых выручных), %")
    ax.set_ylabel("Накопленная доля оценки выручки, %")
    ax.set_title("Концентрация выручки: длинный хвост почти ничего не весит")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    viz.clean_axes(ax)
    return viz.save(fig, name)


def plot_brand_hhi(tbl: pd.DataFrame, name: str = "05_brand_hhi"):
    if tbl.empty:
        return None
    t = tbl.sort_values("HHI").tail(14)
    colors = [viz.ACCENT_4 if v > 2500 else (viz.ACCENT if v > 1500 else viz.ACCENT_3)
              for v in t["HHI"]]

    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.barh(t["Категория"], t["HHI"], color=colors, height=0.68)
    for i, (v, lead) in enumerate(zip(t["HHI"], t["Лидер"])):
        ax.text(v + 60, i, f"{v:,.0f} · {lead}", va="center", fontsize=9, color=viz.INK)

    ax.axvline(1500, color=viz.MUTED, ls="--", lw=1)
    ax.axvline(2500, color=viz.MUTED, ls="--", lw=1)
    ax.text(1500, len(t) - 0.3, " умеренная\n концентрация", fontsize=8, color=viz.MUTED, va="top")
    ax.text(2500, len(t) - 0.3, " высокая", fontsize=8, color=viz.MUTED, va="top")
    ax.set_xlabel("HHI по оценке выручки (0–10 000)")
    ax.set_title("Концентрация брендов внутри категорий")
    ax.set_xlim(0, max(float(t["HHI"].max()) * 1.35, 3000))
    viz.clean_axes(ax)
    return viz.save(fig, name)

# -*- coding: utf-8 -*-
"""Ценообразование: эластичность спроса по цене и эффект скидок.

Оценивается лог-логарифмическая модель

    log10(спрос) = a + b * log10(цена) + (контроли)

где b — эластичность: на сколько процентов меняется спрос при изменении цены
на 1%. Регрессия считается напрямую через МНК (numpy.linalg.lstsq), доверительные
интервалы — бутстрэпом по строкам, без предположения о нормальности остатков.

Оговорка, без которой числа нельзя читать буквально: `bought_last_month` — это
округлённая нижняя граница диапазона, который показывает сам Amazon ('300+',
'6K+'). Такое огрубление сжимает дисперсию зависимой переменной и смещает
оценку эластичности к нулю. Поэтому корректно сравнивать категории между собой,
но не утверждать, что абсолютное значение b точно.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config import MIN_CATEGORY_SIZE, RANDOM_STATE
from .. import viz


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """МНК с константой. -> (коэффициенты вкл. свободный член, R²)."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, r2


def bootstrap_ci(X: np.ndarray, y: np.ndarray, idx: int = 1,
                 n_boot: int = 500, seed: int = RANDOM_STATE) -> tuple[float, float]:
    """Перцентильный бутстрэп-интервал для одного коэффициента."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        try:
            beta, _ = ols(X[s], y[s])
            vals[i] = beta[idx]
        except np.linalg.LinAlgError:
            vals[i] = np.nan
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def _prep(sub: pd.DataFrame, controls: bool) -> tuple[np.ndarray, np.ndarray] | None:
    cols = ["log_price", "log_demand"] + (["rating", "log_reviews"] if controls else [])
    d = sub[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 40:
        return None
    y = d["log_demand"].to_numpy(float)
    X = d[[c for c in cols if c != "log_demand"]].to_numpy(float)
    return X, y


def elasticity_by_category(df: pd.DataFrame,
                           min_size: int = MIN_CATEGORY_SIZE,
                           n_boot: int = 400) -> pd.DataFrame:
    """Эластичность спроса по цене в каждой категории — с контролями и без."""
    rows = []
    for cat, sub in df.groupby("category", observed=True):
        if len(sub) < min_size:
            continue

        simple = _prep(sub, controls=False)
        if simple is None:
            continue
        Xs, ys = simple
        beta_s, r2_s = ols(Xs, ys)
        lo, hi = bootstrap_ci(Xs, ys, idx=1, n_boot=n_boot)

        beta_c, r2_c = (np.array([np.nan] * 4), np.nan)
        full = _prep(sub, controls=True)
        if full is not None:
            Xc, yc = full
            beta_c, r2_c = ols(Xc, yc)

        rows.append({
            "Категория": cat,
            "n": len(ys),
            "Эластичность": round(float(beta_s[1]), 3),
            "CI низ": round(float(lo), 3),
            "CI верх": round(float(hi), 3),
            "R²": round(float(r2_s), 3),
            "Эластичность с контролями": round(float(beta_c[1]), 3),
            "R² с контролями": round(float(r2_c), 3) if not np.isnan(r2_c) else np.nan,
            "Значима": bool(lo * hi > 0),
        })

    out = pd.DataFrame(rows)
    return out.sort_values("Эластичность") if not out.empty else out


def market_elasticity(df: pd.DataFrame, n_boot: int = 600) -> dict:
    """Эластичность по рынку целиком, с контролями и без."""
    res: dict = {}
    simple = _prep(df, controls=False)
    if simple:
        X, y = simple
        beta, r2 = ols(X, y)
        lo, hi = bootstrap_ci(X, y, 1, n_boot)
        res |= {"elasticity": round(float(beta[1]), 3), "ci": (round(lo, 3), round(hi, 3)),
                "r2": round(float(r2), 3), "n": len(y)}
    full = _prep(df, controls=True)
    if full:
        X, y = full
        beta, r2 = ols(X, y)
        lo, hi = bootstrap_ci(X, y, 1, n_boot)
        res |= {"elasticity_controlled": round(float(beta[1]), 3),
                "ci_controlled": (round(lo, 3), round(hi, 3)),
                "r2_controlled": round(float(r2), 3),
                "beta_rating": round(float(beta[2]), 3),
                "beta_log_reviews": round(float(beta[3]), 3)}
    return res


def discount_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Связь глубины скидки со спросом внутри ценовых полос.

    Разрез по полосам обязателен: скидки концентрируются в дешёвом сегменте,
    а дешёвый сегмент и без того более ходовой. Без контроля цены эффект
    скидки оказался бы завышен просто из-за состава товаров.
    """
    d = df[df["has_discount"] & df["log_demand"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["depth"] = pd.cut(d["discount_pct"], [0, 10, 20, 30, 50, 100],
                        labels=["1–10%", "10–20%", "20–30%", "30–50%", "50%+"])

    # Медиана здесь не годится: спрос принимает лишь несколько округлённых
    # значений (100, 200, 500, 1000), и медиана скачками перепрыгивает между
    # ними, создавая ложные «приросты» ровно на 100%. Среднее геометрическое
    # (среднее логарифмов) устойчиво к такой дискретности.
    base_src = df[~df["has_discount"] & df["log_demand"].notna()]
    # Индекс приводится к строкам: сопоставление по категориальному индексу
    # вернуло бы Categorical, с которым не работает арифметика ниже.
    base = base_src.groupby("price_band", observed=True)["log_demand"].mean()
    base.index = base.index.astype(str)
    base_n = base_src.groupby("price_band", observed=True).size()
    base_n.index = base_n.index.astype(str)

    g = d.groupby(["price_band", "depth"], observed=True)
    tbl = pd.DataFrame({
        "Товаров": g.size(),
        "_log": g["log_demand"].mean(),
    }).reset_index()
    tbl["price_band"] = tbl["price_band"].astype(str)
    tbl["depth"] = tbl["depth"].astype(str)
    tbl["_base"] = tbl["price_band"].map(base).astype(float)
    tbl["Средний спрос (геом.)"] = (10 ** tbl["_log"]).round(0)
    tbl["Спрос без скидки (геом.)"] = (10 ** tbl["_base"]).round(0)
    tbl["Разница, %"] = ((10 ** (tbl["_log"] - tbl["_base"]) - 1) * 100).round(1)
    tbl["Товаров без скидки"] = tbl["price_band"].map(base_n).astype("Int64")

    tbl = tbl[(tbl["Товаров"] >= 25) & (tbl["Товаров без скидки"] >= 25)]
    return (tbl.drop(columns=["_log", "_base"])
              .rename(columns={"price_band": "Ценовая полоса", "depth": "Глубина скидки"}))


# --------------------------------------------------------------------------
# Графики
# --------------------------------------------------------------------------


def plot_elasticity(tbl: pd.DataFrame, market: dict, name: str = "06_price_elasticity"):
    if tbl.empty:
        return None
    t = tbl.sort_values("Эластичность")
    y = np.arange(len(t))

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(t) + 3.2))
    for i, (lo, hi, est, sig) in enumerate(zip(
            t["CI низ"], t["CI верх"], t["Эластичность"], t["Значима"])):
        color = viz.ACCENT if sig else viz.MUTED
        ax.plot([lo, hi], [i, i], color=color, lw=2.4, solid_capstyle="round")
        ax.scatter([est], [i], color=color, s=42, zorder=5)

    ax.axvline(0, color=viz.INK, lw=1)
    if "elasticity" in market:
        ax.axvline(market["elasticity"], color=viz.ACCENT_2, ls="--", lw=1.4,
                   label=f"рынок целиком: {market['elasticity']:.2f}")
        ax.legend(loc="lower right")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}  (n={n})" for c, n in zip(t["Категория"], t["n"])])
    ax.set_xlabel("Эластичность спроса по цене (изменение спроса, % на 1% цены)")
    ax.set_title("Эластичность по категориям, 95% бутстрэп-интервал\n"
                 "серым — интервал накрывает ноль (эффект не отличим от нуля)")
    viz.clean_axes(ax)
    return viz.save(fig, name)


def plot_price_demand(df: pd.DataFrame, name: str = "07_price_demand_cloud"):
    d = df[["log_price", "log_demand", "price", "bought_last_month"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return None

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.scatter(d["price"], d["bought_last_month"], s=7, alpha=0.13,
               color=viz.ACCENT_2, edgecolors="none")

    # Медианный спрос по децилям цены — устойчивая линия тренда поверх облака
    q = pd.qcut(d["price"], 20, duplicates="drop")
    med = d.groupby(q, observed=True).agg(p=("price", "median"),
                                          dmd=("bought_last_month", "median"))
    ax.plot(med["p"], med["dmd"], color=viz.ACCENT, lw=2.4, marker="o", ms=4,
            label="медианный спрос по децилям цены")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Цена, $ (лог. шкала)")
    ax.set_ylabel("Куплено за месяц, шт (лог. шкала)")
    ax.set_title("Цена против спроса: связь есть, но она слабая и нелинейная")
    ax.legend()
    viz.clean_axes(ax)
    return viz.save(fig, name)

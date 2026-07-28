# -*- coding: utf-8 -*-
"""Эффект рекламы, значков и купонов — с поправкой на состав товаров.

Наивное сравнение «спонсорские против органических» отвечает не на тот вопрос.
Реклама, значки и купоны распределены по ассортименту неслучайно: они
концентрируются в определённых категориях и ценовых сегментах. Поэтому разница
в средних смешивает собственно эффект метки с тем, что метка достаётся другим
товарам.

Здесь используется стандартная схема корректировки на наблюдаемые различия:

1. Логистической регрессией оценивается propensity score — вероятность
   получить метку при данных характеристиках товара (категория, цена,
   рейтинг, отзывы, скидка).
2. Эффект считается двумя независимыми способами — взвешиванием (IPW, оценка
   ATT) и подбором пар ближайших соседей по propensity с калипером.
3. Качество корректировки проверяется по стандартизованным разностям средних
   (SMD): если после поправки |SMD| < 0.1 по всем ковариатам, группы
   сопоставимы.

Чего эта схема НЕ даёт: это наблюдательные данные, поправка убирает только
*измеренные* различия. Рекламный бюджет продавца, реальная скорость продаж и
history аккаунта в датасете отсутствуют, поэтому корректно говорить
«при равных наблюдаемых характеристиках», а не «реклама приводит к».
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_STATE
from .. import viz

NUM_CONFOUNDERS = ["log_price", "rating", "log_reviews", "discount_pct_filled", "title_words"]
CAT_CONFOUNDERS = ["category"]


def _design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.Index]:
    """Матрица ковариат: числовые (стандартизованные) + one-hot категории."""
    d = df.dropna(subset=NUM_CONFOUNDERS + CAT_CONFOUNDERS)
    num = d[NUM_CONFOUNDERS].to_numpy(float)
    num = StandardScaler().fit_transform(num)
    cat = pd.get_dummies(d[CAT_CONFOUNDERS], drop_first=True, dtype=float)
    X = np.column_stack([num, cat.to_numpy(float)])
    names = NUM_CONFOUNDERS + list(cat.columns)
    return X, names, d.index


def _smd(x: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> float:
    """Стандартизованная разность средних между группами."""
    if w is None:
        w = np.ones(len(t))
    wt, wc = w[t == 1], w[t == 0]
    xt, xc = x[t == 1], x[t == 0]
    if wt.sum() == 0 or wc.sum() == 0:
        return np.nan
    mt = np.average(xt, weights=wt)
    mc = np.average(xc, weights=wc)
    vt = np.average((xt - mt) ** 2, weights=wt)
    vc = np.average((xc - mc) ** 2, weights=wc)
    pooled = np.sqrt((vt + vc) / 2)
    return float((mt - mc) / pooled) if pooled > 0 else np.nan


def adjusted_effect(df: pd.DataFrame, treatment: str, outcome: str = "log_demand",
                    seed: int = RANDOM_STATE) -> dict:
    """Эффект метки `treatment` на `outcome`: наивный, IPW и по подобранным парам."""
    work = df[df[outcome].notna()].copy()
    X, names, idx = _design_matrix(work)
    work = work.loc[idx]
    t = work[treatment].astype(bool).to_numpy()
    y = work[outcome].to_numpy(float)

    n_t, n_c = int(t.sum()), int((~t).sum())
    if n_t < 30 or n_c < 30:
        return {"treatment": treatment, "error": "слишком мало наблюдений в одной из групп"}

    res: dict = {"treatment": treatment, "outcome": outcome,
                 "n_treated": n_t, "n_control": n_c}

    # --- наивная разница ---------------------------------------------------
    res["naive_diff"] = float(y[t].mean() - y[~t].mean())

    # --- propensity score --------------------------------------------------
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    lr.fit(X, t)
    ps_full = np.clip(lr.predict_proba(X)[:, 1], 0.001, 0.999)
    res["ps_auc"] = float(_auc(t, ps_full))

    # --- область общей поддержки ------------------------------------------
    # Сравнивать можно только тех, у кого вообще есть сопоставимая пара.
    # Наблюдения вне пересечения диапазонов propensity отбрасываются: без этого
    # веса «разъезжаются» и баланс не достигается.
    lo = max(ps_full[t].min(), ps_full[~t].min())
    hi = min(ps_full[t].max(), ps_full[~t].max())
    support = (ps_full >= lo) & (ps_full <= hi)
    res["n_dropped_off_support"] = int((~support).sum())

    X, t, y, ps = X[support], t[support], y[support], ps_full[support]
    res["n_treated_on_support"] = int(t.sum())
    if t.sum() < 20 or (~t).sum() < 20:
        res["error"] = "после отсечения по общей поддержке осталось слишком мало наблюдений"
        return res

    # --- IPW (оценка ATT) --------------------------------------------------
    # Веса контрольной группы подрезаются по 99-му перцентилю: единичные
    # наблюдения с propensity около единицы иначе получают вес в сотни единиц
    # и определяют оценку в одиночку.
    w = np.where(t, 1.0, ps / (1 - ps))
    cap = np.quantile(w[~t], 0.99)
    w = np.where(t, 1.0, np.minimum(w, cap))
    res["weight_cap"] = round(float(cap), 2)
    res["ipw_diff"] = float(
        np.average(y[t], weights=w[t]) - np.average(y[~t], weights=w[~t])
    )

    # --- подбор пар по propensity -----------------------------------------
    logit_ps = np.log(ps / (1 - ps))
    caliper = 0.2 * float(np.std(logit_ps))
    nn = NearestNeighbors(n_neighbors=1).fit(logit_ps[~t].reshape(-1, 1))
    dist, ind = nn.kneighbors(logit_ps[t].reshape(-1, 1))
    keep = dist.ravel() <= caliper
    if keep.sum() >= 20:
        yt = y[t][keep]
        yc = y[~t][ind.ravel()[keep]]
        res["matched_diff"] = float(yt.mean() - yc.mean())
        res["n_matched"] = int(keep.sum())
        # бутстрэп по парам
        rng = np.random.default_rng(seed)
        boots = [float((yt[s] - yc[s]).mean())
                 for s in (rng.integers(0, len(yt), len(yt)) for _ in range(500))]
        res["matched_ci"] = tuple(np.percentile(boots, [2.5, 97.5]).round(4))
    else:
        res["matched_diff"] = np.nan
        res["n_matched"] = int(keep.sum())

    # --- баланс ковариат ---------------------------------------------------
    balance = []
    for j, nm in enumerate(names):
        before = _smd(X[:, j], t.astype(int))
        after = _smd(X[:, j], t.astype(int), w)
        balance.append({"Ковариата": nm, "SMD до": before, "SMD после (IPW)": after})
    res["balance"] = pd.DataFrame(balance)
    res["max_abs_smd_before"] = float(res["balance"]["SMD до"].abs().max())
    res["max_abs_smd_after"] = float(res["balance"]["SMD после (IPW)"].abs().max())
    res["balance_ok"] = bool(res["max_abs_smd_after"] < 0.1)

    # Перевод из log10 в проценты — так эффект читается содержательно
    for k in ("naive_diff", "ipw_diff", "matched_diff"):
        v = res.get(k)
        res[k + "_pct"] = (10 ** v - 1) * 100 if v is not None and not np.isnan(v) else np.nan

    # Согласованность двух независимых оценок — вторая проверка надёжности.
    # Если взвешивание и подбор пар расходятся по знаку, доверять числу нельзя.
    ipw_p, m_p = res.get("ipw_diff_pct"), res.get("matched_diff_pct")
    if ipw_p is not None and m_p is not None and not np.isnan(m_p):
        res["methods_agree"] = bool(np.sign(ipw_p) == np.sign(m_p))
    else:
        res["methods_agree"] = False

    res["reliable"] = bool(res["balance_ok"] and res["methods_agree"])
    return res


def _auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC без зависимости от sklearn.metrics (ранговая формула)."""
    y = np.asarray(y_true).astype(int)
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def compare_treatments(df: pd.DataFrame, treatments: list[str],
                       outcome: str = "log_demand") -> tuple[pd.DataFrame, dict]:
    """Прогоняет корректировку по нескольким меткам сразу."""
    rows, details = [], {}
    for tr in treatments:
        r = adjusted_effect(df, tr, outcome)
        details[tr] = r
        if "error" in r:
            continue
        rows.append({
            "Метка": tr,
            "Товаров с меткой": r["n_treated"],
            "Наивно, %": round(r["naive_diff_pct"], 1),
            "С поправкой (IPW), %": round(r["ipw_diff_pct"], 1),
            "По парам, %": round(r["matched_diff_pct"], 1) if not np.isnan(r["matched_diff"]) else np.nan,
            "Пар подобрано": r.get("n_matched", 0),
            "AUC propensity": round(r["ps_auc"], 3),
            "max|SMD| после": round(r["max_abs_smd_after"], 3),
            "Баланс достигнут": "да" if r["balance_ok"] else "нет",
            "Методы согласуются": "да" if r["methods_agree"] else "нет",
            "Оценка надёжна": "да" if r["reliable"] else "нет",
        })
    return pd.DataFrame(rows), details


def naive_profile(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Простое сравнение групп «есть метка / нет метки» по ключевым метрикам."""
    g = df.groupby(flag, observed=True)
    tbl = pd.DataFrame({
        "Товаров": g.size(),
        "Медианная цена, $": g["price"].median().round(2),
        "Медианный рейтинг": g["rating"].median().round(2),
        "Медианное число отзывов": g["num_reviews"].median(),
        "Медианный спрос, шт/мес": g["bought_last_month"].median(),
        "Доля с купоном, %": (g["has_coupon"].mean() * 100).round(1),
        "Медианных показов в выдаче": g["n_listings"].median(),
    })
    tbl.index = ["без метки", "с меткой"]
    return tbl


# --------------------------------------------------------------------------
# Графики
# --------------------------------------------------------------------------

LABELS_RU = {
    "is_sponsored": "Спонсорское размещение",
    "is_best_seller": "Значок Best Seller",
    "is_amazons_choice": "Значок Amazon's Choice",
    "has_coupon": "Купон",
    "has_sustainability_badge": "Эко-значок",
}

COVARIATE_LABELS = {
    "log_price": "Цена (лог)",
    "rating": "Рейтинг",
    "log_reviews": "Число отзывов (лог)",
    "discount_pct_filled": "Глубина скидки, %",
    "title_words": "Слов в названии",
}


def _covariate_label(name: str) -> str:
    """Человекочитаемое имя ковариаты (one-hot категории — просто название)."""
    if name in COVARIATE_LABELS:
        return COVARIATE_LABELS[name]
    if name.startswith("category_"):
        return "категория: " + name[len("category_"):]
    return name


def plot_effects(tbl: pd.DataFrame, name: str = "08_adjusted_effects"):
    if tbl.empty:
        return None
    t = tbl.iloc[::-1]
    y = np.arange(len(t))
    h = 0.36

    reliable = t["Оценка надёжна"].eq("да").to_numpy()
    adj_colors = [viz.ACCENT if r else "#d9a48a" for r in reliable]

    fig, ax = plt.subplots(figsize=(10.5, 0.95 * len(t) + 3.4))
    ax.barh(y + h / 2, t["Наивно, %"], height=h, color=viz.MUTED, label="наивное сравнение")
    ax.barh(y - h / 2, t["С поправкой (IPW), %"], height=h, color=adj_colors,
            label="с поправкой на состав товаров (IPW)")

    for i, (nv, adj) in enumerate(zip(t["Наивно, %"], t["С поправкой (IPW), %"])):
        ax.text(nv + (6 if nv >= 0 else -6), i + h / 2, f"{nv:+.0f}%",
                va="center", ha="left" if nv >= 0 else "right", fontsize=9, color=viz.INK)
        ax.text(adj + (6 if adj >= 0 else -6), i - h / 2, f"{adj:+.0f}%",
                va="center", ha="left" if adj >= 0 else "right", fontsize=9,
                color=viz.ACCENT, fontweight="600")

    ax.axvline(0, color=viz.INK, lw=1)
    ax.set_yticks(y)
    # Ненадёжные оценки помечаются прямо в подписи оси: иначе число прочтут
    # как вывод, не заметив оговорки в легенде.
    labels = [LABELS_RU.get(m, m) + ("" if r else "\n(оценка неустойчива)")
              for m, r in zip(t["Метка"], reliable)]
    ax.set_yticklabels(labels, fontsize=9.5)
    for tick, r in zip(ax.get_yticklabels(), reliable):
        if not r:
            tick.set_color(viz.ACCENT_4)
    ax.set_xlabel("Разница в спросе относительно товаров без метки, %")
    ax.set_title("Что остаётся от «эффекта» метки при сравнении сопоставимых товаров\n"
                 "бледным — оценки, не прошедшие проверку качества", fontsize=11.5)
    ax.set_xlim(min(-40, t[["Наивно, %", "С поправкой (IPW), %"]].min().min() * 1.25),
                max(60, t[["Наивно, %", "С поправкой (IPW), %"]].max().max() * 1.25))
    ax.legend(loc="lower right")
    viz.clean_axes(ax)
    return viz.save(fig, name)


def plot_balance(detail: dict, name: str = "09_covariate_balance"):
    """Love plot: баланс ковариат до и после взвешивания."""
    bal = detail.get("balance")
    if bal is None or bal.empty:
        return None
    b = bal.reindex(bal["SMD до"].abs().sort_values().index).tail(16)
    y = np.arange(len(b))

    fig, ax = plt.subplots(figsize=(9.5, 0.38 * len(b) + 2.8))
    ax.scatter(b["SMD до"].abs(), y, s=46, color=viz.MUTED, label="до поправки", zorder=4)
    ax.scatter(b["SMD после (IPW)"].abs(), y, s=46, color=viz.ACCENT,
               label="после поправки (IPW)", zorder=5)
    for i, (a, c) in enumerate(zip(b["SMD до"].abs(), b["SMD после (IPW)"].abs())):
        ax.plot([a, c], [i, i], color=GRID_LINE, lw=1, zorder=3)

    ax.axvline(0.1, color=viz.ACCENT_4, ls="--", lw=1.2)
    ax.text(0.1, len(b) - 0.4, " порог 0.1", fontsize=9, color=viz.ACCENT_4, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([_covariate_label(c) for c in b["Ковариата"]], fontsize=9)
    ax.set_xlabel("|Стандартизованная разность средних|")
    ax.set_title(f"Баланс ковариат: {LABELS_RU.get(detail['treatment'], detail['treatment'])}\n"
                 "чем ближе к нулю, тем сопоставимее группы")
    ax.legend(loc="lower right")
    viz.clean_axes(ax)
    return viz.save(fig, name)


GRID_LINE = "#d8dade"

# -*- coding: utf-8 -*-
"""Сегментация ассортимента: какие типы товарных карточек вообще бывают.

Кластеризация здесь нужна не ради кластеризации, а чтобы получить язык для
разговора о рынке: вместо 8 тысяч отдельных товаров — несколько устойчивых
архетипов («дешёвый ходовой расходник», «дорогая витрина без продаж» и т.д.),
у каждого своя экономика.

Число кластеров выбирается по силуэту, а не назначается вручную.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_STATE
from .. import viz

FEATURES = ["log_price", "rating", "log_reviews", "log_demand",
            "discount_pct_filled", "n_listings"]


def fit_segments(df: pd.DataFrame, k_range=range(3, 9),
                 seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, dict]:
    """Кластеризует товары, возвращает (таблица с меткой кластера, диагностика)."""
    d = df.dropna(subset=FEATURES).copy()
    if len(d) < 200:
        return d, {"error": "недостаточно данных"}

    X = d[FEATURES].to_numpy(float)
    X[:, FEATURES.index("n_listings")] = np.log10(X[:, FEATURES.index("n_listings")] + 1)
    Xs = StandardScaler().fit_transform(X)

    # Силуэт считаем на подвыборке — на полном наборе это O(n²) по памяти
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(Xs), size=min(4000, len(Xs)), replace=False)

    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        scores[k] = float(silhouette_score(Xs[sample], km.labels_[sample]))

    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(Xs)
    d["segment"] = km.labels_

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(Xs)
    d["_pc1"], d["_pc2"] = coords[:, 0], coords[:, 1]

    diag = {
        "k": best_k,
        "silhouette": round(scores[best_k], 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "explained_variance": [round(float(v), 3) for v in pca.explained_variance_ratio_],
        "n": len(d),
    }
    return d, diag


def profile(d: pd.DataFrame) -> pd.DataFrame:
    """Профиль каждого сегмента в исходных (непреобразованных) единицах."""
    g = d.groupby("segment", observed=True)
    tbl = pd.DataFrame({
        "Товаров": g.size(),
        "Медианная цена, $": g["price"].median().round(2),
        "Медианный рейтинг": g["rating"].median().round(2),
        "Медианно отзывов": g["num_reviews"].median().round(0),
        "Медианный спрос, шт/мес": g["bought_last_month"].median().round(0),
        "Медианная скидка, %": g["discount_pct_filled"].median().round(1),
        "Медианно показов": g["n_listings"].median().round(0),
        "Оценка выручки, $": g["est_revenue"].sum().round(0),
    })
    tbl["Доля товаров, %"] = (tbl["Товаров"] / tbl["Товаров"].sum() * 100).round(1)
    tbl["Доля выручки, %"] = (tbl["Оценка выручки, $"] / tbl["Оценка выручки, $"].sum() * 100).round(1)
    tbl["Топ-категория"] = g["category"].agg(lambda s: s.value_counts().index[0])
    return tbl.sort_values("Оценка выручки, $", ascending=False)


def name_segments(prof: pd.DataFrame) -> dict[int, str]:
    """Присваивает сегментам человекочитаемые имена по их профилю.

    Правила намеренно простые и детерминированные: имя должно следовать из
    чисел в таблице профиля, а не из впечатления аналитика.
    """
    price_med = prof["Медианная цена, $"].median()
    demand_med = prof["Медианный спрос, шт/мес"].median()
    reviews_med = prof["Медианно отзывов"].median()

    names = {}
    for seg, row in prof.iterrows():
        pricey = row["Медианная цена, $"] > price_med
        popular = row["Медианный спрос, шт/мес"] > demand_med
        established = row["Медианно отзывов"] > reviews_med
        # Высокая видимость — отдельная ось: такие товары скрапер встречает
        # в выдаче десятки раз, и это их главная отличительная черта.
        very_visible = row["Медианно показов"] >= 10

        if very_visible:
            label = "Постоянно в выдаче"
        elif popular and not pricey:
            label = "Ходовой недорогой товар"
        elif popular and pricey:
            label = "Дорогой бестселлер"
        elif not popular and pricey:
            label = "Дорогая витрина, слабый спрос"
        elif established:
            label = "Зрелый товар с угасающим спросом"
        else:
            label = "Длинный хвост"
        names[seg] = f"{label}"
    # Разводим одинаковые имена, чтобы они оставались уникальными
    seen: dict[str, int] = {}
    for seg in list(names):
        base = names[seg]
        if base in seen.values():
            seen_count = sum(1 for v in seen.values() if v == base) + 1
            names[seg] = f"{base} ({seen_count})"
        seen[seg] = base
    return names


def plot_segments(d: pd.DataFrame, prof: pd.DataFrame, names: dict,
                  name: str = "11_segments"):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))

    ax = axes[0]
    for i, seg in enumerate(sorted(d["segment"].unique())):
        s = d[d["segment"] == seg]
        ax.scatter(s["_pc1"], s["_pc2"], s=7, alpha=0.3, edgecolors="none",
                   color=viz.SERIES[i % len(viz.SERIES)],
                   label=f"{names.get(seg, seg)} (n={len(s):,})")
    ax.set_xlabel("Главная компонента 1")
    ax.set_ylabel("Главная компонента 2")
    ax.set_title("Сегменты в пространстве главных компонент")
    leg = ax.legend(fontsize=8, markerscale=2.2, loc="best")
    for h in leg.legend_handles:
        h.set_alpha(1)
    viz.clean_axes(ax)

    ax = axes[1]
    p = prof.copy()
    p.index = [names.get(i, i) for i in p.index]
    p = p.sort_values("Доля выручки, %")
    y = np.arange(len(p))
    ax.barh(y - 0.2, p["Доля товаров, %"], height=0.38, color=viz.MUTED,
            label="доля товаров")
    ax.barh(y + 0.2, p["Доля выручки, %"], height=0.38, color=viz.ACCENT,
            label="доля оценки выручки")
    ax.set_yticks(y)
    ax.set_yticklabels(p.index, fontsize=9)
    ax.set_xlabel("%")
    ax.set_title("Сегменты: доля в ассортименте против доли в выручке")
    ax.legend()
    viz.clean_axes(ax)

    fig.tight_layout()
    return viz.save(fig, name)

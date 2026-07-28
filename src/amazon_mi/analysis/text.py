# -*- coding: utf-8 -*-
"""Текст названия как сигнал: какие формулировки сопутствуют высокому спросу.

Названия на Amazon — это не описание, а поле оптимизации: продавец набивает
туда ключевые слова. Поэтому по ним видно, какой язык используют карточки,
попадающие в верхний квартиль спроса.

Метод: TF-IDF по словам и биграммам -> L2-логистическая регрессия -> веса
признаков. Регуляризация обязательна, иначе редкие n-граммы получают огромные
коэффициенты; частотный порог `min_df` отсекает случайный хвост.

Читать результат как «добавьте слово X в название, и продажи вырастут» нельзя:
слова маркируют товарные ниши (батарейки, картриджи), а не влияют на спрос сами
по себе. Это описание языка спроса, а не рецепт.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ..config import DEMAND_TOP_QUANTILE, RANDOM_STATE
from .. import viz

STOP_WORDS = [
    "the", "and", "for", "with", "of", "to", "in", "on", "by", "or", "a", "an",
    "is", "it", "your", "you", "from", "at", "as", "all", "up", "new",
]


def fit_text_model(df: pd.DataFrame, seed: int = RANDOM_STATE) -> dict:
    """Логистическая регрессия на TF-IDF названий -> веса n-грамм."""
    d = df[df["bought_last_month"].notna() & df["title"].notna()].copy()
    if len(d) < 500:
        return {"error": "недостаточно данных"}

    threshold = float(d["bought_last_month"].quantile(DEMAND_TOP_QUANTILE))
    y = (d["bought_last_month"] > threshold).to_numpy(int)
    corpus = d["title"].str.lower().to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        corpus, y, test_size=0.25, stratify=y, random_state=seed
    )

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=25, max_features=25_000,
                          stop_words=STOP_WORDS, sublinear_tf=True)
    Xtr = vec.fit_transform(X_tr)
    Xte = vec.transform(X_te)

    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    clf.fit(Xtr, y_tr)
    auc = float(roc_auc_score(y_te, clf.predict_proba(Xte)[:, 1]))

    terms = np.array(vec.get_feature_names_out())
    coefs = clf.coef_.ravel()
    order = np.argsort(coefs)

    freq = np.asarray((Xtr > 0).sum(axis=0)).ravel()
    top_pos = pd.DataFrame({
        "N-грамма": terms[order[::-1]][:20],
        "Вес": coefs[order[::-1]][:20].round(3),
        "Товаров": freq[order[::-1]][:20],
    })
    top_neg = pd.DataFrame({
        "N-грамма": terms[order][:20],
        "Вес": coefs[order][:20].round(3),
        "Товаров": freq[order][:20],
    })

    return {"auc": round(auc, 4), "n": len(y), "n_features": len(terms),
            "threshold": threshold, "top_positive": top_pos, "top_negative": top_neg}


def plot_terms(res: dict, name: str = "14_title_terms"):
    if "error" in res:
        return None
    pos = res["top_positive"].head(14).iloc[::-1]
    neg = res["top_negative"].head(14).iloc[::-1]
    both = pd.concat([neg, pos])
    colors = [viz.ACCENT_4] * len(neg) + [viz.ACCENT_3] * len(pos)

    fig, ax = plt.subplots(figsize=(9.6, 0.36 * len(both) + 3))
    ax.barh(np.arange(len(both)), both["Вес"], color=colors, height=0.68)
    ax.set_yticks(np.arange(len(both)))
    ax.set_yticklabels(both["N-грамма"], fontsize=9)
    ax.axvline(0, color=viz.INK, lw=1)
    ax.set_xlabel("Вес в логистической регрессии")
    ax.set_title(f"Язык высокого спроса в названиях товаров\n"
                 f"зелёные — сопутствуют верхнему квартилю спроса, красные — нижним "
                 f"(AUC только по тексту = {res['auc']:.3f})")
    viz.clean_axes(ax)
    return viz.save(fig, name)

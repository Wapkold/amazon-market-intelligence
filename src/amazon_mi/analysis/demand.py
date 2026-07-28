# -*- coding: utf-8 -*-
"""Модель спроса: что вообще предсказывает попадание товара в верхний квартиль.

Модель здесь используется как инструмент измерения, а не как продукт. Вопрос,
на который она отвечает: сколько сигнала о спросе содержится в публично видимых
атрибутах карточки, и какие из них несут этот сигнал.

Важные методические решения:

* Оценка качества — только на отложенных фолдах (стратифицированная 5-блочная
  кросс-валидация), поэтому AUC не завышен подгонкой.
* Важность признаков — перестановочная (permutation importance) на отложенной
  части. Встроенная важность деревьев смещена в пользу признаков с большим
  числом уникальных значений и здесь не используется.
* Из признаков исключены `est_revenue` и `reviews_per_demand`: они вычислены
  из самого спроса, и их включение было бы утечкой целевой переменной.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..config import DEMAND_TOP_QUANTILE, RANDOM_STATE
from .. import viz

NUM_FEATURES = ["log_price", "rating", "log_reviews", "discount_pct_filled",
                "coupon_pct", "n_listings", "title_len", "title_words",
                "title_caps_ratio", "free_delivery_threshold"]
BOOL_FEATURES = ["is_sponsored", "is_best_seller", "is_amazons_choice",
                 "has_limited_deal", "has_coupon", "has_buy_box",
                 "has_sustainability_badge", "is_small_business",
                 "free_delivery", "has_fast_delivery", "title_has_digits"]
CAT_FEATURES = ["category"]

FEATURE_LABELS = {
    "log_price": "Цена (лог)",
    "rating": "Рейтинг",
    "log_reviews": "Число отзывов (лог)",
    "discount_pct_filled": "Глубина скидки, %",
    "coupon_pct": "Купон, %",
    "n_listings": "Показов в выдаче",
    "title_len": "Длина названия",
    "title_words": "Слов в названии",
    "title_caps_ratio": "Доля заглавных букв",
    "free_delivery_threshold": "Порог бесплатной доставки",
    "is_sponsored": "Спонсорское размещение",
    "is_best_seller": "Значок Best Seller",
    "is_amazons_choice": "Значок Amazon's Choice",
    "has_limited_deal": "Акция с таймером",
    "has_coupon": "Есть купон",
    "has_buy_box": "Есть buy box",
    "has_sustainability_badge": "Эко-значок",
    "is_small_business": "Малый бизнес",
    "free_delivery": "Бесплатная доставка",
    "has_fast_delivery": "Ускоренная доставка",
    "title_has_digits": "Цифры в названии",
    "category": "Категория",
}


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Матрица признаков и бинарная цель «спрос в верхнем квартиле»."""
    d = df[df["bought_last_month"].notna()].copy()
    threshold = float(d["bought_last_month"].quantile(DEMAND_TOP_QUANTILE))
    y = (d["bought_last_month"] > threshold).to_numpy(int)

    X = d[NUM_FEATURES].astype(float).copy()
    for c in BOOL_FEATURES:
        X[c] = d[c].astype(float)
    for c in CAT_FEATURES:
        X[c] = d[c].astype("category")
    return X, y, threshold


def cross_validate(X: pd.DataFrame, y: np.ndarray,
                   seed: int = RANDOM_STATE) -> dict:
    """Стратифицированная 5-блочная кросс-валидация."""
    cat_mask = [c in CAT_FEATURES for c in X.columns]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs, aps = [], []
    oof = np.zeros(len(y))

    for tr, te in skf.split(X, y):
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.07, max_depth=6,
            categorical_features=cat_mask, random_state=seed,
        )
        clf.fit(X.iloc[tr], y[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        aucs.append(roc_auc_score(y[te], p))
        aps.append(average_precision_score(y[te], p))

    return {
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "auc_folds": [round(a, 4) for a in aucs],
        "ap_mean": round(float(np.mean(aps)), 4),
        "baseline_rate": round(float(y.mean()), 4),
        "oof_pred": oof,
        "n": len(y),
    }


def fit_and_importance(X: pd.DataFrame, y: np.ndarray,
                       seed: int = RANDOM_STATE) -> tuple[object, pd.DataFrame, dict]:
    """Обучает финальную модель и считает перестановочную важность."""
    cat_mask = [c in CAT_FEATURES for c in X.columns]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.07, max_depth=6,
        categorical_features=cat_mask, random_state=seed,
    )
    clf.fit(X_tr, y_tr)

    p_te = clf.predict_proba(X_te)[:, 1]
    holdout = {"auc": round(float(roc_auc_score(y_te, p_te)), 4),
               "n_test": len(y_te), "y_test": y_te, "p_test": p_te}

    imp = permutation_importance(clf, X_te, y_te, n_repeats=10,
                                 random_state=seed, scoring="roc_auc", n_jobs=1)
    tbl = pd.DataFrame({
        "Признак": [FEATURE_LABELS.get(c, c) for c in X.columns],
        "Падение AUC": imp.importances_mean.round(5),
        "Разброс": imp.importances_std.round(5),
    }).sort_values("Падение AUC", ascending=False)
    return clf, tbl, holdout


def plot_importance(tbl: pd.DataFrame, name: str = "12_feature_importance"):
    t = tbl.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(t) + 2.6))
    colors = [viz.ACCENT if v > 0 else viz.MUTED for v in t["Падение AUC"]]
    ax.barh(t["Признак"], t["Падение AUC"], xerr=t["Разброс"],
            color=colors, height=0.66,
            error_kw={"ecolor": viz.MUTED, "elinewidth": 1, "capsize": 2})
    ax.set_xlabel("Падение AUC при перемешивании признака")
    ax.set_title("Что реально предсказывает высокий спрос\n"
                 "перестановочная важность на отложенной выборке")
    viz.clean_axes(ax)
    return viz.save(fig, name)


def plot_roc(holdout: dict, cv: dict, name: str = "13_roc"):
    fpr, tpr, _ = roc_curve(holdout["y_test"], holdout["p_test"])
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot(fpr, tpr, color=viz.ACCENT, lw=2.4,
            label=f"модель (AUC = {holdout['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=viz.MUTED, ls="--", lw=1, label="случайное угадывание")
    ax.set_xlabel("Доля ложных срабатываний")
    ax.set_ylabel("Доля верно найденных")
    ax.set_title(f"Качество модели спроса\nкросс-валидация: AUC = {cv['auc_mean']:.3f} ± {cv['auc_std']:.3f}")
    ax.legend(loc="lower right")
    viz.clean_axes(ax)
    return viz.save(fig, name)

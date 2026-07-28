# -*- coding: utf-8 -*-
"""Шаг 2: аналитика поверх очищенных данных + сборка HTML-отчёта.

Запуск:  python scripts/02_analyze.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from amazon_mi import cleaning, report, viz
from amazon_mi.analysis import demand, market, pricing, promo, ratings, segments, text
from amazon_mi.config import PRODUCTS_CSV, REPORTS, ensure_dirs

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def fmt(n) -> str:
    """Разделитель разрядов — только внутри самого числа."""
    return f"{n:,}".replace(",", " ")


def main() -> None:
    ensure_dirs()
    viz.setup()

    if not PRODUCTS_CSV.exists():
        sys.exit("Сначала запустите scripts/01_build_dataset.py")

    products = pd.read_csv(PRODUCTS_CSV, encoding="utf-8-sig")
    # pandas теряет тип категории при чтении CSV — восстанавливаем
    from amazon_mi.features import PRICE_BAND_LABELS
    products["price_band"] = pd.Categorical(
        products["price_band"], categories=PRICE_BAND_LABELS, ordered=True)

    df = cleaning.analytic_frame(products)
    R: dict = {
        "n_products_total": len(products),
        "n_products_analysed": len(df),
        # Максимум берётся по всем товарам, а не по аналитическому срезу:
        # рекордсмен по числу показов — как раз не-товар (подписка TRX).
        "max_listings_per_product": int(products["n_listings"].max()),
    }

    hr("ШАГ 2. АНАЛИЗ")
    from amazon_mi import taxonomy
    cov = taxonomy.coverage(df)
    R["taxonomy_coverage"] = cov
    print(f"Товаров всего: {fmt(len(products))}")
    print(f"В аналитическом срезе (без служебных строк и не-товаров): {fmt(len(df))}")
    print(f"Исключено не-товаров (страховки, подписки): "
          f"{fmt(int(products['is_non_product'].sum()))}")
    print(f"Покрытие категориями: {cov['coverage_pct']}% "
          f"({fmt(cov['unclassified'])} товаров в «Прочем»)")

    # ---------------------------------------------------------------- рынок
    hr("1. СТРУКТУРА РЫНКА")
    cat_tbl = market.category_economics(df)
    print(cat_tbl.to_string())
    R["category_economics"] = cat_tbl

    conc = market.revenue_concentration(df)
    R["concentration"] = conc
    print("\nКонцентрация выручки:")
    for k, v in conc.items():
        print(f"   {k:34s} {fmt(v) if isinstance(v, (int, float)) else v}")

    brands = market.top_brands(df)
    R["top_brands"] = brands
    print("\nТоп брендов по оценке выручки:")
    print(brands.to_string())

    hhi = market.brand_concentration(df)
    R["brand_hhi"] = hhi
    print("\nКонцентрация брендов внутри категорий (HHI):")
    print(hhi.to_string(index=False))

    bands = market.price_band_profile(df)
    R["price_bands"] = bands
    print("\nЦеновые полосы:")
    print(bands.to_string())

    market.plot_category_economics(cat_tbl)
    market.plot_pareto(df, conc)
    market.plot_brand_hhi(hhi)

    # ------------------------------------------------------------ эластичность
    hr("2. ЦЕНА И СПРОС")
    mkt_el = pricing.market_elasticity(df)
    R["market_elasticity"] = mkt_el
    print("Эластичность по рынку целиком:")
    for k, v in mkt_el.items():
        print(f"   {k:26s} {v}")

    el_tbl = pricing.elasticity_by_category(df)
    R["elasticity"] = el_tbl
    print("\nПо категориям:")
    print(el_tbl.to_string(index=False))

    disc = pricing.discount_effect(df)
    R["discount"] = disc
    print("\nЭффект глубины скидки внутри ценовых полос:")
    print(disc.to_string(index=False))

    pricing.plot_elasticity(el_tbl, mkt_el)
    pricing.plot_price_demand(df)

    # ------------------------------------------------------------- метки
    hr("3. РЕКЛАМА, ЗНАЧКИ, КУПОНЫ — С ПОПРАВКОЙ НА СОСТАВ")
    treatments = ["is_sponsored", "is_best_seller", "is_amazons_choice",
                  "has_coupon", "has_sustainability_badge"]
    eff_tbl, eff_detail = promo.compare_treatments(df, treatments)
    R["effects"] = eff_tbl
    R["effect_details"] = eff_detail
    print(eff_tbl.to_string(index=False))

    print("\nНаивный профиль спонсорских товаров:")
    sp_prof = promo.naive_profile(df, "is_sponsored")
    R["sponsored_profile"] = sp_prof
    print(sp_prof.to_string())

    promo.plot_effects(eff_tbl)
    if "is_sponsored" in eff_detail and "balance" in eff_detail["is_sponsored"]:
        promo.plot_balance(eff_detail["is_sponsored"])

    # ------------------------------------------------------------ рейтинги
    hr("4. НАДЁЖНОСТЬ РЕЙТИНГОВ")
    rel = ratings.rating_reliability(df)
    R["rating_reliability"] = rel
    print(rel.to_string())

    raw_top, shr_top = ratings.top_by_rating(df)
    R["top_raw_rating"] = raw_top
    R["top_shrunk_rating"] = shr_top
    print("\nТоп по сырому рейтингу:")
    print(raw_top.to_string(index=False))
    print("\nТоп по сглаженному рейтингу:")
    print(shr_top.to_string(index=False))

    best, worst = ratings.outliers(df)
    R["rating_best"], R["rating_worst"] = best, worst
    ratings.plot_shrinkage(df)

    # ------------------------------------------------------------ сегменты
    hr("5. СЕГМЕНТЫ АССОРТИМЕНТА")
    seg_df, seg_diag = segments.fit_segments(df)
    R["segment_diag"] = seg_diag
    print(f"Выбрано кластеров: {seg_diag.get('k')} "
          f"(силуэт {seg_diag.get('silhouette')}); силуэт по k: {seg_diag.get('scores')}")
    if "error" not in seg_diag:
        prof = segments.profile(seg_df)
        names = segments.name_segments(prof)
        prof_named = prof.copy()
        prof_named.index = [names[i] for i in prof_named.index]
        R["segments"] = prof_named
        print(prof_named.to_string())
        segments.plot_segments(seg_df, prof, names)

    # ------------------------------------------------------------ модель
    hr("6. МОДЕЛЬ СПРОСА")
    X, y, thr = demand.build_xy(df)
    print(f"Обучающих объектов: {fmt(len(y))}")
    print(f"Порог «высокого спроса»: > {fmt(round(thr))} шт/мес "
          f"(доля положительного класса {y.mean():.1%})")

    cv = demand.cross_validate(X, y)
    R["model_cv"] = cv
    R["demand_threshold"] = thr
    print(f"Кросс-валидация AUC: {cv['auc_mean']:.4f} ± {cv['auc_std']:.4f} "
          f"(по фолдам: {cv['auc_folds']})")
    print(f"Average precision: {cv['ap_mean']:.4f} при базовой доле {cv['baseline_rate']:.4f}")

    clf, imp, holdout = demand.fit_and_importance(X, y)
    R["importance"] = imp
    R["holdout"] = {k: v for k, v in holdout.items() if k not in ("y_test", "p_test")}
    print("\nПерестановочная важность признаков:")
    print(imp.to_string(index=False))

    demand.plot_importance(imp)
    demand.plot_roc(holdout, cv)

    # ------------------------------------------------------------ текст
    hr("7. ЯЗЫК НАЗВАНИЙ")
    txt = text.fit_text_model(df)
    R["text"] = txt
    if "error" not in txt:
        print(f"AUC только по тексту названия: {txt['auc']:.4f} "
              f"({fmt(txt['n_features'])} n-грамм)")
        print("\nСопутствуют высокому спросу:")
        print(txt["top_positive"].to_string(index=False))
        print("\nСопутствуют низкому спросу:")
        print(txt["top_negative"].to_string(index=False))
        text.plot_terms(txt)

    # ------------------------------------------------------------ отчёт
    hr("8. СБОРКА ОТЧЁТА")
    quality_bits = pd.read_pickle(REPORTS / "_quality.pkl")
    path = report.build(R, quality_bits, df)
    print(f"Отчёт: {path}")
    print("\nГотово.\n")


if __name__ == "__main__":
    main()

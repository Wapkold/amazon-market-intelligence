# -*- coding: utf-8 -*-
"""Шаг 1: сырой CSV -> listings.csv + products.csv + отчёт о качестве данных.

Запуск:  python scripts/01_build_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from amazon_mi import cleaning, features, quality, taxonomy, viz
from amazon_mi.config import (DATA_RAW, LISTINGS_CSV, PRODUCTS_CSV, REPORTS,
                              ensure_dirs)


def fmt(n) -> str:
    """Разделитель разрядов неразрывным пробелом — только для самого числа."""
    return f"{n:,}".replace(",", " ") if isinstance(n, int) else str(n)


def main() -> None:
    ensure_dirs()
    viz.setup()

    print("=" * 78)
    print("ШАГ 1. СБОРКА ДАТАСЕТА")
    print("=" * 78)

    print(f"\n[1/5] Чтение {DATA_RAW.name}")
    raw = cleaning.load_raw(DATA_RAW)
    print(f"      {fmt(raw.shape[0])} строк x {raw.shape[1]} колонок")

    print("\n[2/5] Парсинг полей -> уровень размещений")
    listings, lstats = cleaning.build_listings(raw)
    for k, v in lstats.items():
        print(f"      {k:36s} {fmt(v)}")

    cov = taxonomy.coverage(listings)
    lstats["taxonomy_coverage_pct"] = cov["coverage_pct"]
    print(f"      {'покрытие категориями, %':36s} {cov['coverage_pct']} "
          f"({cov['n_categories']} категорий)")

    print("\n[3/5] Схлопывание до уровня товара (ASIN)")
    products, pstats = cleaning.build_products(listings)
    for k, v in pstats.items():
        print(f"      {k:36s} {fmt(v)}")

    print("\n[4/5] Аудит качества данных")
    scorecard = quality.column_scorecard(raw)
    junk = quality.bought_junk_taxonomy(raw)
    impact = quality.entity_resolution_impact(listings, products)
    dup = quality.duplication_profile(products)

    print("\n--- Заполненность против валидности ---")
    print(scorecard.to_string(index=False))
    print("\n--- Из чего состоит мусор в колонке спроса ---")
    print(junk.to_string(index=False))
    print("\n--- Последствия счёта по строкам вместо товаров ---")
    print(impact.to_string(index=False))
    print("\n--- Как распределены показы по товарам ---")
    print(dup.to_string(index=False))

    quality.plot_scorecard(scorecard, len(raw))
    quality.plot_entity_impact(impact, dup)

    print("\n[5/5] Сохранение")
    products = features.add_features(products)
    products["rating_shrunk"] = features.empirical_bayes_rating(products)

    listings.drop(columns=[c for c in listings.columns if c.startswith("_")]).to_csv(
        LISTINGS_CSV, index=False, encoding="utf-8-sig")
    products.to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")
    print(f"      {LISTINGS_CSV.name}: {listings.shape}")
    print(f"      {PRODUCTS_CSV.name}: {products.shape}")

    # Отчёт о качестве в markdown
    md = REPORTS / "data_quality.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Отчёт о качестве данных\n\n")
        f.write(f"Исходная выгрузка: **{fmt(len(raw))} строк**, ")
        f.write(f"после удаления полных дублей: **{fmt(lstats['rows_listings'])} строк**, ")
        f.write(f"уникальных товаров: **{fmt(pstats['n_products'])}**.\n\n")
        f.write("## Заполненность против валидности\n\n")
        f.write(scorecard.to_markdown(index=False) + "\n\n")
        f.write("## Состав мусора в колонке спроса\n\n")
        f.write(junk.to_markdown(index=False) + "\n\n")
        f.write("## Последствия счёта по строкам вместо товаров\n\n")
        f.write(impact.to_markdown(index=False) + "\n\n")
        f.write("## Распределение показов по товарам\n\n")
        f.write(dup.to_markdown(index=False) + "\n")
    print(f"      {md.name}")

    # Ключевые числа для итогового отчёта
    pd.to_pickle({"listing_stats": lstats, "product_stats": pstats,
                  "scorecard": scorecard, "junk": junk,
                  "impact": impact, "dup": dup},
                 REPORTS / "_quality.pkl")

    print("\nГотово.\n")


if __name__ == "__main__":
    main()

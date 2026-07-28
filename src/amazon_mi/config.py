# -*- coding: utf-8 -*-
"""Пути и глобальные параметры проекта."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw" / "amazon_sales_raw.csv"
DATA_PROC = ROOT / "data" / "processed"
LISTINGS_CSV = DATA_PROC / "listings.csv"      # уровень строки выдачи (после очистки)
PRODUCTS_CSV = DATA_PROC / "products.csv"      # уровень товара (после схлопывания по ASIN)

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

RANDOM_STATE = 42

# Порог «высокого спроса» для классификационной модели: верхний квартиль
DEMAND_TOP_QUANTILE = 0.75

# Минимальное число товаров в категории, чтобы считать по ней отдельные оценки
MIN_CATEGORY_SIZE = 120


def ensure_dirs() -> None:
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

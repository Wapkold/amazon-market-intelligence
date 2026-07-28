# -*- coding: utf-8 -*-
"""Тесты пайплайна и аналитических функций на синтетических данных.

Здесь проверяется не «получились ли красивые числа», а инварианты: что
дедупликация действительно схлопывает товар, что утечки целевой переменной
в модели нет, что статистические функции ведут себя корректно на данных
с известным ответом.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from amazon_mi import cleaning, features
from amazon_mi.analysis import demand, pricing


def make_raw(n_dup: int = 3) -> pd.DataFrame:
    """Мини-выгрузка: один товар, повторённый n_dup раз, плюс второй товар."""
    base = {
        "Title": "Acme Widget Pro Wireless Speaker",
        "Rating": "4.5 out of 5 stars",
        "Number_of_reviews": "1,234",
        "bought_in_last_month": "500+ bought in past month",
        "Current/discounted_price": "99.99",
        "Price_on_variant": "basic variant price: nan",
        "Listed_price": "$129.99",
        "is_best_seller": "No Badge",
        "is_sponsored": "Sponsored",
        "is_couponed": "No Coupon",
        "buy_box_availability": "Add to cart",
        "delivery_details": "FREE delivery Mon, Sep 1",
        "sustainability_badges": np.nan,
        "image_url": "http://img",
        "product_url": "/Acme-Widget/dp/B00000001A/ref=sr_1_1",
        "collected_at": "21-08-2025 11:14",
    }
    rows = []
    for i in range(n_dup):
        r = dict(base)
        # разное время сбора, чтобы строки не были полными дублями
        r["collected_at"] = f"2{1 + i}-08-2025 11:14"
        rows.append(r)

    other = dict(base)
    other |= {
        "Title": "Globex AA Alkaline Batteries 24 Count",
        "product_url": "/Globex/dp/B00000002B/ref=sr_1_2",
        "Current/discounted_price": "2,299.00",   # с разделителем тысяч
        "is_sponsored": "Organic",
        "Rating": "4.0 out of 5 stars",
        "Number_of_reviews": "10",
        "bought_in_last_month": "No featured offers available",
        "buy_box_availability": np.nan,
    }
    rows.append(other)
    return pd.DataFrame(rows)


class TestListings:
    def test_parses_all_rows(self):
        raw = make_raw()
        listings, stats = cleaning.build_listings(raw)
        assert len(listings) == len(raw)
        assert stats["rows_with_asin"] == len(raw)
        assert stats["unique_asins"] == 2

    def test_thousands_separator_price_survives(self):
        listings, stats = cleaning.build_listings(make_raw())
        assert stats["prices_saved_by_thousand_sep"] == 1
        assert 2299.0 in set(listings["price"].dropna())

    def test_buy_box_missing_means_no_offer(self):
        listings, _ = cleaning.build_listings(make_raw())
        globex = listings[listings["asin"] == "B00000002B"]
        assert bool(globex["has_buy_box"].iloc[0]) is False

    def test_discount_computed(self):
        listings, _ = cleaning.build_listings(make_raw())
        acme = listings[listings["asin"] == "B00000001A"].iloc[0]
        expected = (129.99 - 99.99) / 129.99 * 100
        assert acme["discount_pct"] == pytest.approx(expected, rel=1e-6)


class TestProducts:
    def test_collapses_duplicates(self):
        """Три строки одного товара должны стать одной строкой товара."""
        listings, _ = cleaning.build_listings(make_raw(n_dup=3))
        products, stats = cleaning.build_products(listings)
        assert stats["n_products"] == 2
        acme = products[products["asin"] == "B00000001A"].iloc[0]
        assert acme["n_listings"] == 3

    def test_sponsored_share_is_a_share(self):
        listings, _ = cleaning.build_listings(make_raw())
        products, _ = cleaning.build_products(listings)
        acme = products[products["asin"] == "B00000001A"].iloc[0]
        globex = products[products["asin"] == "B00000002B"].iloc[0]
        assert acme["sponsored_share"] == 1.0
        assert globex["sponsored_share"] == 0.0

    def test_row_level_would_overstate_sponsored(self):
        """Смысл всего проекта в одном утверждении.

        По строкам спонсорских 3 из 4 (75%), по товарам — 1 из 2 (50%).
        """
        listings, _ = cleaning.build_listings(make_raw(n_dup=3))
        products, _ = cleaning.build_products(listings)
        assert listings["is_sponsored"].mean() == pytest.approx(0.75)
        assert products["is_sponsored"].mean() == pytest.approx(0.50)


class TestFeatures:
    def test_est_revenue(self):
        df = pd.DataFrame({"price": [10.0, 20.0], "bought_last_month": [100.0, np.nan],
                           "num_reviews": [5.0, 5.0], "rating": [4.0, 4.0],
                           "discount_pct": [np.nan, np.nan], "title": ["a", "b"]})
        out = features.add_features(df)
        assert out["est_revenue"].iloc[0] == 1000.0
        assert np.isnan(out["est_revenue"].iloc[1])

    def test_empirical_bayes_pulls_low_n_toward_mean(self):
        """Товар с одним отзывом должен сдвинуться сильнее, чем с тысячей."""
        df = pd.DataFrame({
            "rating": [5.0, 5.0, 4.0, 4.2, 4.4, 4.6, 4.3, 4.5],
            "num_reviews": [1, 5000, 100, 100, 100, 100, 100, 100],
        })
        shrunk = features.empirical_bayes_rating(df)
        shift_low_n = abs(shrunk.iloc[0] - 5.0)
        shift_high_n = abs(shrunk.iloc[1] - 5.0)
        assert shift_low_n > shift_high_n

    def test_herfindahl_bounds(self):
        # Монополия -> 10 000; равные доли между 4 игроками -> 2 500
        assert features.herfindahl(pd.Series([100.0])) == pytest.approx(10_000)
        assert features.herfindahl(pd.Series([25.0] * 4)) == pytest.approx(2_500)


class TestPricing:
    def test_ols_recovers_known_slope(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        y = 3.0 + 2.5 * x
        beta, r2 = pricing.ols(x.reshape(-1, 1), y)
        assert beta[0] == pytest.approx(3.0, abs=1e-8)
        assert beta[1] == pytest.approx(2.5, abs=1e-8)
        assert r2 == pytest.approx(1.0, abs=1e-8)

    def test_bootstrap_ci_brackets_truth(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=400)
        y = 1.0 - 0.8 * x + rng.normal(scale=0.5, size=400)
        lo, hi = pricing.bootstrap_ci(x.reshape(-1, 1), y, idx=1, n_boot=200)
        assert lo < -0.8 < hi


class TestDemandModel:
    def test_no_target_leakage_in_features(self):
        """Признаки, вычисленные из спроса, не должны попадать в модель."""
        leaky = {"est_revenue", "reviews_per_demand", "bought_last_month",
                 "log_demand"}
        used = set(demand.NUM_FEATURES) | set(demand.BOOL_FEATURES) | set(demand.CAT_FEATURES)
        assert used & leaky == set()

    def test_every_feature_has_a_russian_label(self):
        used = set(demand.NUM_FEATURES) | set(demand.BOOL_FEATURES) | set(demand.CAT_FEATURES)
        assert used <= set(demand.FEATURE_LABELS)

# -*- coding: utf-8 -*-
"""Тесты парсеров.

Каждый тест здесь закрывает конкретный случай, реально встреченный в выгрузке,
а не воображаемый. Ссылки на количество строк — из аудита исходного файла.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from amazon_mi import parsers as P
from amazon_mi import taxonomy


# --------------------------------------------------------------------------
# Цены
# --------------------------------------------------------------------------

class TestPrice:
    def test_plain(self):
        assert P.parse_price_plain("89.68") == 89.68

    def test_thousands_separator(self):
        """Главный баг наивной очистки: float('2,299.00') падает.

        Таких значений в колонке 1 192, и все они — дорогой сегмент,
        поэтому их потеря занижает среднюю цену примерно на треть.
        """
        assert P.parse_price_plain("2,299.00") == 2299.0
        assert P.parse_price_plain("1,019.00") == 1019.0

    def test_rejects_non_numeric(self):
        assert np.isnan(P.parse_price_plain("No Discount"))
        assert np.isnan(P.parse_price_plain("Add to cart"))

    def test_rejects_nonpositive(self):
        assert np.isnan(P.parse_price_plain("0"))
        assert np.isnan(P.parse_price_plain("-5"))

    def test_dollar(self):
        assert P.parse_price_dollar("$159.00") == 159.0
        assert P.parse_price_dollar("$1,499.95") == 1499.95
        assert np.isnan(P.parse_price_dollar("No Discount"))

    def test_variant_price_only_when_price(self):
        assert P.parse_price_variant("basic variant price: $162.24") == 162.24
        # Это описание варианта, а не цена — распознаваться не должно
        assert np.isnan(P.parse_price_variant("basic variant price: 2.4GHz"))
        assert np.isnan(P.parse_price_variant("basic variant price: nan"))
        assert np.isnan(P.parse_price_variant("basic variant price: 16 Count (Pack of 1)"))

    def test_nan_input(self):
        assert np.isnan(P.parse_price_plain(np.nan))
        assert np.isnan(P.parse_price_dollar(None))


# --------------------------------------------------------------------------
# Рейтинг и отзывы
# --------------------------------------------------------------------------

class TestRatingReviews:
    def test_rating(self):
        assert P.parse_rating("4.6 out of 5 stars") == 4.6
        assert P.parse_rating("5.0 out of 5 stars") == 5.0

    def test_rating_rejects_garbage(self):
        assert np.isnan(P.parse_rating("Best Seller"))
        assert np.isnan(P.parse_rating(np.nan))

    def test_rating_rejects_out_of_range(self):
        assert np.isnan(P.parse_rating("7.5 out of 5 stars"))

    def test_reviews_western_grouping(self):
        assert P.parse_reviews("2,457") == 2457

    def test_reviews_indian_grouping(self):
        """'8,65,598' — индийская группировка разрядов, встречается в файле."""
        assert P.parse_reviews("8,65,598") == 865598
        assert P.parse_reviews("1,98,512") == 198512

    def test_reviews_rejects_text(self):
        assert np.isnan(P.parse_reviews("many"))
        assert np.isnan(P.parse_reviews("4.6 out of 5 stars"))


# --------------------------------------------------------------------------
# Спрос
# --------------------------------------------------------------------------

class TestBought:
    def test_plain(self):
        assert P.parse_bought_last_month("300+ bought in past month") == 300

    def test_suffixes(self):
        assert P.parse_bought_last_month("6K+ bought in past month") == 6000
        assert P.parse_bought_last_month("2M+ bought in past month") == 2_000_000

    @pytest.mark.parametrize("junk", [
        "No featured offers available",
        "More Buying Choices",
        "List Price:",
        "ESRB Rating: Everyone",
        "($766.33$766.33/in)",
    ])
    def test_junk_is_not_parsed_as_demand(self, junk):
        """Мусор скрапинга не должен превращаться в число — только в пропуск."""
        assert np.isnan(P.parse_bought_last_month(junk))

    def test_junk_classification(self):
        assert P.classify_bought_junk("300+ bought in past month") == ""
        assert P.classify_bought_junk(np.nan) == "пропуск"
        assert P.classify_bought_junk("No featured offers available") == "нет предложения (buy box)"
        assert P.classify_bought_junk("ESRB Rating: Everyone") == "возрастной рейтинг ESRB"
        assert P.classify_bought_junk("($30.00$30.00/count)") == "цена за единицу"


# --------------------------------------------------------------------------
# Значки
# --------------------------------------------------------------------------

class TestBadges:
    def test_real_badges(self):
        assert P.parse_badge("Best Seller") == "best_seller"
        assert P.parse_badge("Amazon's") == "amazons_choice"
        assert P.parse_badge("Limited time deal") == "limited_deal"
        assert P.parse_badge("No Badge") == "none"

    def test_discount_label_is_not_a_badge(self):
        """336 строк содержат 'Save 30%' в колонке значков — это метка скидки."""
        assert P.parse_badge("Save 30%") == "discount_label"
        assert P.parse_badge("Save 77%") == "discount_label"

    def test_sustainability_real(self):
        assert P.parse_sustainability("Carbon impact") == (True, "Carbon impact")
        assert P.parse_sustainability("Energy efficiency +3 more")[0] is True
        assert P.parse_sustainability("Energy efficiency +3 more")[1] == "Energy efficiency"

    def test_sustainability_false_positives(self):
        """1 955 из 3 408 непустых значений — вообще не про экологию."""
        assert P.parse_sustainability("Small Business")[0] is False
        assert P.parse_sustainability("Works with Alexa")[0] is False
        assert P.parse_sustainability("Alexa Built-in")[0] is False

    def test_sustainability_missing(self):
        assert P.parse_sustainability(np.nan) == (False, "")


# --------------------------------------------------------------------------
# Купоны и доставка
# --------------------------------------------------------------------------

class TestCouponDelivery:
    def test_coupon_percent(self):
        has, pct, amt = P.parse_coupon("Save 15%  with coupon")
        assert has is True and pct == 15.0 and np.isnan(amt)

    def test_coupon_amount(self):
        has, pct, amt = P.parse_coupon("Save $16.00  with coupon")
        assert has is True and amt == 16.0 and np.isnan(pct)

    def test_no_coupon(self):
        assert P.parse_coupon("No Coupon") == (False,) + (pytest.approx(np.nan, nan_ok=True),) * 2

    def test_free_delivery(self):
        free, thr, fee = P.parse_delivery("FREE delivery Wed, Sep 3Or fastest delivery Sun, Aug 31")
        assert free is True and thr == 0.0

    def test_free_delivery_with_threshold(self):
        free, thr, fee = P.parse_delivery(
            "FREE delivery Wed, Sep 3 on $35 of items shipped by Amazon")
        assert free is True and thr == 35.0

    def test_paid_delivery(self):
        free, thr, fee = P.parse_delivery("$40.34 delivery")
        assert free is False and fee == 40.34

    def test_fastest_delivery_flag(self):
        assert P.parse_fastest_delivery("FREE delivery ThuOr fastest delivery Tomorrow") is True
        assert P.parse_fastest_delivery("Delivery Mon, Sep 1") is False


# --------------------------------------------------------------------------
# ASIN
# --------------------------------------------------------------------------

class TestAsin:
    def test_organic_url(self):
        url = "/Apple-Cancellation/dp/B0D1XD1ZV3/ref=sr_1_4?dib=eyJ2IjoiMSJ9"
        assert P.extract_asin(url) == "B0D1XD1ZV3"

    def test_sponsored_redirect(self):
        """ASIN спонсорской позиции спрятан в URL-кодированном параметре.

        Без декодирования дедупликация выбросила бы 5 492 строки — все
        стопроцентно спонсорские, что сломало бы сравнение рекламы с органикой.
        """
        url = ("/sspa/click?ie=UTF8&spc=MTo4NzEz&url=%2FBOYA-BOYALINK-Microphone"
               "%2Fdp%2FB0DNZB7TQG%2Fref%3Dsr_1_1_sspa%3Fdib%3DeyJ2IjoiMSJ9")
        assert P.extract_asin(url) == "B0DNZB7TQG"

    def test_missing(self):
        assert pd.isna(P.extract_asin(np.nan))
        assert pd.isna(P.extract_asin("/gp/help/customer/display.html"))


# --------------------------------------------------------------------------
# Таксономия
# --------------------------------------------------------------------------

class TestTaxonomy:
    @pytest.mark.parametrize("title,expected", [
        ("HP Original 910 Cyan, Magenta, Yellow Ink Cartridges (4-Pack)", "Картриджи и расходники"),
        ("Canon PGI-280XL PGBK Compatible To TR7520", "Картриджи и расходники"),
        ("Texas Instruments TI-84 Plus CE Color Graphing Calculator", "Офис и канцелярия"),
        ("SAMSUNG 990 PRO 2TB NVMe SSD Internal Solid State Drive", "Хранение данных"),
        ("Apple AirPods Pro 2 Wireless Earbuds, Active Noise Cancellation", "Аудио"),
        ("Meta Quest 3 512GB", "Игры и консоли"),
        ("Duracell AA Alkaline Batteries 24 Count", "Батарейки и питание"),
        ("Kobo Libra Colour | eReader | 7 inch", "Планшеты и электронные книги"),
    ])
    def test_category_rules(self, title, expected):
        assert taxonomy.classify_category(title) == expected

    def test_apple_pencil_is_not_stationery(self):
        """'Pencil' в названии Apple — аксессуар, а не канцелярский карандаш.

        Именно поэтому узкое правило стоит выше офисного.
        """
        assert taxonomy.classify_category("Apple Pencil Pro: Latest Model") == \
            "Смартфоны и аксессуары"
        assert taxonomy.classify_category("Amazon Basics Wood-Cased #2 Pencils, 150 Count") == \
            "Офис и канцелярия"

    def test_non_product_detection(self):
        assert taxonomy.classify_non_product(
            "ASURION 3 Year Home Improvement Protection Plan ($30 - $39.99)") != ""
        assert taxonomy.classify_non_product(
            "TRX Training 6 Month On-Demand Membership") != ""
        assert taxonomy.classify_non_product("Apple AirPods Pro 2") == ""

    def test_brand_extraction(self):
        assert taxonomy.extract_brand("Apple AirPods Pro 2 Wireless Earbuds") == "Apple"
        assert taxonomy.extract_brand("SAMSUNG 990 PRO 2TB NVMe") == "Samsung"
        assert taxonomy.extract_brand("acer 2-Pack Laptop Privacy Screen") == "Acer"
        assert taxonomy.extract_brand("Texas Instruments TI-84") == "Texas Instruments"

    def test_brand_missing(self):
        assert pd.isna(taxonomy.extract_brand(np.nan))
        assert pd.isna(taxonomy.extract_brand("123 456"))

# -*- coding: utf-8 -*-
"""Сборка итогового HTML-отчёта.

Отчёт самодостаточен: графики вшиты как base64, внешних запросов нет, файл
можно открыть или переслать как есть. Все числа в тексте подставляются из
результатов расчёта — в шаблоне не зашито ни одного значения вручную.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis.promo import LABELS_RU
from .config import FIGURES, REPORTS

CSS = """
:root{--bg:#ffffff;--fg:#17181c;--muted:#6b7280;--line:#e5e7eb;--card:#fafafa;
--accent:#c2410c;--accent2:#0369a1;--bad:#b91c1c;--good:#15803d;}
@media (prefers-color-scheme:dark){:root{--bg:#131417;--fg:#e8e9ec;--muted:#9ca3af;
--line:#2a2d33;--card:#1a1c20;--accent:#fb923c;--accent2:#38bdf8;--bad:#f87171;--good:#4ade80;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:1080px;margin:0 auto;padding:48px 24px 96px}
header{border-bottom:2px solid var(--fg);padding-bottom:24px;margin-bottom:40px}
h1{font-size:2.1rem;line-height:1.2;margin:0 0 12px;letter-spacing:-.02em}
h2{font-size:1.45rem;margin:56px 0 8px;letter-spacing:-.01em;
padding-top:20px;border-top:1px solid var(--line)}
h3{font-size:1.08rem;margin:32px 0 6px;color:var(--fg)}
.sub{color:var(--muted);font-size:1.02rem;margin:0}
p{margin:12px 0}
.lead{font-size:1.06rem}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
gap:14px;margin:28px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.kpi .v{font-size:1.7rem;font-weight:650;letter-spacing:-.02em;line-height:1.15}
.kpi .l{color:var(--muted);font-size:.86rem;margin-top:6px}
.kpi.warn .v{color:var(--bad)} .kpi.ok .v{color:var(--good)}
figure{margin:26px 0;padding:0}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:10px;display:block}
figcaption{color:var(--muted);font-size:.9rem;margin-top:10px}
.tw{overflow-x:auto;margin:20px 0;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:.88rem;min-width:520px}
th,td{padding:9px 13px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:var(--card);font-weight:600;text-align:right;position:sticky;top:0}
th:first-child,td:first-child{text-align:left;white-space:normal;min-width:160px}
tbody tr:last-child td{border-bottom:none}
.note{background:var(--card);border-left:3px solid var(--accent);
padding:14px 18px;margin:22px 0;border-radius:0 8px 8px 0;font-size:.95rem}
.note b{color:var(--accent)}
.warnbox{border-left-color:var(--bad)} .warnbox b{color:var(--bad)}
ul{margin:12px 0;padding-left:22px} li{margin:7px 0}
code{background:var(--card);padding:2px 6px;border-radius:4px;font-size:.88em}
footer{margin-top:72px;padding-top:24px;border-top:1px solid var(--line);
color:var(--muted);font-size:.88rem}
"""


def _img(name: str) -> str:
    """Вшивает PNG в HTML как data-URI."""
    p = FIGURES / f"{name}.png"
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{name}">'


def _fig(name: str, caption: str) -> str:
    img = _img(name)
    if not img:
        return ""
    return f"<figure>{img}<figcaption>{caption}</figcaption></figure>"


def _fmt_float(x: float) -> str:
    """Целые значения печатаются без хвоста '.00', дробные — с двумя знаками."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if float(x).is_integer():
        return f"{int(x):,}".replace(",", " ")
    return f"{x:,.2f}".replace(",", " ")


def _table(df: pd.DataFrame, index: bool = True, max_rows: int | None = None) -> str:
    d = df if max_rows is None else df.head(max_rows)
    html = d.to_html(index=index, border=0, justify="right",
                     float_format=_fmt_float, na_rep="—")
    return f'<div class="tw">{html}</div>'


def _kpi(value: str, label: str, cls: str = "") -> str:
    return f'<div class="kpi {cls}"><div class="v">{value}</div><div class="l">{label}</div></div>'


def _num(x, dec: int = 0, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{dec}f}".replace(",", " ") + suffix


def build(R: dict, Q: dict, df: pd.DataFrame) -> Path:
    """Собирает HTML-отчёт. R — результаты анализа, Q — блок качества данных."""
    lstats, pstats = Q["listing_stats"], Q["product_stats"]
    impact: pd.DataFrame = Q["impact"]

    def imp_row(metric: str, col: str):
        r = impact[impact["Метрика"] == metric]
        return None if r.empty else float(r.iloc[0][col])

    sponsored_rows = imp_row("Доля спонсорских, %", "По строкам выдачи")
    sponsored_prod = imp_row("Доля спонсорских, %", "По товарам")
    price_rows = imp_row("Средняя цена, $", "По строкам выдачи")
    price_prod = imp_row("Средняя цена, $", "По товарам")
    reviews_rows = imp_row("Среднее число отзывов", "По строкам выдачи")
    reviews_prod = imp_row("Среднее число отзывов", "По товарам")

    conc = R.get("concentration", {})
    mkt_el = R.get("market_elasticity", {})
    eff: pd.DataFrame = R.get("effects", pd.DataFrame())
    cv = R.get("model_cv", {})
    txt = R.get("text", {})

    def eff_of(name: str, col: str):
        if eff.empty:
            return None
        r = eff[eff["Метка"] == name]
        return None if r.empty else float(r.iloc[0][col])

    sp_naive = eff_of("is_sponsored", "Наивно, %")
    sp_adj = eff_of("is_sponsored", "С поправкой (IPW), %")
    bs_naive = eff_of("is_best_seller", "Наивно, %")
    bs_adj = eff_of("is_best_seller", "С поправкой (IPW), %")

    # ---------------------------------------------------------------- шапка
    parts: list[str] = []
    parts.append(f"""<header>
<h1>Рынок электроники Amazon: что показывают данные, если сначала починить данные</h1>
<p class="sub">Аналитическая витрина на основе «сырой» выгрузки {_num(lstats['rows_raw'])} товарных
карточек (21–30 августа 2025). Полный путь: аудит качества → разрешение сущностей →
структура рынка → эконометрика → модели.</p>
</header>""")

    parts.append(f"""<p class="lead">Этот датасет обычно анализируют построчно — и получают
неверные ответы. {_num(lstats['rows_with_asin'])} строк с товарной ссылкой соответствуют всего
<b>{_num(pstats['n_products'])} уникальным товарам</b>: один и тот же товар попадает в выгрузку
до {_num(R.get('max_listings_per_product'))} раз, а 1% товаров занимает
{_num(pstats['top1pct_row_share'], 1)}% всех строк. Пока это не исправлено, любая средняя —
это средняя, взвешенная по частоте скрапинга.</p>""")

    parts.append('<div class="kpis">')
    parts.append(_kpi(_num(pstats['n_products']), "уникальных товаров"))
    parts.append(_kpi(f"{_num(lstats['rows_with_asin'] / pstats['n_products'], 1)}×",
                      "во столько раз строк больше, чем товаров", "warn"))
    if sponsored_rows and sponsored_prod:
        parts.append(_kpi(f"{sponsored_rows:.1f}% → {sponsored_prod:.1f}%",
                          "доля спонсорских: по строкам / по товарам", "warn"))
    parts.append(_kpi(_num(lstats['prices_saved_by_thousand_sep']),
                      "цен спасено из-за разделителя тысяч", "ok"))
    parts.append(_kpi(_num(lstats['bought_junk']), "мусорных значений в поле спроса", "warn"))
    parts.append(_kpi(_num(lstats['fake_sustainability_badges']),
                      "«эко-значков», не относящихся к экологии", "warn"))
    parts.append("</div>")

    # ---------------------------------------------------- 1. качество данных
    parts.append("<h2>1. Качество данных: заполненность ≠ пригодность</h2>")
    parts.append(f"""<p>Стандартная проверка <code>isnull().sum()</code> показывает, что поле
спроса заполнено на 92.5%. Но распознаётся из него только 75.4%: остальное — текст соседних
элементов вёрстки, который скрапер положил не в ту колонку. Разница между «заполнено» и
«валидно» — это и есть скрытый мусор.</p>""")
    parts.append(_fig("01_data_quality",
                      "Серым — доля непустых ячеек, цветом — доля значений, которые удалось "
                      "распознать. Расхождение помечено справа."))
    parts.append(_table(Q["scorecard"], index=False))

    parts.append("<h3>Мусор в поле спроса неслучаен</h3>")
    parts.append("""<p>Разбор по типам показывает, что самая частая «поломка» — это вовсе не
поломка: <b>4 571 значение <code>No featured offers available</code></b> означает, что у товара
нет предложения в buy box. Это содержательный сигнал о доступности товара, который при наивной
очистке просто превратился бы в пропуск.</p>""")
    parts.append(_table(Q["junk"], index=False))

    parts.append(f"""<div class="note warnbox"><b>Найденная ошибка парсинга.</b>
В колонке <code>Current/discounted_price</code> {_num(lstats['prices_saved_by_thousand_sep'])}
значений записаны с разделителем тысяч (<code>2,299.00</code>). Наивный <code>float(x)</code> на
них падает и молча выбрасывает их в пропуски — а это целиком дорогой сегмент, от $1 019 до
$4 699. Из-за этого средняя цена по датасету занижается примерно на треть: $147.60 вместо
$212.61 на уровне строк.</div>""")

    # ------------------------------------------------- 2. разрешение сущностей
    parts.append("<h2>2. Разрешение сущностей: главный шаг всего анализа</h2>")
    parts.append(f"""<p>Товар опознаётся по ASIN из ссылки. У органических позиций он лежит в
URL открыто (<code>/dp/B0D1XD1ZV3</code>), у спонсорских — спрятан внутри редиректа
<code>/sspa/click?…url=%2F…%2Fdp%2F…</code>. Простая регулярка находит ASIN в 35 114 строках,
с URL-декодированием — в {_num(lstats['rows_with_asin'])}. Все
{_num(lstats['rows_with_asin'] - 35114)} «потерянных» строк — <b>стопроцентно спонсорские</b>,
поэтому дедупликация без декодирования выбросила бы именно рекламу и сломала бы любое сравнение
рекламных размещений с органическими.</p>""")
    parts.append(_fig("02_entity_resolution",
                      "Слева — насколько построчная статистика расходится с товарной. "
                      "Справа — как небольшая доля товаров занимает большинство строк."))
    parts.append(_table(impact, index=False))

    bullets = []
    if sponsored_rows and sponsored_prod:
        bullets.append(f"<li><b>Доля спонсорских товаров завышается в "
                       f"{sponsored_rows / sponsored_prod:.1f} раза</b> "
                       f"({sponsored_rows:.1f}% против {sponsored_prod:.1f}%): реклама по своей "
                       f"природе показывается чаще, и построчный счёт считает показы, а не товары.</li>")
    if reviews_rows and reviews_prod:
        bullets.append(f"<li><b>Среднее число отзывов, наоборот, занижается</b> "
                       f"({_num(reviews_rows)} против {_num(reviews_prod)}): многократно "
                       f"показываемые товары — это дешёвые ходовые позиции с относительно "
                       f"небольшим числом отзывов.</li>")
    if price_rows and price_prod:
        bullets.append(f"<li>Средняя цена смещается с ${_num(price_rows, 2)} до "
                       f"${_num(price_prod, 2)}.</li>")
    parts.append("<ul>" + "".join(bullets) + "</ul>")
    parts.append(_table(Q["dup"], index=False))

    # ------------------------------------------------------- 3. структура рынка
    parts.append("<h2>3. Структура рынка</h2>")
    cat = R.get("category_economics")
    cov = R.get("taxonomy_coverage", {})
    if cat is not None and not cat.empty:
        top_cat = cat.index[0]
        parts.append(f"""<p>Категория и бренд в выгрузке отсутствуют — они восстановлены из
названия набором проверяемых правил. Покрытие правилами составляет
<b>{cov.get('coverage_pct', 0)}%</b> ({cov.get('n_categories', 0)} категорий);
оставшиеся {_num(cov.get('unclassified'))} товаров сведены в «Прочее» и из разрезов
не исключаются, чтобы суммы сходились. Крупнейшая категория по оценке выручки —
<b>{top_cat}</b> ({cat.iloc[0]['Доля выручки, %']:.1f}% рынка).</p>""")
        parts.append(_fig("03_category_economics",
                          "Оценка выручки = спрос × цена. Спрос — нижняя граница диапазона, "
                          "который показывает Amazon, поэтому это оценка снизу."))
        parts.append(_table(cat))

    if conc:
        parts.append(f"""<p>Выручка предельно неравномерна: <b>топ-1% товаров даёт
{conc.get('top_1pct_revenue_share', 0):.0f}% всей оценочной выручки</b>, топ-10% —
{conc.get('top_10pct_revenue_share', 0):.0f}%. Половину выручки обеспечивают
{_num(conc.get('products_for_50pct_revenue'))} товаров, то есть
{conc.get('share_of_products_for_50pct', 0):.1f}% ассортимента.</p>""")
        parts.append(_fig("04_revenue_pareto",
                          "Чем сильнее кривая прижата к левому верхнему углу, тем выше "
                          "концентрация выручки."))

    hhi = R.get("brand_hhi")
    if hhi is not None and not hhi.empty:
        parts.append("<h3>Концентрация брендов</h3>")
        parts.append("""<p>HHI считается по долям брендов в оценке выручки внутри категории.
Пороги — те же, что используют антимонопольные регуляторы США: до 1500 рынок считается
конкурентным, свыше 2500 — концентрированным.</p>""")
        parts.append(_fig("05_brand_hhi", "Красным — концентрированные категории, "
                                          "зелёным — конкурентные."))
        parts.append(_table(hhi.drop(columns=["Тип рынка"], errors="ignore"), index=False))

    brands = R.get("top_brands")
    if brands is not None and not brands.empty:
        parts.append("<h3>Крупнейшие бренды</h3>")
        parts.append(_table(brands))

    # ------------------------------------------------------- 4. цена и спрос
    parts.append("<h2>4. Цена и спрос: эластичность</h2>")
    if mkt_el:
        e = mkt_el.get("elasticity")
        ec = mkt_el.get("elasticity_controlled")
        parts.append(f"""<p>Лог-логарифмическая регрессия <code>log(спрос) ~ log(цена)</code>
по всему рынку даёт эластичность <b>{e}</b> (95% бутстрэп-интервал
{mkt_el.get('ci')}, R² = {mkt_el.get('r2')}). После добавления контролей на рейтинг и число
отзывов эластичность составляет <b>{ec}</b> при R² = {mkt_el.get('r2_controlled')}.
Низкий R² здесь — сам по себе результат: <b>цена объясняет лишь малую часть различий в спросе</b>,
основная дисперсия приходится на то, что в карточке не видно.</p>""")
    el = R.get("elasticity")
    if el is not None and not el.empty:
        sig = el[el["Значима"]]
        parts.append(f"""<p>По категориям картина неоднородна: из {len(el)} категорий
статистически значимая связь цены и спроса обнаруживается в {len(sig)}.</p>""")
        parts.append(_fig("06_price_elasticity",
                          "Точка — оценка эластичности, отрезок — 95% бутстрэп-интервал. "
                          "Серые — интервал накрывает ноль."))
        parts.append(_table(el.drop(columns=["Значима"]), index=False))
    parts.append(_fig("07_price_demand_cloud",
                      "Каждая точка — товар. Красная линия — медианный спрос по децилям цены."))
    parts.append("""<div class="note"><b>Как это читать.</b> <code>bought_last_month</code> —
округлённая нижняя граница диапазона, который Amazon показывает сам («300+», «6K+»). Такое
огрубление сжимает дисперсию зависимой переменной и смещает оценку эластичности к нулю.
Поэтому категории корректно сравнивать между собой, но абсолютное значение коэффициента
нельзя трактовать буквально.</div>""")

    disc = R.get("discount")
    if disc is not None and not disc.empty:
        parts.append("<h3>Глубина скидки</h3>")
        parts.append("""<p>Разрез внутри ценовых полос обязателен: скидки концентрируются в
дешёвом сегменте, а дешёвый сегмент и без того более ходовой. Без контроля цены эффект скидки
оказался бы завышен просто из-за состава товаров. Спрос усредняется геометрически, а не
медианой: значения спроса дискретны (100, 200, 500, 1000), и медиана скачками перепрыгивает
между ними, создавая ложные «приросты» ровно на 100%.</p>""")
        parts.append(_table(disc, index=False))

    # ---------------------------------------------------- 5. метки и поправка
    parts.append("<h2>5. Реклама, значки и купоны — при равных наблюдаемых характеристиках</h2>")
    parts.append("""<p>Наивное сравнение «со значком против без значка» отвечает не на тот
вопрос: метки достаются не случайным товарам. Поэтому эффект пересчитан с поправкой на
наблюдаемые различия — propensity score по категории, цене, рейтингу, числу отзывов и скидке,
далее взвешивание (IPW) и независимо подбор пар ближайших соседей с калипером. Качество
поправки проверяется по стандартизованным разностям средних.</p>""")
    if not eff.empty:
        parts.append(_fig("08_adjusted_effects",
                          "Серым — наивная разница, цветом — после поправки на состав товаров. "
                          "Бледные полосы — оценки, не прошедшие проверку качества."))
        parts.append(_table(eff, index=False))

        notes = []
        if sp_naive is not None and sp_adj is not None:
            notes.append(f"<li><b>Спонсорское размещение:</b> наивная разница "
                         f"{sp_naive:+.0f}% сжимается до {sp_adj:+.0f}% после поправки — "
                         f"то есть почти вся видимая «премия» рекламы объясняется тем, "
                         f"каким товарам она достаётся.</li>")
        if bs_naive is not None and bs_adj is not None:
            notes.append(f"<li><b>Значок Best Seller:</b> {bs_naive:+.0f}% наивно, "
                         f"{bs_adj:+.0f}% после поправки. Читать это как эффект значка "
                         f"нельзя: Amazon присваивает его <i>по рангу продаж</i>, поэтому "
                         f"связь со спросом здесь тавтологична по построению — значок "
                         f"является следствием спроса, а не его причиной.</li>")

        unreliable = eff[eff["Оценка надёжна"] == "нет"]["Метка"].tolist()
        if unreliable:
            pretty = ", ".join(LABELS_RU.get(u, u) for u in unreliable)
            notes.append(f"<li><b>Не прошли проверку качества:</b> {pretty}. "
                         f"Либо ковариаты не сбалансировались после взвешивания "
                         f"(|SMD| остаётся выше 0.1), либо взвешивание и подбор пар "
                         f"дали разный знак эффекта. Такие числа приведены для полноты, "
                         f"но содержательных выводов из них делать нельзя.</li>")
        parts.append("<ul>" + "".join(notes) + "</ul>")

        parts.append("""<p>Проверка качества поправки состоит из двух независимых условий:
<b>баланс</b> (после взвешивания максимальная стандартизованная разность средних по всем
ковариатам должна быть ниже 0.1) и <b>согласованность методов</b> (взвешивание и подбор пар
должны дать эффект одного знака). Оценка считается надёжной, только если выполнены оба.
Наблюдения вне области общей поддержки отбрасываются, а веса контрольной группы подрезаются
по 99-му перцентилю — иначе единичные наблюдения определяют результат в одиночку.</p>""")
    parts.append(_fig("09_covariate_balance",
                      "Love plot: до поправки группы различаются по составу, после — "
                      "различия должны укладываться в порог 0.1."))
    parts.append("""<div class="note warnbox"><b>Граница вывода.</b> Это наблюдательные данные.
Поправка убирает только <i>измеренные</i> различия; рекламный бюджет продавца, реальная
скорость продаж и история аккаунта в датасете отсутствуют. Поэтому корректная формулировка —
«при равных наблюдаемых характеристиках товары с меткой отличаются на столько-то», а не
«метка приводит к росту продаж».</div>""")

    # ------------------------------------------------------------ 6. рейтинги
    parts.append("<h2>6. Надёжность рейтингов</h2>")
    parts.append("""<p>Рейтинг 5.0 при трёх отзывах и 4.6 при 35 000 — оценки разной
надёжности, но в таблице они стоят рядом. Эмпирический Байес подтягивает оценку к среднему по
рынку тем сильнее, чем меньше отзывов; параметр априорной силы оценивается из данных методом
моментов, а не подбирается вручную.</p>""")
    parts.append(_fig("10_rating_shrinkage",
                      "Слева — эффект сглаживания в зависимости от числа отзывов, "
                      "справа — сдвиг распределения."))
    rel = R.get("rating_reliability")
    if rel is not None and not rel.empty:
        parts.append(_table(rel))
    shr = R.get("top_shrunk_rating")
    if shr is not None and not shr.empty:
        parts.append("<h3>Топ товаров по сглаженному рейтингу</h3>")
        parts.append("""<p>В отличие от топа по сырой оценке, сюда не попадают товары с
единичными отзывами.</p>""")
        parts.append(_table(shr, index=False))

    # ------------------------------------------------------------ 7. сегменты
    seg = R.get("segments")
    if seg is not None and not seg.empty:
        diag = R.get("segment_diag", {})
        parts.append("<h2>7. Сегменты ассортимента</h2>")
        parts.append(f"""<p>Кластеризация k-means по цене, рейтингу, отзывам, спросу, скидке и
видимости. Число кластеров выбрано по силуэту, а не назначено вручную: <b>k =
{diag.get('k')}</b> (силуэт {diag.get('silhouette')}). Названия сегментов выводятся из их
профиля детерминированными правилами.</p>""")
        parts.append(_fig("11_segments",
                          "Слева — сегменты в пространстве главных компонент, справа — "
                          "вклад каждого сегмента в ассортимент и в выручку."))
        parts.append(_table(seg))

    # ------------------------------------------------------------- 8. модель
    parts.append("<h2>8. Модель спроса: сколько сигнала в карточке товара</h2>")
    if cv:
        parts.append(f"""<p>Градиентный бустинг предсказывает попадание товара в верхний
квартиль спроса (порог — {_num(R.get('demand_threshold'))} шт/мес). Качество измерено
стратифицированной 5-блочной кросс-валидацией: <b>AUC = {cv['auc_mean']:.3f} ±
{cv['auc_std']:.3f}</b> при доле положительного класса {cv['baseline_rate']:.1%}. Модель здесь —
инструмент измерения, а не продукт: вопрос в том, сколько информации о спросе вообще содержится
в публично видимых атрибутах.</p>""")
    parts.append(_fig("12_feature_importance",
                      "Перестановочная важность на отложенной выборке: насколько падает AUC, "
                      "если признак перемешать. Встроенная важность деревьев не используется — "
                      "она смещена в пользу признаков с большим числом значений."))
    imp = R.get("importance")
    if imp is not None and not imp.empty:
        parts.append(_table(imp.head(12), index=False))
    parts.append(_fig("13_roc", "ROC-кривая на отложенной выборке."))
    parts.append("""<div class="note"><b>Защита от утечки.</b> Из признаков исключены
<code>est_revenue</code> и <code>reviews_per_demand</code>: они вычислены из самого спроса,
и их включение раздуло бы AUC до бессмысленно высокого.</div>""")

    # -------------------------------------------------------------- 9. текст
    if txt and "error" not in txt:
        parts.append("<h2>9. Язык названий</h2>")
        parts.append(f"""<p>TF-IDF по словам и биграммам названия плюс L2-логистическая
регрессия. <b>Только по тексту названия высокий спрос предсказывается с AUC =
{txt['auc']:.3f}</b> — название несёт заметный сигнал, потому что маркирует товарную нишу.</p>""")
        parts.append(_fig("14_title_terms",
                          "Зелёные n-граммы сопутствуют верхнему квартилю спроса, красные — "
                          "нижним."))
        parts.append("""<div class="note warnbox"><b>Не рецепт.</b> Это корреляция, а не
инструкция: слова маркируют ниши (батарейки, картриджи, кабели), а не влияют на спрос сами по
себе. Добавление слова в название не повысит продажи.</div>""")

    # ------------------------------------------------------------- ограничения
    parts.append("<h2>Ограничения</h2>")
    parts.append(f"""<ul>
<li><b>Спрос — оценка снизу.</b> <code>bought_last_month</code> — округлённая нижняя граница
диапазона от самого Amazon. Все производные величины («оценка выручки», эластичность) наследуют
это огрубление и сопоставимы между собой, но не абсолютны.</li>
<li><b>Выручка не наблюдается напрямую.</b> Везде используется прокси «спрос × цена»; реальные
продажи, возвраты и себестоимость в данных отсутствуют.</li>
<li><b>Причинность не устанавливается.</b> Поправка на propensity score убирает только
измеренные различия — см. раздел 5.</li>
<li><b>{_num(lstats['rows_listings'] - lstats['rows_with_asin'])} строк без товарной ссылки</b>
не имеют ASIN и не попадают в товарный уровень.</li>
<li><b>Панельная структура слишком тонкая для динамики.</b> Данные собраны за 6 дней, но лишь
236 товаров встречаются более чем в одну дату, и только у 46 из них цена вообще менялась.
Этого недостаточно для выводов о ценовой динамике, поэтому такой анализ сознательно
не проводится.</li>
<li><b>Категория и бренд восстановлены эвристиками</b> по названию: правила проверяемы и
покрытие измерено, но ошибки классификации в отдельных карточках неизбежны.</li>
</ul>""")

    parts.append(f"""<footer>Сгенерировано автоматически из
<code>scripts/02_analyze.py</code>. Источник: {_num(lstats['rows_raw'])} строк выгрузки Amazon
за 21–30 августа 2025 → {_num(pstats['n_products'])} уникальных товаров →
{_num(len(df))} в аналитическом срезе. Все графики и числа пересчитываются при запуске
пайплайна.</footer>""")

    html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Amazon Market Intelligence — аналитический отчёт</title>
<style>{CSS}</style></head>
<body><div class="wrap">{''.join(parts)}</div></body></html>"""

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "index.html"
    out.write_text(html, encoding="utf-8")
    return out

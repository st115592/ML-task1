"""Russian narrative report built from the completed Servo experiment."""

import ast
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {
    "baseline": "Прогноз средним",
    "ols": "МНК, исходные признаки",
    "physical": "МНК, физические преобразования",
    "poly2": "Полином второй степени",
    "ridge": "Гребневая регрессия",
    "lasso": "Лассо-регрессия",
    "elastic": "Эластичная сеть",
}


def feature_name(name: str) -> str:
    """Use physical labels instead of program identifiers in the report."""
    if name == "intercept":
        return "Свободный член"
    result = name.replace("motor_", "Двигатель ").replace("screw_", "Передача ")
    return (
        result.replace("*", " · ")
        .replace("p^2", "p²")
        .replace("v^2", "v²")
        .replace("p v", "pv")
    )


def number(value: float, digits: int = 3) -> str:
    """Format report numbers using the Russian decimal separator."""
    value = 0.0 if abs(value) < 0.5 * 10 ** (-digits) else value
    return f"{value:.{digits}f}".replace(".", ",")


def detailed_metrics(table: pd.DataFrame, target: str) -> str:
    """Show every required metric without empty technical columns."""
    rows = []
    for family, label in NAMES.items():
        for split, split_name in (
            ("train", "Обучение"),
            ("CV", "Проверка"),
            ("test", "Тест"),
        ):
            row = table[
                (table.family == family)
                & (table.target == target)
                & (table.split == split)
            ].iloc[0]
            entry = {"Модель": label, "Выборка": split_name}
            for metric, heading in (("MAE", "MAE"), ("RMSE", "RMSE"), ("R2", "R²")):
                value = number(row[metric])
                if split == "CV":
                    value += " ± " + number(row[metric + "_std"])
                entry[heading] = value
            rows.append(entry)
    return pd.DataFrame(rows).to_markdown(index=False)


def draw_figures(root: Path) -> None:
    """Redraw saved results with Russian captions; do not fit any model."""
    results = root / "results"
    folder = results / "figures"
    folder.mkdir(exist_ok=True)
    diagnostic = pd.read_csv(results / "oof_diagnostics.csv")
    test = pd.read_csv(results / "test_predictions.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    def save(fig: plt.Figure, name: str) -> None:
        fig.tight_layout()
        fig.savefig(folder / name, dpi=170)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.scatter(
        diagnostic.y, diagnostic.prediction, alpha=0.55, label="Перекрёстная проверка"
    )
    ax.scatter(
        test.y, test.prediction, marker="x", color="crimson", label="Отложенный тест"
    )
    low = min(diagnostic.prediction.min(), test.prediction.min(), 0) - 0.2
    high = max(diagnostic.y.max(), test.y.max()) + 0.3
    ax.plot([low, high], [low, high], "k--", lw=1)
    ax.set(xlabel="Время нарастания в данных", ylabel="Предсказанное время нарастания")
    ax.legend()
    save(fig, "measured_predicted.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, column, label in zip(
        axes,
        ["prediction", "pgain", "vgain"],
        ["Предсказанное время", "Позиционное усиление p", "Скоростное усиление v"],
    ):
        ax.scatter(diagnostic[column], diagnostic.residual, alpha=0.6)
        ax.axhline(0, color="black", ls="--")
        ax.set(xlabel=label, ylabel="Остаток: значение − прогноз")
    save(fig, "residuals.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for family, color in (
        ("ridge", "steelblue"),
        ("lasso", "seagreen"),
        ("elastic", "mediumpurple"),
    ):
        grid = pd.read_csv(results / f"grid_{family}_raw.csv")
        parameters = grid.params.map(ast.literal_eval)
        keep = parameters.map(
            lambda p: p.get("regressor__pre__gain__poly__degree") == 2
            and not p.get("regressor__cross__enabled", False)
            and p.get("regressor__reg__l1_ratio", 0.8) == 0.8
        )
        selected = grid.loc[keep]
        alpha = parameters[keep].map(lambda p: p["regressor__reg__alpha"])
        axes[0].plot(
            alpha, -selected.mean_test_score, "o-", color=color, label=NAMES[family]
        )
        axes[0].plot(alpha, -selected.mean_train_score, "--", color=color, alpha=0.6)
    axes[0].set(
        xscale="log",
        xlabel="Коэффициент регуляризации α",
        ylabel="Средняя абсолютная ошибка",
        title="Сплошная линия — проверка; пунктир — обучение",
    )
    axes[0].legend(fontsize=9)
    for degree, family in ((1, "ols"), (2, "poly2")):
        grid = pd.read_csv(results / f"grid_{family}_raw.csv")
        row = grid.iloc[0]  # quadratic row 0 excludes category*gain
        axes[1].bar(
            degree - 0.16,
            -row.mean_train_score,
            width=0.32,
            color="steelblue",
            label="Обучение" if degree == 1 else None,
        )
        axes[1].bar(
            degree + 0.16,
            -row.mean_test_score,
            width=0.32,
            color="darkorange",
            label="Проверка" if degree == 1 else None,
        )
    axes[1].set(
        xticks=[1, 2], xlabel="Степень полинома", ylabel="Средняя абсолютная ошибка"
    )
    axes[1].legend()
    save(fig, "complexity.png")

    heat = diagnostic.pivot_table(
        index="vgain", columns="pgain", values="abs_error", aggfunc="mean"
    )
    counts = diagnostic.pivot_table(
        index="vgain", columns="pgain", values="abs_error", aggfunc="count"
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    im = ax.imshow(heat, origin="lower", aspect="auto", cmap="YlOrRd")
    for i in range(len(heat)):
        for j in range(len(heat.columns)):
            if pd.notna(heat.iloc[i, j]):
                ax.text(
                    j,
                    i,
                    f"{number(heat.iloc[i, j], 2)}\nn = {int(counts.iloc[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
    ax.set(
        xticks=range(len(heat.columns)),
        xticklabels=heat.columns,
        yticks=range(len(heat.index)),
        yticklabels=heat.index,
        xlabel="Позиционное усиление p",
        ylabel="Скоростное усиление v",
    )
    fig.colorbar(im, ax=ax, label="Средняя абсолютная ошибка")
    save(fig, "error_map.png")

    stable = pd.read_csv(results / "bootstrap_coefficients.csv", index_col=0)
    top = (
        stable.drop(index="intercept")
        .sort_values("median", key=lambda s: s.abs())
        .tail(12)
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.errorbar(
        top["median"],
        range(len(top)),
        xerr=[top["median"] - top.q025, top.q975 - top["median"]],
        fmt="o",
    )
    ax.axvline(0, color="gray", ls="--")
    ax.set(
        yticks=range(len(top)),
        yticklabels=[feature_name(n) for n in top.index],
        xlabel="Коэффициент: медиана и центральные 95% повторных оценок",
    )
    save(fig, "coefficient_stability.png")


def write_readable_report(root: Path) -> None:
    """Write the final scientific account from unchanged numerical results."""
    results = root / "results"
    table = pd.read_csv(results / "metrics.csv")
    selection = json.loads(
        (results / "selection_frozen.json").read_text(encoding="utf-8")
    )
    audit = json.loads((results / "data_audit.json").read_text(encoding="utf-8"))
    diag = json.loads((results / "diagnostics.json").read_text(encoding="utf-8"))
    chosen = selection["family"]
    beta = pd.read_csv(results / "coefficients.csv", index_col=0).iloc[:, 0]
    stable = pd.read_csv(results / "bootstrap_coefficients.csv", index_col=0)
    ols = pd.read_csv(results / "bootstrap_ols_same_basis.csv", index_col=0)
    cv = pd.read_csv(results / "nested_selection.csv")
    ablation = pd.read_csv(results / "ablation_train_cv.csv")
    ablation_mean = ablation.groupby("basis").MAE.mean()
    paired = ablation.pivot(index="fold", columns="basis", values="MAE")
    # Preserve the existing derived artifact for reproducibility.
    pd.DataFrame(
        {
            "add_pv": paired.quadratic_with_pv - paired.squares_without_pv,
            "add_category_gain": paired.quadratic_with_category_gain
            - paired.quadratic_with_pv,
        }
    ).to_csv(results / "ablation_paired_differences.csv")
    groups = pd.read_csv(results / "oof_by_pgain.csv")
    regime = pd.read_csv(results / "regime_metrics.csv")

    def metric(family: str, target: str, split: str) -> pd.Series:
        return table[
            (table.family == family) & (table.target == target) & (table.split == split)
        ].iloc[0]

    comparison = []
    for family, label in NAMES.items():
        raw = metric(family, "raw", "CV")
        log = metric(family, "log", "CV")
        comparison.append(
            {
                "Модель": label,
                "Проверочная MAE, цель y": number(raw.MAE)
                + " ± "
                + number(raw.MAE_std),
                "Проверочная MAE, цель ln(y)": number(log.MAE)
                + " ± "
                + number(log.MAE_std),
            }
        )
    final_rows = []
    for split, label in (
        ("train", "Обучение"),
        ("CV", "Перекрёстная проверка"),
        ("test", "Отложенный тест"),
    ):
        row = metric(chosen, "raw", split)
        final_rows.append(
            {
                "Оценка": label,
                "MAE": number(row.MAE),
                "RMSE": number(row.RMSE),
                "R²": number(row.R2),
            }
        )
    important = ["p", "p^2", "motor_D", "screw_D", "p v", "screw_C*v"]
    stability_rows = []
    for name in important:
        r, other = stable.loc[name], ols.loc[name]
        sign = "−" if r.negative_fraction > r.positive_fraction else "+"
        frequency = max(r.negative_fraction, r.positive_fraction) * 100
        stability_rows.append(
            {
                "Коэффициент при": feature_name(name),
                "Медиана, лассо": number(r["median"]),
                "Диапазон 95%, лассо": f"[{number(r.q025)}; {number(r.q975)}]",
                "Частый знак и его доля": f"{sign} ({number(frequency, 1)}%)",
                "Диапазон 95%, МНК": f"[{number(other.q025)}; {number(other.q975)}]",
            }
        )
    regime_rows = []
    for model, label in (
        ("regime_model", "Без режима p = 3 в обучении"),
        ("ordinary_model", "С режимом p = 3 в обучении"),
    ):
        for name, description in (
            ("interpolation_p_gt_3", "p > 3"),
            ("extrapolation_p_eq_3", "p = 3"),
        ):
            r = regime[(regime.model == model) & (regime.regime == name)].iloc[0]
            regime_rows.append(
                {
                    "Модель": label,
                    "Проверяемый режим": description,
                    "Число примеров": int(r.n_test),
                    "MAE": number(r.MAE),
                    "RMSE": number(r.RMSE),
                    "R²": number(r.R2),
                }
            )
    coefficient_rows = []
    for prefix, label in (("motor_", "Двигатель"), ("screw_", "Передача")):
        for level in "BCDE":
            name = prefix + level
            coefficient_rows.append(
                {
                    "Конфигурация": label + " " + level,
                    "a": number(beta[name], 6),
                    "b": number(beta[name + "*p"], 6),
                    "c": number(beta[name + "*v"], 6),
                }
            )
    stable_count = int(
        (
            (stable.drop(index="intercept").positive_fraction >= 0.95)
            | (stable.drop(index="intercept").negative_fraction >= 0.95)
        ).sum()
    )
    active = int((beta.drop(index="intercept").abs() > 1e-9).sum())
    draw_figures(root)
    report = rf"""# Прогноз времени нарастания сервопривода линейной моделью

## 1. Задача и данные

Цель работы — предсказать время нарастания переходного процесса сервопривода
по настройкам усиления и типу механической передачи. Требуется выяснить,
может ли линейная модель дать достаточно точное и устойчивое описание отклика
и в каких режимах её применение становится ненадёжным.

Использован набор Servo из репозитория UCI: 167 наблюдений, полученных Карлом
Ульрихом в MIT в 1986 году и переданных в репозиторий Россом Куинланом в 1993 году.
Описанная установка включает усилитель, двигатель, винтовую передачу и подвижную
каретку. По сведениям автора, данные получены при моделировании системы.

Время нарастания характеризует быстроту реакции привода на ступенчатое изменение
заданного положения. В исходном описании такая интерпретация отклика названа
предположительной; точное определение времени и единицы измерения не приведены.
Поэтому далее время и ошибки выражены **в единицах исходного набора данных**,
без перевода в секунды. Отсутствие единиц не позволяет выполнить перевод
в систему СИ.

| Обозначение в данных | Смысл | Наблюдаемые значения |
|---|---|---|
| `motor` | Тип двигателя | A, B, C, D, E |
| `screw` | Тип винтовой передачи | A, B, C, D, E |
| `pgain`, далее p | Настройка позиционного усиления | 3, 4, 5, 6 |
| `vgain`, далее v | Настройка скоростного усиления | 1, 2, 3, 4, 5 |
| y | Время нарастания | От {number(audit['target_min'])} до {number(audit['target_max'])} |

Единицы усилений также не указаны. Их названия трактуются по обозначениям
переменных; подробной схемы регулятора нет. Пропусков и повторяющихся
конфигураций в данных не обнаружено. Буквы A–E обозначают разные конструкции,
а не числовую шкалу: например, двигатель E не считается «больше» двигателя A.

## 2. Физические предположения и выбор признаков

До обучения сформулированы три проверяемых предположения:

1. При прочих равных увеличение позиционного усиления обычно сокращает время нарастания.
2. Влияние скоростного усиления зависит от позиционного усиления: их эффекты нельзя полностью разделить.
3. Разные двигатели и передачи по-разному реагируют на изменение усилений.

Для проверки первого предположения сопоставлялись наблюдения с одинаковыми
двигателем, передачей и скоростным усилением. Второе и третье проверялись
сравнением моделей с соответствующими произведениями признаков и без них.

Модели для исходного времени линейны по оцениваемым коэффициентам:

$$
\hat y=\beta_0+\sum_j\beta_j\varphi_j(x).
$$

При этом сами признаки могут быть нелинейными функциями входных величин.
В работе сравнивались следующие варианты:

* **Базовый прогноз:** одно и то же среднее время для всех наблюдений.
* **Метод наименьших квадратов (МНК):** исходные усиления и типы механики.
* **МНК с физически мотивированными преобразованиями:** вместо p и v
  использованы 1/p и v/p. Обратная величина проверяет предположение об ускорении
  отклика с ростом усиления, а отношение — о совместном влиянии двух настроек.
  Это приближённое описание, а не выведенный закон движения привода.
* **Полином второй степени:** добавлены p², pv и v². Отдельно проверены
  произведения усилений на признаки типа двигателя и передачи. Они позволяют
  наклону зависимости отличаться у разных механических конфигураций.
* **Регуляризованные модели:** гребневая регрессия, лассо-регрессия и эластичная
  сеть. Они ограничивают величины коэффициентов; лассо также может обращать
  часть из них в ноль. Это снижает чувствительность модели к малой выборке.

МНК подбирает коэффициенты по минимуму суммы квадратов ошибок. Гребневая
регрессия добавляет штраф за сумму квадратов коэффициентов, лассо — за сумму
их модулей, а эластичная сеть сочетает оба штрафа. Свободный член не штрафуется.

Для каждой категориальной переменной использованы четыре индикатора «0 или 1»;
вариант A принят за исходный уровень. Получено 10 признаков в исходной и физически
мотивированной моделях, 13 — с квадратами и произведением усилений и 29 —
при добавлении взаимодействий с механикой. Полином третьей степени не применялся:
объём данных мал, а оснований для дальнейшего усложнения не получено.

Преобразования усилений выполнялись относительно одной единицы их исходных шкал.
Для логарифма использовано ln(y/y₀), где y₀ равен одной единице времени набора.
Перед регуляризацией из каждого признака вычиталось среднее, а результат
делился на стандартное отклонение. Обе величины рассчитывались только
по обучающим наблюдениям.

## 3. Как проверялись модели

Сначала случайным образом отложены 34 наблюдения для окончательного теста.
Остальные 133 использованы для обучения и выбора модели. Тестовые значения
не участвовали ни в выборе признаков, ни в настройке коэффициента регуляризации.

На обучающей части проведена **вложенная перекрёстная проверка**. Во внешнем
цикле данные делились на пять частей: модель обучалась на четырёх и проверялась
на оставшейся. Разбиение повторено три раза, всего получено 15 внешних оценок.
Внутри каждого обучения ещё одна проверка по схеме «пять частей, три повтора»
выбирала признаки и силу регуляризации. По той же схеме выполнен окончательный
выбор на всех 133 обучающих наблюдениях. Обработка категорий, построение признаков
и стандартизация каждый раз выполнялись заново, без проверочной части.

Коэффициент регуляризации выбирался из 0,001; 0,01; 0,1; 1; 10; 100.
Для эластичной сети доля штрафа лассо составляла 0,2 или 0,8. Основной критерий —
средняя абсолютная ошибка. Среди близких по точности вариантов предпочтение
отдавалось меньшему числу признаков: допустимое отклонение от лучшей оценки
равнялось стандартному отклонению ошибок, делённому на √15. Это правило выбора
сложности, а не доверительный интервал: повторные разбиения зависимы.
Для воспроизведения разбиений заданы начальные значения генераторов 42
(внешнее разбиение и тест) и 142 (внутренний подбор).

Использованы три показателя качества:

* **MAE** — средняя абсолютная ошибка; показывает типичную величину отклонения.
* **RMSE** — корень из средней квадратичной ошибки; сильнее реагирует на большие промахи.
* **R²** — доля объяснённого разброса отклика. Отрицательное значение означает,
  что прогноз хуже константы, равной среднему значению проверяемой группы.

## 4. Точность и выбор итоговой модели

В таблице приведены ошибки внешней перекрёстной проверки. Запись «среднее ±
стандартное отклонение» показывает, насколько результат менялся между 15
разбиениями. Во всех случаях ошибка рассчитана в исходной шкале времени,
в том числе после обучения на логарифме.

{pd.DataFrame(comparison).to_markdown(index=False)}

Исходная линейная зависимость описывает данные недостаточно точно.
Обратные величины усилений улучшают результат, а квадратичные признаки и
взаимодействия с механикой дают дальнейшее снижение ошибки. Среди моделей
для исходного времени наименьшая средняя ошибка у лассо-регрессии.

**Итоговая модель — лассо-регрессия второй степени с взаимодействиями усилений
и типа механики, α = {number(selection['params']['regressor__reg__alpha'])}.** Из {diag['n_features']} коэффициентов
при признаках {active} ненулевые. Получены следующие оценки:

{pd.DataFrame(final_rows).to_markdown(index=False)}

Отдельно проверена вся процедура выбора между семействами моделей, а не только
лассо. Её внешняя MAE равна {number(cv.MAE.mean())} ± {number(cv.MAE.std())},
RMSE — {number(cv.RMSE.mean())} ± {number(cv.RMSE.std())},
R² — {number(cv.R2.mean())} ± {number(cv.R2.std())}. Тестовая ошибка меньше средней
проверочной, поэтому опираться только на благоприятный результат 34 тестовых
наблюдений было бы недостаточно.

Обучение на ln(y) улучшает точность: например, для лассо проверочная MAE снижается
до {number(metric('lasso', 'log', 'CV').MAE)}, а тестовая — до {number(metric('lasso', 'log', 'test').MAE)}.
Однако после обратного преобразования прогноз имеет вид exp(β₀ + Σβⱼφⱼ)
и уже не является линейным по коэффициентам в исходной шкале времени.
Поэтому он приведён для сравнения, а итоговая модель предсказывает непосредственно y.
Обратное преобразование выполнено без поправки на смещение среднего прогноза.
При обучении на логарифме базовый прогноз равен геометрическому среднему времени.

![Зависимость ошибки от сложности и регуляризации](results/figures/complexity.png)

На рисунке видно недообучение простой линейной модели: велики и обучающая,
и проверочная ошибки. Вторая степень уменьшает обе ошибки. Слишком сильная
регуляризация снова ухудшает прогноз. Для сравнения силы штрафа на левом
графике зафиксирован один и тот же набор из 13 признаков. Численные значения
α между методами напрямую не сопоставляются из-за разной нормировки штрафов.

## 5. Где модель ошибается

![Значения из набора данных и прогнозы](results/figures/measured_predicted.png)

На диагонали находятся точные прогнозы. Кружки показывают среднее трёх
проверочных прогнозов для каждого обучающего наблюдения; каждый из них
получен без использования этого наблюдения в обучении. Крестики — отдельный
отложенный тест. Такая усреднённая картина используется для диагностики,
а численные оценки качества выше рассчитаны по отдельным разбиениям.
Под значением из данных здесь понимается результат моделирования, а не новое
экспериментальное измерение.

![Остатки модели](results/figures/residuals.png)

Остаток — разность между значением из данных и прогнозом. В остатках видна
структура: они образуют несколько ветвей и зависят от предсказанного времени.
Ранговая корреляция с прогнозом равна {number(diag['prediction']['rho'])}.
Корреляции с p и v близки к нулю ({number(diag['pgain']['rho'])} и {number(diag['vgain']['rho'])}),
но это не означает отсутствия ошибки по режимам: особенно велик разброс при p = 3.

![Средняя абсолютная ошибка в разных режимах](results/figures/error_map.png)

В ячейках указаны средняя абсолютная ошибка и число наблюдений n.
Пустые ячейки означают отсутствие данных. При p = 3 ошибка составляет
в среднем {number(groups.loc[groups.pgain == 3, 'MAE'].iloc[0])}; при остальных значениях p —
от {number(groups.loc[groups.pgain > 3, 'MAE'].min())} до {number(groups.loc[groups.pgain > 3, 'MAE'].max())}.
Оценки по одной-двум точкам ненадёжны: диапазоны p и v заполнены неравномерно.

Есть и физически недопустимые результаты: на тесте получены три отрицательных
времени при p = 4, v = 1 и передаче C, для двигателей B, C и E.
Следовательно, даже внутри наблюдаемого диапазона нельзя считать модель
пригодной для любой конфигурации. Отрицательные прогнозы не заменялись нулём:
в таком случае окончательная зависимость перестала бы быть строго линейной.

## 6. Физические гипотезы и устойчивость коэффициентов

**Рост позиционного усиления обычно ускоряет отклик.** В обучающей выборке
найдено {audit['train_hypotheses']['matched_differences']} пар соседних значений p при одинаковых остальных
входах. В {number(100 * audit['train_hypotheses']['negative_fraction'], 1)}% случаев время уменьшалось, в остальных не менялось.
Гипотеза согласуется с данными. При этом у итогового полинома производная
по p отрицательна лишь в {number(100 * diag['negative_partial_p_fraction'], 1)}% обучающих конфигураций:
он не гарантирует монотонность во всём диапазоне.

**Совместное влияние двух усилений подтверждено слабо.** Добавление только pv
к модели с квадратами снижает среднюю проверочную MAE с
{number(ablation_mean['squares_without_pv'])} до {number(ablation_mean['quadratic_with_pv'])}.
Улучшение мало и не сохраняется на всех разбиениях. Поэтому уверенного
вывода о необходимости именно этого члена сделать нельзя.

**Зависимость от механической конфигурации заметна сильнее.** Разрешение
разных наклонов по усилениям для разных двигателей и передач снижает MAE
с {number(ablation_mean['quadratic_with_pv'])} до {number(ablation_mean['quadratic_with_category_gain'])}.
Такое улучшение согласуется с третьей гипотезой.

Устойчивость коэффициентов проверена на 300 повторных выборках из обучающих
наблюдений, сформированных с возвращением. В каждом повторе коэффициенты
оценивались заново при фиксированных признаках и силе штрафа. Таблица показывает
медиану и диапазон, содержащий центральные 95% оценок. Для сравнения приведён
МНК с теми же признаками, но без регуляризации.

{pd.DataFrame(stability_rows).to_markdown(index=False)}

![Разброс оценок коэффициентов](results/figures/coefficient_stability.png)

Знак сохраняется не менее чем в 95% повторов у {stable_count} из {diag['n_features']} коэффициентов.
Стабильно отрицательны, например, коэффициенты при p и при индикаторе двигателя D;
коэффициент при p² положителен. Для pv и ряда взаимодействий знак меняется
или коэффициент обращается в ноль. Отдельные оценки поэтому нельзя надёжно
трактовать как самостоятельные физические эффекты.

Одна из причин — близость признаков: корреляция p и p² составляет 0,994.
Разные сочетания их коэффициентов могут давать похожие прогнозы. Лассо
уменьшает разброс ряда оценок по сравнению с МНК, но не устраняет эту
неоднозначность. Приведённые диапазоны не учитывают повторный выбор семейства
и набора признаков, поэтому описывают устойчивость только выбранной модели.

## 7. Проверка на исключённом режиме

Чтобы проверить перенос на другой диапазон усиления, из обучающей части
удалены все наблюдения с p = 3. На оставшихся 96 примерах заново выполнен
выбор модели; выбрана гребневая регрессия. Она проверена на тех же отложенных
данных: 21 наблюдении с p > 3 и 13 наблюдениях с p = 3. Для сравнения приведена
основная модель, в обучении которой режим p = 3 присутствовал.

{pd.DataFrame(regime_rows).to_markdown(index=False)}

Без данных о режиме p = 3 ошибка при переносе в него возрастает до 1,837,
тогда как для представленного в обучении диапазона составляет 0,155.
Это показывает, что продолжение зависимости за нижнюю границу обучающего
диапазона не описывает резкий рост времени нарастания. Вместе с тем группы
малы и различаются по составу механики, поэтому это оценка надёжности прогноза,
а не доказательство причинного влияния одного усиления. Отрицательные R²
в отдельных группах также показывают, что высокий общий R² скрывает
неудовлетворительное описание некоторых режимов.

## 8. Выводы

Квадратичная лассо-регрессия описывает отклик лучше простой линейной модели.
На отложенном тесте её MAE равна {number(metric(chosen, 'raw', 'test').MAE)},
RMSE — {number(metric(chosen, 'raw', 'test').RMSE)}, R² — {number(metric(chosen, 'raw', 'test').R2)}.
Однако получить одновременно очень компактную, точную и устойчивую по всем
коэффициентам модель не удалось: из 29 признаков сохранены 26, а знак
устойчив только у 10 коэффициентов. Физически мотивированная модель из
10 признаков проще, но заметно уступает по точности.

Наиболее сложная область — малое позиционное усиление p = 3. Перенос в этот
режим без соответствующих обучающих данных ненадёжен. Дополнительный отказ
наблюдается при p = 4, v = 1 и передаче C: модель может предсказывать
отрицательное время. Поэтому полученная зависимость пригодна как приближённое
описание части наблюдаемых режимов, но не как универсальный закон сервопривода.

## Приложение. Подробные результаты и формула

В следующих таблицах приведены все три показателя для обучения, перекрёстной
проверки и отложенного теста. Знак ± относится только к разбросу между
проверочными разбиениями; это не доверительный интервал среднего.

### Обучение на исходном времени y

{detailed_metrics(table, 'raw')}

### Обучение на ln(y), ошибки после возврата к исходной шкале

{detailed_metrics(table, 'log')}

### Формула итоговой модели

При исходных шкалах p и v прогноз задаётся выражением

$$
\hat y={beta['intercept']:.6f}{beta['p']:+.6f}p{beta['v']:+.6f}v{beta['p^2']:+.6f}p^2{beta['p v']:+.6f}pv
+\Delta_M(p,v)+\Delta_S(p,v).
$$

Здесь ΔM и ΔS — поправки для двигателя и передачи. Для варианта A поправка
равна нулю. Для остальных вариантов она имеет вид a + bp + cv:

{pd.DataFrame(coefficient_rows).to_markdown(index=False)}

Коэффициент при v² равен нулю. Значения в формуле и таблице округлены
до шести знаков после запятой. Поправки относятся к выбранному полиному
и не являются характеристиками двигателей или передач вне этой модели.

**Источник данных:** [Ulrich, K. Servo. UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/87/servo).
DOI: 10.24432/C5Q30F. Лицензия CC BY 4.0.
"""
    (root / "REPORT.md").write_text(report, encoding="utf-8")

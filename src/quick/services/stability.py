r"""Расчёт устойчивости СМО и коэффициента запаса устойчивости.

Модуль формализует понятие устойчивости, принятое в исследовании, и даёт
две согласованные количественные оценки: мгновенную и интегральную.

**Определение.** Система считается устойчивой, если базовая характеристика
качества обслуживания ``a(t)`` на всём переходном интервале
:math:`[t_0, t_{пер}]` не опускается ниже критического уровня
:math:`a_{кр}`. В отличие от классического условия существования
стационарного режима (:math:`\rho < 1`), это определение фиксирует не факт
сходимости, а сохранение требуемого качества обслуживания в динамике: СМО
с конечным буфером и нетерпеливыми заявками стационарна при любых
интенсивностях, но в переходном режиме может нарушать требования к
качеству.

В роли ``a(t)`` выступает вероятность обслуживания
:math:`a(t)=1-P_{loss}(t)` либо пропускная способность :math:`A(t)`.
Критический уровень для вероятности обслуживания равен
:math:`a_{кр}=1-P_{доп}`, где :math:`P_{доп}` — допустимая доля потерь.

**Мгновенный коэффициент устойчивости**

.. math::

    K_{уст}(t)=\frac{a(t)}{a_{кр}}.

Значение :math:`K_{уст}(t)<1` означает, что в этот момент требование
нарушено. Эта величина рассчитывается в каждой временной точке и
выводится как отдельная характеристика системы.

**Интегральный коэффициент устойчивости**

.. math::

    K_{уст}=\frac{1}{a_{кр}\,t_{пер}}
        \int_{t_0}^{t_0+t_{пер}}a(t)\,dt,

то есть отношение среднего уровня характеристики за переходный процесс к
критическому уровню. :math:`K_{уст}=1{,}2` соответствует запасу 20%.

**Показатель запаса устойчивости R**

.. math::

    R=\frac{\int_{t_0}^{t_{пер}}a(t)\,dt-a_{кр}(t_{пер}-t_0)}
           {\bigl(a(t_0)-a_{кр}\bigr)(t_{пер}-t_0)}\cdot 100\%,

числитель — площадь между кривой ``a(t)`` и горизонталью
:math:`a_{кр}`, знаменатель нормирует её на максимально возможный избыток.
Если ``a(t)`` опускается ниже :math:`a_{кр}` хотя бы в одной точке,
система признаётся неустойчивой и ``R = 0``.

Формула воспроизведена по ВКР без изменений, вместе с её областью
применимости. Знаменатель считает наибольшим значением характеристики
начальное: в ВКР переходный процесс спадающий, и это так. Если же
``a(t)`` растёт — а в QUICK расчёт обычно стартует с пустой системы, —
то ``a(t_0)`` максимумом не является и ``R`` выходит за 100%, а при
``a(t_0)``, близком к ``a_кр``, обращается в бесконечность. Поэтому ``R``
сопоставим между вариантами системы только тогда, когда переходный
процесс начинается с наилучшего значения; в остальных случаях следует
опираться на ``K_уст``, у которого такого ограничения нет.

Длительность переходного процесса :math:`t_{пер}` определяется по выходу
``a(t)`` в окрестность установившегося значения с заданным относительным
допуском.
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain.enums import StabilityVerdict
from quick.domain.params.stability import (
    BORDERLINE_THRESHOLD,
    StabilityParams,
)


@dataclass(frozen=True)
class StabilityResult:
    """Результат оценки устойчивости системы за переходный интервал.

    Attributes:
        coefficient (float): Интегральный коэффициент устойчивости
            ``K_уст``.
        margin (float): Показатель запаса устойчивости ``R`` в процентах.
            Нормирован по начальному значению ``a(t_0)`` согласно ВКР,
            поэтому у растущего переходного процесса может превышать 100%.
        settling_time (float): Длительность переходного процесса
            ``t_пер``.
        minimum (float): Минимальное значение ``a(t)`` на интервале.
        critical_level (float): Критический уровень ``a_кр``.
        verdict (StabilityVerdict): Качественная оценка устойчивости.
        skipped_points (int): Число временных точек, в которых
            характеристика не определена и которые исключены из оценки.
    """

    coefficient: float
    margin: float
    settling_time: float
    minimum: float
    critical_level: float
    verdict: StabilityVerdict
    skipped_points: int = 0

    @property
    def holds_critical_level(self) -> bool:
        """Выдерживается ли критический уровень на всём интервале.

        Returns:
            bool: ``True``, если ``a(t) >= a_кр`` во всех точках.
        """
        return bool(self.minimum >= self.critical_level)


def calculate_settling_time(
    values: NDArray[np.float64],
    time_array: NDArray[np.float64],
    tolerance: float,
) -> float:
    r"""Оценивает длительность переходного процесса ``t_пер``.

    За установившееся принимается последнее значение ``a(t)``. Моментом
    выхода на режим считается первая точка, после которой характеристика
    больше не покидает коридор шириной ``tolerance`` вокруг этого
    значения:

    .. math::

        t_{пер}=\min\{t:\ |a(\tau)-a_\infty|\le
            \varepsilon|a_\infty|\ \ \forall \tau\ge t\}-t_0.

    Условие проверяется «до конца интервала», а не поточечно, чтобы
    колебательный переходный процесс с затухающими выбросами не был
    засчитан установившимся на первом же пересечении коридора.

    Args:
        values (NDArray[np.float64]): Значения характеристики во времени.
        time_array (NDArray[np.float64]): Массив временных точек.
        tolerance (float): Относительный допуск коридора.

    Returns:
        float: Длительность переходного процесса. Если характеристика не
            успевает выйти на режим, возвращается длина всего интервала.

    Raises:
        ValueError: Если массивы пусты или их длины не совпадают.
    """
    values = np.asarray(values, dtype=np.float64)
    time_array = np.asarray(time_array, dtype=np.float64)

    if values.shape != time_array.shape:
        raise ValueError(
            "Длины массивов характеристики и времени должны совпадать: "
            f"{values.shape} и {time_array.shape}."
        )

    if values.size == 0:
        raise ValueError("Массив значений характеристики пуст.")

    # Точки, в которых характеристика не определена, не могут ни
    # подтвердить выход на режим, ни опровергнуть его.
    finite_mask = np.isfinite(values)

    if not np.any(finite_mask):
        raise ValueError("Характеристика не определена ни в одной точке.")

    values = values[finite_mask]
    time_array = time_array[finite_mask]

    total_duration = float(time_array[-1] - time_array[0])

    if values.size == 1:
        return total_duration

    steady_value = float(values[-1])
    corridor = tolerance * abs(steady_value)

    # Нулевой коридор возникает при a_inf = 0 и делает условие невыполнимым,
    # поэтому в этом случае допуск берётся абсолютным.
    if np.isclose(corridor, 0.0):
        corridor = tolerance

    outside_corridor = np.abs(values - steady_value) > corridor

    if not np.any(outside_corridor):
        return 0.0

    last_excursion = int(np.max(np.flatnonzero(outside_corridor)))

    if last_excursion >= values.size - 1:
        return total_duration

    return float(time_array[last_excursion + 1] - time_array[0])


def calculate_stability_series(
    values: NDArray[np.float64], critical_level: float
) -> NDArray[np.float64]:
    r"""Вычисляет мгновенный коэффициент устойчивости ``K_уст(t)``.

    .. math::

        K_{уст}(t)=\frac{a(t)}{a_{кр}}.

    Args:
        values (NDArray[np.float64]): Значения характеристики ``a(t)``.
        critical_level (float): Критический уровень ``a_кр``.

    Returns:
        NDArray[np.float64]: Значения ``K_уст(t)``.

    Raises:
        ValueError: Если критический уровень неположителен.
    """
    if critical_level <= 0:
        raise ValueError("Критический уровень a_кр должен быть положителен.")

    return np.asarray(values, dtype=np.float64) / critical_level


def _integrate(values: NDArray[np.float64], time_array: NDArray[np.float64]) -> float:
    """Интегрирует характеристику по времени методом трапеций.

    Args:
        values (NDArray[np.float64]): Значения характеристики.
        time_array (NDArray[np.float64]): Массив временных точек.

    Returns:
        float: Значение определённого интеграла.
    """
    if values.size < 2:
        return 0.0

    return float(np.trapezoid(values, time_array))


def evaluate_stability(
    values: NDArray[np.float64],
    time_array: NDArray[np.float64],
    params: StabilityParams,
) -> StabilityResult:
    """Оценивает устойчивость системы за переходный интервал.

    Интегральные показатели считаются по отрезку ``[t_0, t_0 + t_пер]``:
    именно на переходном участке система может нарушить требования к
    качеству обслуживания, тогда как на установившемся участке значение
    характеристики уже постоянно и лишь размывало бы оценку.

    Args:
        values (NDArray[np.float64]): Значения характеристики ``a(t)``.
        time_array (NDArray[np.float64]): Массив временных точек.
        params (StabilityParams): Параметры оценки устойчивости.

    Returns:
        StabilityResult: Коэффициент устойчивости, запас, длительность
            переходного процесса и качественная оценка.

    Raises:
        ValueError: Если массивы пусты, их длины не совпадают или значения
            характеристики не определены ни в одной точке.
    """
    logger.info("Оценка устойчивости системы за переходный интервал")

    params.validate()

    values = np.asarray(values, dtype=np.float64)
    time_array = np.asarray(time_array, dtype=np.float64)

    if values.shape != time_array.shape:
        raise ValueError(
            "Длины массивов характеристики и времени должны совпадать: "
            f"{values.shape} и {time_array.shape}."
        )

    # Характеристика бывает не определена в отдельных точках: в расчёте по
    # одному датчику в начальный момент вся вероятность сосредоточена в
    # фазе другого датчика, поэтому lambda_j(t) = 0, а P_uns,j(t) = 0/0.
    # Такие точки исключаются из оценки — иначе NaN расходится по интегралу
    # и коэффициент теряет смысл, — а сама оценка начинается с момента,
    # когда характеристика определена.
    finite_mask = np.isfinite(values)
    skipped_points = int(values.size - np.count_nonzero(finite_mask))

    if np.count_nonzero(finite_mask) < 2:
        raise ValueError(
            "Характеристика определена менее чем в двух временных точках: "
            "оценить устойчивость невозможно."
        )

    if skipped_points:
        logger.warning(
            f"Характеристика не определена в {skipped_points} точках; "
            "они исключены из оценки устойчивости."
        )

    values = values[finite_mask]
    time_array = time_array[finite_mask]

    settling_time = calculate_settling_time(
        values, time_array, params.settling_tolerance
    )

    # Переходный участок: от начала расчёта до момента выхода на режим.
    # Нулевая длительность означает, что система установилась сразу; тогда
    # оценка выполняется по всему доступному интервалу, иначе интеграл
    # выродился бы в ноль.
    upper_bound = time_array[0] + settling_time
    transient_mask = time_array <= upper_bound

    if np.count_nonzero(transient_mask) < 2:
        transient_mask = np.ones_like(time_array, dtype=bool)

    transient_values = values[transient_mask]
    transient_time = time_array[transient_mask]
    duration = float(transient_time[-1] - transient_time[0])

    minimum = float(np.nanmin(transient_values))
    critical_level = params.critical_level

    if duration <= 0:
        # Расчёт выполнен в единственной точке: интегральные показатели
        # вырождаются в отношение значений.
        coefficient = float(transient_values[0]) / critical_level
    else:
        integral = _integrate(transient_values, transient_time)
        coefficient = integral / (critical_level * duration)

    margin = _calculate_margin(
        transient_values, transient_time, critical_level, duration
    )
    verdict = classify_stability(
        coefficient=coefficient,
        minimum=minimum,
        params=params,
    )

    logger.success(
        f"Устойчивость оценена: K_уст={coefficient:.4f}, R={margin:.2f}%, "
        f"t_пер={settling_time:.6g}, вердикт — {verdict.value.lower()}"
    )

    return StabilityResult(
        coefficient=coefficient,
        margin=margin,
        settling_time=settling_time,
        minimum=minimum,
        critical_level=critical_level,
        verdict=verdict,
        skipped_points=skipped_points,
    )


def _calculate_margin(
    values: NDArray[np.float64],
    time_array: NDArray[np.float64],
    critical_level: float,
    duration: float,
) -> float:
    """Вычисляет показатель запаса устойчивости ``R`` в процентах.

    Args:
        values (NDArray[np.float64]): Значения характеристики на переходном
            интервале.
        time_array (NDArray[np.float64]): Временные точки интервала.
        critical_level (float): Критический уровень ``a_кр``.
        duration (float): Длительность переходного интервала.

    Returns:
        float: Значение ``R`` в процентах. Ноль, если характеристика
            опускается ниже критического уровня. Может превышать 100%,
            если ``a(t)`` поднимается выше своего начального значения —
            см. ограничение формулы в описании модуля.
    """
    if duration <= 0:
        return 0.0

    # Правило ВКР: пересечение критического уровня обнуляет запас, потому
    # что гарантировать качество обслуживания на всём интервале уже нельзя.
    if float(np.nanmin(values)) < critical_level:
        return 0.0

    excess_area = _integrate(values, time_array) - critical_level * duration

    # Знаменатель взят буквально по формуле (1.15) ВКР: нормировка ведётся
    # по начальному значению характеристики, а не по её максимуму на
    # интервале. В ВКР переходный процесс спадающий (пропускная способность
    # падает с начального уровня после отказа), поэтому a(t_0) там и есть
    # наибольшее значение, и R не превышает 100%.
    #
    # ВНИМАНИЕ. В QUICK расчёт обычно стартует с пустой системы, и a(t)
    # растёт. Тогда a(t_0) — не максимум, и R выходит за 100% тем сильнее,
    # чем ниже старт: при a(t_0), близком к a_кр, знаменатель стремится к
    # нулю, и показатель теряет смысл. Это свойство исходной формулы, а не
    # ошибка расчёта: сравнивать R между собой имеет смысл только для
    # процессов, начинающихся с наилучшего значения. Нормировка по
    # max a(t) удержала бы R в пределах 100% и совпала бы с (1.15) на
    # спадающем процессе, но это была бы уже другая формула, поэтому здесь
    # сохранено определение ВКР.
    initial_excess = (float(values[0]) - critical_level) * duration

    if np.isclose(initial_excess, 0.0):
        return 0.0

    return float(excess_area / initial_excess * 100.0)


def classify_stability(
    coefficient: float,
    minimum: float,
    params: StabilityParams,
) -> StabilityVerdict:
    """Относит систему к одной из областей устойчивости.

    Проверка минимума первична: интегральный коэффициент усредняет
    характеристику по интервалу и может оставаться высоким при коротком,
    но недопустимом провале ниже критического уровня. Неопределённые
    значения трактуются как отсутствие устойчивости.

    Args:
        coefficient (float): Интегральный коэффициент устойчивости.
        minimum (float): Минимум характеристики на переходном интервале.
        params (StabilityParams): Параметры оценки устойчивости.

    Returns:
        StabilityVerdict: Качественная оценка устойчивости.
    """
    # Сравнения с NaN всегда ложны, поэтому без явной проверки
    # неопределённый коэффициент дошёл бы до последней ветки и был бы
    # объявлен устойчивым. Отсутствие данных — не свидетельство
    # устойчивости, поэтому такой случай трактуется консервативно.
    if not np.isfinite(coefficient) or not np.isfinite(minimum):
        logger.warning(
            "Коэффициент устойчивости не определён; система не может быть "
            "признана устойчивой."
        )
        return StabilityVerdict.UNSTABLE

    if minimum < params.critical_level:
        return StabilityVerdict.UNSTABLE

    if coefficient < BORDERLINE_THRESHOLD:
        return StabilityVerdict.UNSTABLE

    if coefficient < params.stable_threshold:
        return StabilityVerdict.BORDERLINE

    return StabilityVerdict.STABLE

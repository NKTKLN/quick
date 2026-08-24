"""Компонент выбора режима расчёта по датчикам MAP-системы.

Модель QUICK описывает состояние MAP-системы парой «уровень очереди k,
номер датчика i», поэтому одни и те же характеристики можно считать
двумя способами: суммируя вероятности по всем датчикам либо оставляя
состояния одного выбранного. Оба режима сосуществуют — индивидуальный
расчёт не заменяет агрегированный, а дополняет его.
"""

import streamlit as st

MODE_AGGREGATED = "Все датчики (агрегированно)"
MODE_PER_SENSOR = "Все датчики (по отдельности)"

SENSOR_MODE_HELP = (
    "**Агрегированно** — характеристики суммируются по всем датчикам: "
    "sum_i P(k, i, t). Это исходный режим расчёта.\n\n"
    "**По отдельности** — те же характеристики раскладываются на вклад "
    "каждого датчика и выводятся набором кривых.\n\n"
    "**Датчик j** — в каждой формуле суммирование по датчикам заменяется "
    "единственным слагаемым P(k, j, t): система рассматривается только "
    "через состояния выбранного датчика."
)


def sensor_label(sensor_index: int) -> str:
    """Возвращает подпись датчика по его внутреннему номеру.

    Внутри модели датчики нумеруются с нуля, в интерфейсе — с единицы.

    Args:
        sensor_index (int): Внутренний номер датчика.

    Returns:
        str: Подпись вида «Датчик 1».
    """
    return f"Датчик {sensor_index + 1}"


def render_sensor_mode(
    sensor_count: int, key: str = "sensor_mode"
) -> tuple[bool, int | None]:
    """Отображает выбор режима расчёта по датчикам.

    Args:
        sensor_count (int): Число датчиков MAP-процесса.
        key (str): Ключ Streamlit-виджета.

    Returns:
        tuple[bool, int | None]:
            - per_sensor (bool): Раскладывать характеристики по всем
              датчикам сразу.
            - sensor_index (int | None): Номер выбранного датчика или
              None в агрегированном и покомпонентном режимах.
    """
    options = [MODE_AGGREGATED, MODE_PER_SENSOR] + [
        sensor_label(index) for index in range(sensor_count)
    ]

    mode = st.selectbox(
        "Режим расчёта по датчикам",
        options,
        index=0,
        key=key,
        help=SENSOR_MODE_HELP,
    )

    if mode == MODE_AGGREGATED:
        return False, None

    if mode == MODE_PER_SENSOR:
        return True, None

    return False, options.index(mode) - 2

"""Модуль стратегии отображения прогресса для Streamlit.

Содержит реализацию StreamlitProgressStrategy интерфейса AbstractProgress
из основного пакета, использующую st.progress для визуализации.
"""

import time
from collections.abc import Iterable, Iterator, Sized
from typing import TypeVar

import streamlit as st
from loguru import logger

from quick.utils import AbstractProgress

T = TypeVar("T")


def _format_time(seconds: float) -> str:
    """Форматирует время в секундах в строку вида ЧЧ:ММ:СС или ММ:СС.

    Args:
        seconds (float): Время в секундах.

    Returns:
        str: Отформатированное время.
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"


class StreamlitProgressStrategy(AbstractProgress[T]):
    """Стратегия отображения прогресса для интерфейса Streamlit.

    Использует st.progress для визуального отображения прогресса выполнения задачи.
    """

    def wrap(self, iterable: Iterable[T], description: str = "Загрузка") -> Iterator[T]:
        """Оборачивает iterable и отображает прогресс выполнения в Streamlit.

        Args:
            iterable (Iterable[T]): Итерируемый объект, длину которого можно определить.
            description (str, optional): Описание задачи. По умолчанию "Загрузка".

        Yields:
            Iterator[T]: Элементы iterable, проходящие через этот генератор.
        """
        if not isinstance(iterable, Sized):
            logger.warning(
                f"Невозможно отобразить прогресс: объект типа {type(iterable)} "
                "не имеет длины"
            )
            yield from iterable
            return

        total = len(iterable)
        start_time = time.time()
        progressbar = st.progress(0, text=f"{description}: 0.0% [00:00<00:00]")

        for i, item in enumerate(iterable, start=1):
            yield item
            elapsed = time.time() - start_time
            eta = (elapsed / i * (total - i)) if i else 0
            percent = i / total * 100
            progressbar.progress(
                i / total,
                text=(
                    f"{description}: {percent:.1f}% "
                    f"[{_format_time(elapsed)}<{_format_time(eta)}]"
                ),
            )
        elapsed = time.time() - start_time
        logger.debug(
            f"Задача '{description}' завершена: {total} итераций за "
            f"{_format_time(elapsed)}"
        )

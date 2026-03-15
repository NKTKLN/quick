"""Модуль для организации и отображения прогресса выполнения итераций.

Содержит абстрактный интерфейс для реализации различных стратегий отслеживания
прогресса, а также реализацию для Streamlit.

Предоставляет менеджер для выбора стратегии и обёртки итерируемых объектов.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sized
from typing import Generic, TypeVar

import streamlit as st
from loguru import logger

T = TypeVar("T")


def _format_time(seconds: float) -> str:
    """Форматирует время в секундах в строку вида ЧЧ:ММ:СС или ММ:СС.

    Args:
        seconds (float): Время в секундах.

    Returns:
        str: Отформатированное время. Если есть часы — формат "ЧЧ:ММ:СС",
            иначе — "ММ:СС".
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"


class AbstractProgress(ABC, Generic[T]):
    """Абстрактный интерфейс стратегии отображения прогресса итерации.

    Определяет метод wrap для обёртывания итерируемого объекта с целью отслеживания
    прогресса.
    """

    @abstractmethod
    def wrap(
        self, iterable: Iterable[T], description: str = "Processing"
    ) -> Iterator[T]:
        """Оборачивает итерируемый объект для отслеживания прогресса выполнения.

        Args:
            iterable (Iterable[T]): Итерируемый объект, который нужно обернуть.
            description (str): Описание текущей операции. По умолчанию "Processing".

        Returns:
            Iterator[T]: Итератор, который выдаёт элементы из iterable с
            отслеживанием прогресса.
        """


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


class NoProgressStrategy(AbstractProgress[T]):
    """Стратегия без отображения прогресса.

    Просто передаёт элементы iterable без изменений.
    """

    def wrap(self, iterable: Iterable[T], _description: str = "") -> Iterator[T]:
        """Оборачивает iterable без отслеживания прогресса.

        Args:
            iterable (Iterable[T]): Итерируемый объект.
            _description (str, optional): Описание задачи (не используется).

        Yields:
            Iterator[T]: Элементы iterable без изменений.
        """
        logger.debug("NoProgressStrategy: прогресс не отображается")
        yield from iterable


class Progress:
    """Менеджер стратегии отображения прогресса.

    Позволяет переключать текущую стратегию отображения прогресса. По умолчанию не
    отображает прогресс (NoProgressStrategy).
    """

    _strategy: AbstractProgress = NoProgressStrategy()

    @classmethod
    def set_strategy(cls, strategy: AbstractProgress) -> None:
        """Устанавливает стратегию отображения прогресса.

        Args:
            strategy (AbstractProgress): Объект стратегии, реализующий метод wrap.
        """
        cls._strategy = strategy
        logger.debug(f"Установлена стратегия прогресса: {strategy.__class__.__name__}")

    @classmethod
    def wrap(
        cls, iterable: Iterable[T], description: str = "Processing"
    ) -> Iterator[T]:
        """Оборачивает iterable текущей стратегией отображения прогресса.

        Args:
            iterable (Iterable[T]): Итерируемый объект.
            description (str, optional): Описание задачи. По умолчанию "Processing".

        Returns:
            Iterator[T]: Итератор с прогрессом (или без, если стратегия NoProgress).
        """
        return cls._strategy.wrap(iterable, description)

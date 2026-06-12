"""Модуль для организации и отображения прогресса выполнения итераций.

Содержит абстрактный интерфейс для реализации различных стратегий отслеживания
прогресса и стратегию по умолчанию без отображения.

Предоставляет менеджер для выбора стратегии и обёртки итерируемых объектов.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import TypeVar

from loguru import logger

T = TypeVar("T")


class AbstractProgress[T](ABC):
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
        raise NotImplementedError


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

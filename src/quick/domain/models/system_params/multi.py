"""Модуль параметров для многоканальной СМО с уходом заявок.

Реализует класс MultiSystemParams, который описывает параметры системы массового
обслуживания с несколькими обслуживающими каналами и возможностью ухода заявок.
"""

from dataclasses import dataclass

from .base import BaseSystemParams


@dataclass
class MultiSystemParams(BaseSystemParams):
    """Базовые параметры многоканальной СМО с уходом нетерпеливых заявок.

    Attributes:
        processor_count (int): Количество обслуживающих каналов (m).
        Остальные параметры наследуются от BaseSystemParams.
    """

    processor_count: int

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах. # TODO
        """
        super().validate()
        if self.processor_count <= 0:
            raise ValueError("Переменная processor_count должна быть положительна.")

"""Модуль параметров многоканальных систем массового обслуживания (СМО).

Содержит классы параметров для многоканальных СМО без и с уходом заявок, включая
поддержку перебора интенсивности ухода заявок (ν).
"""

from dataclasses import dataclass

from app.domain.models.single import BasicSingleServerParams


@dataclass
class BasicMultiServerParams(BasicSingleServerParams):
    """Параметры многоканальной СМО без учета ухода заявок.

    Расширяет параметры одноканальной системы, добавляя количество
    обслуживающих каналов (процессоров).

    Attributes:
        processor_count (int): Количество обслуживающих каналов (m).
        Остальные параметры наследуются от BasicSingleServerParams.
    """

    processor_count: int

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if self.processor_count <= 0:
            raise ValueError("Переменная processor_count должна быть положительна.")


@dataclass
class MultiServerParams(BasicMultiServerParams):
    """Параметры многолинейной СМО с учетом ухода нетерпеливых заявок.

    Расширяет модель одноканальной системы с учетом множественных обслуживающих
    каналов (процессоров) и ухода заявок с фиксированной интенсивностью ν.

    Attributes:
        nu_rate (float): Интенсивность ухода нетерпеливых заявок из очереди (ν ≥ 0).
        Остальные параметры наследуются из BasicMultiServerParams.
    """

    nu_rate: float

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if self.nu_rate <= 0:
            raise ValueError("Интенсивность ν должна быть положительна.")
        if (
            self.initial_probabilities.shape[0]
            != self.max_customers + self.processor_count + 1
        ):
            raise ValueError(
                "Размер начальных вероятностей должен совпадать с максимальным числом "
                "заявок в системе + колличество процессоров + 1."
            )

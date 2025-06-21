"""Модуль параметров систем массового обслуживания с нетерпеливыми заявками.

Содержит классы параметров для одно- и многолинейных систем массового обслуживания
с учетом интенсивности ухода нетерпеливых заявок (ν), включая модели с MAP-потоками.

Предоставляет структуры данных для моделирования переходных процессов и анализа
характеристик производительности системы при наличии оттока заявок.
"""

from dataclasses import dataclass

from app.domain.base_params import (
    BasicMAPServerParams,
    BasicMultiServerParams,
    BasicSingleServerParams,
)


@dataclass
class SingleServerParams(BasicSingleServerParams):
    """Параметры однолинейной СМО с учетом ухода нетерпеливых заявок.

    Используются для моделирования систем, в которых часть заявок может покинуть
    очередь до обслуживания с фиксированной интенсивностью ν.

    Attributes:
        nu_rate (float): Интенсивность ухода нетерпеливых заявок из очереди (ν ≥ 0).
        Остальные параметры наследуются из BasicSingleServerParams.
    """

    nu_rate: float

    def validate(self):
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if self.nu_rate <= 0:
            raise ValueError("Интенсивность ν должна быть положительна.")
        if self.initial_probabilities.shape[0] != self.max_customers:
            raise ValueError(
                "Размер начальных вероятностей должен совпадать с максимальным числом "
                "заявок в системе."
            )


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

    def validate(self):
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


@dataclass
class MAPServerParams(BasicMAPServerParams):
    """Параметры СМО с MAP-потоками и уходом нетерпеливых заявок.

    Описывает систему с марковским модулированным пуассоновским входным потоком,
    а также оттоком нетерпеливых заявок. Включает матрицы интенсивностей переходов.

    Attributes:
        nu_rate (float): Интенсивность ухода заявок (ν ≥ 0).
        Остальные параметры наследуются от BasicMAPServerParams.
    """

    nu_rate: float

    def validate(self):
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if self.nu_rate <= 0:
            raise ValueError("Интенсивность ν должна быть положительна.")

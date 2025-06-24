"""Модуль параметров одноканальных систем массового обслуживания (СМО).

Содержит классы параметров для одноканальных СМО без учета ухода заявок,
с фиксированной интенсивностью ухода, а также для перебора интенсивности ухода (ν).
"""

from dataclasses import dataclass

from app.domain.models.base import BasicServerParams


@dataclass
class BasicSingleServerParams(BasicServerParams):
    """Параметры одноканальной СМО без учета ухода заявок.

    Содержит базовые параметры, описывающие входной поток, обслуживание,
    ограничения по числу заявок и начальные условия системы.

    Attributes:
        lambda_rate (float): Интенсивность поступления заявок (λ > 0).
        Остальные параметры наследуются от BasicServerParams.
    """

    lambda_rate: float

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if self.lambda_rate <= 0:
            raise ValueError("Интенсивность λ должна быть положительна.")


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

    def validate(self) -> None:
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

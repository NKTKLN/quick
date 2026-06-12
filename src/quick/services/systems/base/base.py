"""Модуль, содержащий абстрактный базовый класс для серверных систем массового обслуживания (СМО).

Включает в себя определение класса BaseServerSystem с основными методами и свойствами
для расчёта вероятностей состояний, интенсивности входного потока, коэффициентов загрузки
и других базовых характеристик СМО.
"""

from abc import ABC, abstractmethod

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain import (
    ComputationConfig,
)
from quick.domain.params import SystemParams
from quick.services.systems_behavior import BaseSystemBehavior


class BaseServerSystem(ABC):
    """Абстрактный базовый класс серверной системы массового обслуживания (СМО).

    Обеспечивает общий интерфейс и базовые свойства для вычисления вероятностей
    состояний, интенсивности поступления заявок и производных характеристик
    системы массового обслуживания.
    """

    def __init__(
        self,
        params: SystemParams,
        system_behavior: type[BaseSystemBehavior],
        config: ComputationConfig | None = None,
    ) -> None:
        """Инициализирует систему массового обслуживания с заданными параметрами.

        Args:
            params (ServerParams): Параметры СМО.
            system_behavior (BaseSystemBehavior): Зависимости СМО.
            config (ComputationConfig | None, optional): Конфигурация вычислений.
        """
        self.params = params
        self.config = config
        self.system_behavior = system_behavior(self.params)
        self._probabilities: NDArray[np.float64] | None = None
        self._lambda_rate: float | NDArray[np.float64] | None = None

        logger.debug(
            f"СМО {self.__class__.__name__} инициализирована с параметрами: "
            f"{params.__class__.__name__=}, {config.__class__.__name__=}"
        )

    @property
    def beta(self) -> float:
        """Отношение интенсивности ухода заявок к интенсивности обслуживания.

        Returns:
            float: Значение коэффициента beta = nu_rate / mu_rate.
        """
        return self.params.base_params.nu_rate / self.params.base_params.mu_rate

    @property
    def rho(self) -> float | NDArray[np.float64]:
        """Коэффициент загрузки системы.

        Returns:
            float | NDArray[np.float64] : Значение коэффициента загрузки rho = lambda_rate / mu_rate.
        """
        return self.lambda_rate / self.params.base_params.mu_rate

    @abstractmethod
    def calculate_probabilities(self) -> None:
        """Выполняет расчет вектора вероятностей состояний СМО.

        Метод должен быть реализован в подклассах, в соответствии с конкретной
        моделью СМО. Результат расчёта должен сохраняться во внутреннем
        поле `_probabilities`.
        """
        raise NotImplementedError

    @property
    def lambda_rate(self) -> float | NDArray[np.float64]:
        """Ленивая загрузка: возвращает интенсивность поступления заявок.

        Raises:
            ValueError: Если параметры имеют тип MAPSystemParams.
        """
        if self._lambda_rate is None:
            self._lambda_rate = self.system_behavior.lambda_rate
        return self._lambda_rate

    @property
    def probabilities(self) -> NDArray[np.float64]:
        """Ленивая загрузка: возвращает вероятности состояний или инициирует их расчёт.

        Returns:
            NDArray[np.float64]: Вектор вероятностей состояний системы.
        """
        if self._probabilities is None:
            self.calculate_probabilities()

        if self._probabilities is None:
            raise RuntimeError("Вероятности не были вычислены.")

        return self._probabilities

    @abstractmethod
    def calculate(self) -> dict[str, NDArray[np.float64]]:
        """Универсальный метод вычислений.

        Returns:
            dict[str, NDArray[np.float64]]: Результат вычислений.
        """
        raise NotImplementedError

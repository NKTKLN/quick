"""Модуль, содержащий абстрактный базовый класс для систем массового обслуживания (СМО).

Включает в себя определение класса BaseSystem с основными методами и свойствами
для расчёта вероятностей состояний, пропускной способности, средней нагрузки
и других ключевых характеристик СМО.
"""

from abc import ABC, abstractmethod
from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from app.domain import (
    CalculationMode,
    ComputationConfig,
)
from app.domain.models import SystemParams
from app.services.systems_behavior import BaseSystemBehavior


class BaseServerSystem(ABC):
    """Абстрактный базовый класс системы массового обслуживания (СМО).

    Обеспечивает интерфейс и основные методы для вычисления вероятностей состояний,
    пропускной способности, средних значений нагрузки и вероятностей отказа/обслуживания
    для различных моделей СМО.
    """

    def __init__(
        self,
        params: SystemParams,
        system_behavior: BaseSystemBehavior,
        config: ComputationConfig | None = None,
    ) -> None:
        """Инициализирует систему массового обслуживания с заданными параметрами.

        Args:
            params (ServerParams): Параметры СМО.
            config (ComputationConfig | None, optional): Конфигурация вычислений.
        """
        self.params = params
        self.config = config
        self.system_behavior = system_behavior(self.params)
        self._probabilities: NDArray[np.float64] | None = None
        self._lambda_rate: float | None = None

        logger.debug(
            f"СМО {self.__class__.__name__} инициализирована с параметрами: "
            f"{params.__class__.__name__}, "
            f"config={config.__class__.__name__}"
        )

    @property
    def beta(self) -> float:
        """Отношение интенсивности ухода заявок к интенсивности обслуживания.

        Returns:
            float: Значение коэффициента beta = nu_rate / mu_rate.
        """
        return self.params.base_params.nu_rate / self.params.base_params.mu_rate

    @property
    def rho(self) -> float:
        """Коэффициент загрузки системы.

        Returns:
            float: Значение коэффициента загрузки rho = lambda_rate / mu_rate.
        """
        return self.lambda_rate / self.params.base_params.mu_rate

    @abstractmethod
    def calculate_probabilities(self) -> None:
        """Выполняет расчет вектора вероятностей состояний СМО.

        Метод должен быть реализован в подклассах, в соответствии с конкретной
        моделью СМО. Результат расчёта должен сохраняться во внутреннем
        поле `_probabilities`.
        """
        pass

    @property
    def lambda_rate(self) -> float:
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
        return self._probabilities

    def calculate_throughput(self) -> NDArray[np.float64]:
        """Вычисляет общую пропускную способность СМО.

        Returns:
            NDArray[np.float64]: Пропускная способность.
        """
        logger.info("Начат расчёт пропускной способности СМО")
        n = self.probabilities.shape[0]
        throughput = (
            1
            - self.probabilities[(n - self.params.base_params.max_customers) :].sum(
                axis=0
            )
        ) * self.lambda_rate

        logger.success("Расчёт общей пропускной способности завершён успешно")
        return cast(NDArray, throughput)

    def calculate_relative_throughput(self) -> NDArray[np.float64]:
        """Вычисляет относительную пропускную способность СМО.

        Returns:
            NDArray[np.float64]: Относительная пропускная способность.
        """
        logger.info("Начат расчёт относительной пропускной способности для СМО")

        N_b = self.calculate_avg_system_length()

        relative_throughput = (
            1 - (self.params.base_params.nu_rate / self.lambda_rate) * N_b
        )

        logger.success("Расчёт относительной пропускной способности завершён успешно")
        return cast(NDArray, relative_throughput)

    def calculate_absolute_throughput(self) -> NDArray[np.float64]:
        """Вычисляет абсолютную пропускную способность СМО.

        Returns:
            NDArray[np.float64]: Абсолютная пропускная способность.
        """
        logger.info("Начат расчёт абсолютной пропускной способности для СМО")

        N_b = self.calculate_avg_system_length()
        absolute_throughput = self.lambda_rate - self.params.base_params.nu_rate * N_b

        logger.success("Расчёт абсолютной пропускной способности завершён успешно")
        return cast(NDArray, absolute_throughput)

    # TODO:
    def calculate_p_queue(self) -> NDArray[np.float64]:
        """Вычисляет вероятность наличия заявок в очереди.

        Returns:
            NDArray[np.float64]: Вероятность того, что в системе есть хотя бы
                одна заявка.
        """
        logger.info("Вычисление вероятности наличия заявок в очереди")

        p_queue = 1 - self.probabilities[0] * (1 + self.rho)

        logger.success("Вероятность наличия заявок в очереди успешно вычислена")
        return cast(NDArray, p_queue)

    def calculate_avg_buffer_length(self) -> NDArray[np.float64]:
        """Вычисляет среднее число заявок в буфере в стационарном режиме.

        Returns:
            NDArray[np.float64]: Среднее число заявок в буфере.
        """
        logger.info("Вычисление среднего числа заявок в буфере")

        k̄ = self.calculate_avg_busy_channels()

        L = (self.rho + (self.beta - 1) * k̄) / self.beta

        logger.success("Среднее число заявок в буфере вычислено")
        return L

    def calculate_avg_system_length(self) -> NDArray[np.float64]:
        return self.system_behavior.calculate_avg_system_length(self.probabilities)

    # TODO:
    def calculate_avg_busy_channels(self) -> NDArray[np.float64]:
        """Вычисляет среднее число занятых обслуживающих каналов.

        Returns:
            NDArray[np.float64]: Среднее количество занятых каналов (k̄).
        """
        logger.info("Вычисление среднего числа занятых каналов (k̄)")

        k̄ = 1 - self.probabilities[0]

        logger.success("Среднее число занятых каналов успешно вычислено")
        return cast(NDArray, k̄)

    def calculate_service_probability(self) -> NDArray[np.float64]:
        """Вычисляет вероятность того, что заявка будет обслужена.

        Returns:
            NDArray[np.float64]: Вероятность того, что заявка будет обслужена.
        """
        logger.info("Вычисление вероятности обслуживания заявки (Pоб)")

        q = self.calculate_relative_throughput()

        logger.success("Вероятность обслуживания успешно вычислена")
        return cast(NDArray, q)

    def calculate_rejection_probability(self) -> NDArray[np.float64]:
        """Вычисляет вероятность того, что заявка покинет систему необслуженной.

        Returns:
            NDArray[np.float64]: Вероятность отказа в обслуживании.
        """
        logger.info("Вычисление вероятности ухода заявки из очереди (Pух)")

        p_reject = 1 - self.calculate_relative_throughput()

        logger.success("Вероятность ухода из очереди успешно вычислена")
        return p_reject

    def calculate(self) -> dict[str, NDArray[np.float64]]:
        """Универсальный метод вычислений по режиму из CalculationMode.

        Returns:
            dict[str, NDArray[np.float64]]: Результат вычислений.

        Raises:
            ValueError: Если режим не поддерживается.
        """
        result: dict[str, NDArray[np.float64]]

        match self.params.settings.calculation_mode:
            case CalculationMode.PROBABILITY:
                result = {"probability": self.probabilities}
            case CalculationMode.THROUGHPUT:
                result = {"throughput": self.calculate_throughput()}
            case CalculationMode.ABSOLUTE_THROUGHPUT:
                result = {"absolute_throughput": self.calculate_absolute_throughput()}
            case CalculationMode.RELATIVE_THROUGHPUT:
                result = {"relative_throughput": self.calculate_relative_throughput()}
            case CalculationMode.AVG_BUFFER_LENGTH:
                result = {"avg_buffer_length": self.calculate_avg_buffer_length()}
            case CalculationMode.AVG_SYSTEM_LENGTH:
                result = {"avg_system_length": self.calculate_avg_system_length()}
            case CalculationMode.REJECTION_PROBABILITY:
                result = {
                    "rejection_probability": self.calculate_rejection_probability()
                }
            case CalculationMode.SERVICE_PROBABILITY:
                result = {"service_probability": self.calculate_service_probability()}
            case CalculationMode.AIO:
                result = {
                    "probability": self.probabilities,
                    "throughput": self.calculate_throughput(),
                    "absolute_throughput": self.calculate_absolute_throughput(),
                    "relative_throughput": self.calculate_relative_throughput(),
                    "avg_buffer_length": self.calculate_avg_buffer_length(),
                    "avg_system_length": self.calculate_avg_system_length(),
                    "rejection_probability": self.calculate_rejection_probability(),
                    "service_probability": self.calculate_service_probability(),
                }
            case _:
                raise ValueError(
                    "Неподдерживаемый режим расчёта: "
                    f"{self.params.settings.calculation_mode}"
                )

        return result

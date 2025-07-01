"""Модуль базовых моделей пропускной способности системы массового обслуживания (СМО).

Содержит абстрактные базовые классы, определяющие интерфейсы и общую структуру для
вычисления различных метрик пропускной способности СМО с нетерпеливыми заявками,
основываясь на вероятностных моделях состояния системы.
"""

from abc import ABC

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.special import factorial

from app.domain.models import MAPServerParams
from app.domain.models.multi import MultiServerParams
from app.domain.models.single import SingleServerParams
from app.services.systems.base import BaseServerSystem
from app.services.systems.probability import ServerProbabilitySystem


class BaseServerThroughputSystem(BaseServerSystem, ABC):
    """Абстрактный базовый класс для анализа пропускной способности СМО.

    Определяет интерфейс и общую логику для систем массового обслуживания,
    рассчитывающих пропускную способность через вероятностную модель состояния системы.
    """

    def _calculate_probabilities(
        self, params: SingleServerParams | MultiServerParams | MAPServerParams
    ) -> NDArray[np.float64]:
        """Вычисляет вероятности состояний СМО для заданных параметров.

        Args:
            params: Параметры системы (интенсивности, размеры очереди, др.).

        Returns:
            NDArray[np.float64]: Массив вероятностей состояний системы.
                Последний элемент соответствует вероятности потери заявки.
        """
        queue_system = ServerProbabilitySystem(params, self.config)
        probabilities = queue_system.calculate()
        return probabilities


class BaseServerExpectedThroughputSystem(BaseServerThroughputSystem, ABC):
    """Абстрактный базовый класс для расширенного анализа пропускной способности СМО.

    Добавляет метод для вычисления ожидаемого значения числа заявок, ушедших из системы
    из-за нетерпения, на основе параметров системы и рассчитанных вероятностей
    состояний.
    """

    def _calculate_expected_value(
        self, p_loss: NDArray[np.float64], lambda_rate: float | None = None
    ) -> NDArray[np.float64]:
        """Вычисляет ожидаемое число ушедших заявок.

        Args:
            p_loss (NDArray[np.float64]): Вероятность потери заявки.
            lambda_rate (float | None): Интенсивность поступления заявок (λ).
                Если None, то берет значение из self.params. По умолчанию None.

        Returns:
            NDArray[np.float64]: Ожидаемое значение числа ушедших заявок.
        """
        logger.debug("Начинаем вычисление ожидаемого значения ушедших заявок")

        if lambda_rate is None and isinstance(self.params.lambda_rate, float):
            lambda_rate = self.params.lambda_rate
        else:
            logger.error("Интенсивность λ не передана и не определена в параметрах.")
            raise ValueError(
                "Интенсивность λ не передана и не определена в параметрах."
            )

        β = self.params.nu_rate / self.params.mu_rate
        ρ = lambda_rate / self.params.mu_rate

        n = self.params.max_customers
        k = n
        if isinstance(self.params, MultiServerParams):
            k = self.params.processor_count + n

        sum_term = 0.0
        for k_i in range(1, k + 1):
            numerator = k_i * ρ**k_i
            denominator = np.prod([n + j * β for j in range(1, k_i + 1)])
            sum_term += float(numerator / denominator)

        prefactor = (ρ**n / factorial(n)) * p_loss
        N_b = prefactor * sum_term

        logger.success(
            "Вычисление ожидаемого значения ушедших заявок завершено успешно"
        )
        return N_b

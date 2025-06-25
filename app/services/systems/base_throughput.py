"""Модуль базовых моделей пропускной способности системы массового обслуживания (СМО).

Содержит абстрактные базовые классы, определяющие интерфейсы и общую структуру
для вычисления различных метрик пропускной способности СМО с нетерпеливыми заявками,
основываясь на вероятностных моделях состояния системы.
"""

import warnings
from abc import ABC

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.special import factorial, gammaln  # pylint: disable=no-name-in-module

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

    Добавляет метод для вычисления ожидаемого значения числа заявок,
    ушедших из системы из-за нетерпения, на основе параметров системы и
    рассчитанных вероятностей состояний.
    """

    def _calculate_expected_value_core(
        self, lambda_rate: float, p_loss: NDArray[np.float64], max_k: int = 1000
    ) -> NDArray[np.float64]:
        """Ядро вычисления ожидаемого числа ушедших заявок.

        Args:
            lambda_rate (float): Интенсивность поступления заявок (λ).
            p_loss (NDArray[np.float64]): Вероятность потери заявки.
            max_k (int): Максимальное число итераций для суммы.

        Returns:
            NDArray[np.float64]: Ожидаемое значение числа ушедших заявок.
        """
        n = self.params.max_customers

        β = self.params.nu_rate / self.params.mu_rate
        ρ = lambda_rate / self.params.mu_rate

        k = np.arange(1, max_k + 1)

        log_numerator = np.log(k) + k * np.log(ρ)
        log_denominator = gammaln(n + (k + 1) * β) - gammaln(n + β)
        log_terms = log_numerator - log_denominator

        max_log_term = np.max(log_terms)
        sum_term = np.sum(np.exp(log_terms - max_log_term))
        sum_term *= np.exp(max_log_term)

        prefactor = (ρ**n / factorial(n)) * p_loss
        N_b = prefactor * float(sum_term)

        logger.debug("Завершено вычисление ядра ожидаемого значения")
        return N_b

    def _calculate_expected_value(
        self, lambda_rate: float, p_loss: np.ndarray, max_k: int = 1000
    ) -> np.ndarray:
        """Вычисляет ожидаемое число ушедших заявок с защитой от переполнения.

        Выполняет вычисления с постепенным уменьшением max_k при возникновении
        переполнения или других предупреждений, связанных с вычислениями.

        Args:
            lambda_rate (float): Интенсивность поступления заявок (λ).
            p_loss (NDArray[np.float64]): Вероятность потери заявки.
            max_k (int): Максимальное число итераций для суммы.

        Raises:
            RuntimeWarning: Если вычислить значение не удалось при минимальном max_k.

        Returns:
            NDArray[np.float64]: Ожидаемое значение числа ушедших заявок.
        """
        min_k = 10
        if max_k < min_k:
            logger.error(
                "Не удалось вычислить значение без overflow при минимальном "
                "max_k={min_k}"
            )
            raise RuntimeWarning(
                "Не удалось вычислить значение без overflow с допустимым max_k."
            )

        logger.info(f"Вычисление ожидаемого значения ушедших заявок с max_k={max_k}")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = self._calculate_expected_value_core(lambda_rate, p_loss, max_k)

            warning_overflow = any(
                issubclass(warning.category, RuntimeWarning)
                and (
                    "overflow" in str(warning.message)
                    or "invalid value" in str(warning.message)
                )
                for warning in w
            )
            if warning_overflow:
                logger.info(
                    "Обнаружено предупреждение overflow/invalid value при "
                    f"max_k={max_k}, уменьшаем max_k и повторяем"
                )
                return self._calculate_expected_value(
                    lambda_rate, p_loss, max_k=max_k // 2
                )

            logger.info(
                f"Успешно вычислено ожидаемое значение ушедших заявок при max_k={max_k}"
            )
            return result

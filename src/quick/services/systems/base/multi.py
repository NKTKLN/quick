"""Модуль, содержащий базовый класс для многоканальных систем массового обслуживания (СМО).

Включает в себя определение класса BaseMultiServerSystem с методами для расчёта
пропускной способности, средней длины очереди, средней длины системы,
среднего числа занятых каналов и других характеристик многоканальной СМО.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from .base import BaseServerSystem


class BaseMultiServerSystem(BaseServerSystem):
    """Абстрактный базовый класс многоканальной системы массового обслуживания (СМО).

    Расширяет базовый класс BaseServerSystem методами для вычисления
    характеристик многоканальной СМО, включая абсолютную и относительную
    пропускную способность, среднее число заявок в системе и буфере,
    а также вероятность обслуживания и отказа.
    """

    def calculate_throughput(self) -> NDArray[np.float64]:
        """Вычисляет пропускную способность A(t).

        Returns:
            NDArray[np.float64]: Пропускная способность системы.
        """
        logger.info("Вычисление пропускной способности A(t)")

        n = self.probabilities.shape[0]

        throughput = (
            1
            - self.probabilities[(n - self.params.base_params.max_customers) :].sum(
                axis=0
            )
        ) * self.lambda_rate

        logger.success("Пропускная способность успешно вычислена")
        return cast(NDArray, throughput)

    def calculate_relative_throughput(self) -> NDArray[np.float64]:
        """Вычисляет относительную пропускную способность P_s(t).

        Returns:
            NDArray[np.float64]: Относительная пропускная способность.
        """
        logger.info("Вычисление относительной пропускной способности P_s(t)")

        N_b = self.calculate_avg_system_length()  # TODO

        relative_throughput = (
            1 - (self.params.base_params.nu_rate / self.lambda_rate) * N_b
        )

        logger.success("Относительная пропускная способность успешно вычислена")
        return cast(NDArray, relative_throughput)

    def calculate_absolute_throughput(self) -> NDArray[np.float64]:
        """Вычисляет абсолютную пропускную способность A_s(t).

        Returns:
            NDArray[np.float64]: Абсолютная пропускная способность.
        """
        logger.info("Вычисление абсолютной пропускной способности A_s(t)")

        N_b = self.calculate_avg_system_length()  # TODO

        absolute_throughput = self.lambda_rate - self.params.base_params.nu_rate * N_b

        logger.success("Абсолютная пропускная способность успешно вычислена")
        return cast(NDArray, absolute_throughput)

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
        """Вычисляет среднее число заявок в буфере N_b(t).

        Returns:
            NDArray[np.float64]: Среднее число заявок в буфере.
        """
        logger.info("Вычисление среднего числа заявок в буфере N_b(t)")

        k̄ = self.calculate_avg_busy_channels()

        N_b = (self.rho + (self.beta - 1) * k̄) / self.beta

        logger.success("Среднее число заявок в буфере вычислено")
        return N_b

    def calculate_avg_system_length(self) -> NDArray[np.float64]:
        """Вычисляет среднее число заявок в системе.

        Returns:
            NDArray[np.float64]: Среднее число заявок в системе.
        """
        return self.system_behavior.calculate_avg_system_length(self.probabilities)

    def calculate_avg_busy_channels(self) -> NDArray[np.float64]:
        """Вычисляет среднее число занятых обслуживающих каналов.

        Returns:
            NDArray[np.float64]: Среднее количество занятых каналов (k̄).
        """
        logger.info("Вычисление среднего числа занятых каналов (k̄)")

        k̄ = 1 - self.probabilities[0]

        logger.success("Среднее число занятых каналов успешно вычислено")
        return cast(NDArray, k̄)

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
        """Универсальный метод вычислений.

        Returns:
            dict[str, NDArray[np.float64]]: Результат вычислений.
        """
        calculations: dict[str, Callable] = {
            "probability": lambda: self.probabilities,
            "throughput": self.calculate_throughput,
            "absolute_throughput": self.calculate_absolute_throughput,
            "relative_throughput": self.calculate_relative_throughput,
            "avg_buffer_length": self.calculate_avg_buffer_length,
            "avg_system_length": self.calculate_avg_system_length,
            "rejection_probability": self.calculate_rejection_probability,
        }

        logger.info("Запуск расчёта характеристик СМО")

        result = {
            metric_name: calculation()
            for metric_name, calculation in calculations.items()
        }

        logger.success("Расчёт успешно завершён")
        return result

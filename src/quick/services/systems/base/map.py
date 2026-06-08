"""Модуль, содержащий базовый класс для серверных систем массового обслуживания (СМО) с Марковскими входными потоками.

Включает в себя определение класса BaseMAPServerSystem с методами для преобразования
вероятностей состояний и расчёта основных характеристик СМО с Марковскими входными потоками:
средней длины буфера, вероятностей отказа, ухода, потерь, обслуживания
и пропускной способности.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain.models.system_params import MAPSystemParams

from .base import BaseServerSystem


class BaseMAPServerSystem(BaseServerSystem):
    """Абстрактный базовый класс серверной СМО с Марковскими входными потоками.

    Расширяет базовый класс BaseServerSystem методами, специфичными для систем
    массового обслуживания с Марковскими входными потоками, включая преобразование вероятностей
    состояний по фазам MAP и расчёт производных характеристик системы.
    """

    def _reshape_map_probabilities(self) -> NDArray[np.float64]:
        """Преобразует плоский вектор вероятностей в матрицу [macro_state, map_phase].

        Returns:
            NDArray[np.float64]: Массив формы [num_macro_states, M].

        Raises:
            TypeError: Если параметры системы не являются MAPSystemParams.
            ValueError: Если размер вектора вероятностей не кратен числу фаз MAP.
        """
        if not isinstance(self.params.base_params, MAPSystemParams):
            logger.error("Базовые параметры не являются экземпляром MAPSystemParams")
            raise TypeError("Базовые параметры должны быть экземпляром MAPSystemParams")

        m = self.params.base_params.sensor_count
        probs = self.probabilities

        if probs.ndim == 1:
            probs = probs[:, np.newaxis]

        if probs.shape[0] % m != 0:
            raise ValueError(
                "Размер вектора вероятностей не кратен числу фаз MAP: "
                f"{probs.shape[0]} % {m} != 0"
            )

        if probs.shape[1] == 1:
            return cast(
                NDArray[np.float64],
                probs.reshape(probs.shape[0] // m, m, -1).sum(axis=2),
            )

        return cast(
            NDArray[np.float64],
            probs.reshape(probs.shape[0] // m, m, probs.shape[1]),
        )

    def calculate_avg_system_length(self) -> NDArray[np.float64]:
        """Вычисляет среднее число заявок в системе.

        Returns:
            NDArray[np.float64]: Среднее число заявок в системе.
        """
        return self.system_behavior.calculate_avg_system_length(self.probabilities)

    def calculate_rejection_probability(self) -> NDArray[np.float64]:
        """Вычисляет вероятность отказа P_fail(t).

        Returns:
            NDArray[np.float64]: Вероятность отказа.
        """
        logger.info("Вычисление вероятности отказа P_fail(t)")

        probs = self._reshape_map_probabilities()
        p_reject = cast(NDArray[np.float64], np.sum(probs[-1], axis=0))

        logger.success("Вероятность отказа успешно вычислена")
        return p_reject

    def calculate_quit_probability(self) -> NDArray[np.float64]:
        """Вычисляет вероятность ухода P_uns(t).

        Returns:
            NDArray[np.float64]: Вероятность ухода заявки из системы.
        """
        logger.info("Вычисление вероятности ухода P_uns(t)")

        p_quit = (
            self.params.base_params.nu_rate
            / self.lambda_rate
            * self.calculate_avg_system_length()
        )

        logger.success("Вероятность ухода успешно вычислена")
        return p_quit

    def calculate_loss_probability(self) -> NDArray[np.float64]:
        """Вычисляет вероятность потерь P_loss(t).

        Returns:
            NDArray[np.float64]: Вероятность потерь.
        """
        logger.info("Вычисление вероятности потерь P_loss(t)")

        rejection_probability = self.calculate_rejection_probability()
        quit_probability = self.calculate_quit_probability()

        if quit_probability.ndim == 2 and rejection_probability.ndim == 1:
            rejection_probability = rejection_probability[:, np.newaxis]

        p_loss = rejection_probability + quit_probability

        logger.success("Вероятность потерь успешно вычислена")
        return p_loss

    def calculate_service_probability(self) -> NDArray[np.float64]:
        """Вычисляет вероятность обслуживания P_s(t).

        Returns:
            NDArray[np.float64]: Вероятность обслуживания.
        """
        logger.info("Вычисление вероятности обслуживания P_s(t)")

        p_s = 1.0 - self.calculate_loss_probability()

        logger.success("Вероятность обслуживания успешно вычислена")
        return p_s

    def calculate_throughput(self) -> NDArray[np.float64]:
        """Вычисляет пропускную способность A(t).

        Returns:
            NDArray[np.float64]: Пропускная способность системы.
        """
        logger.info("Вычисление пропускной способности A(t)")

        throughput = (1.0 - self.calculate_loss_probability()) * self.lambda_rate

        logger.success("Пропускная способность успешно вычислена")
        return throughput

    def calculate(self) -> dict[str, NDArray[np.float64]]:
        """Универсальный метод вычислений.

        Returns:
            dict[str, NDArray[np.float64]]: Результат вычислений.
        """
        calculations: dict[str, Callable] = {
            "probability": lambda: self.probabilities,
            "throughput": self.calculate_throughput,
            "avg_system_length": self.calculate_avg_system_length,
            "rejection_probability": self.calculate_rejection_probability,
            "loss_probability": self.calculate_loss_probability,
            "service_probability": self.calculate_service_probability,
            "quit_probability": self.calculate_quit_probability,
        }

        logger.info("Запуск расчёта характеристик СМО")

        result = {
            metric_name: calculation()
            for metric_name, calculation in calculations.items()
        }

        logger.success("Расчёт успешно завершён")
        return result

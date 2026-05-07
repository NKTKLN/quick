"""Содержит реализацию поведения многосенсорной системы массового обслуживания с Марковскими входными потоками.

Модуль определяет класс MultiSensorMAPSystemBehavior, который расширяет базовый
интерфейс поведения СМО для случая многосенсорной системы с входным потоком
типа MAP. Класс отвечает за вычисление интенсивности поступления заявок и
среднего числа заявок в системе на основе матриц переходов MAP-процесса и
вектора вероятностей состояний.
"""

from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain.models.system_params import MAPSystemParams
from quick.services.systems_behavior.map import MultiSensorMAPSystemBehavior


class MultiSensorMAPSensorBehavior(MultiSensorMAPSystemBehavior):
    """Класс поведения многосенсорной системы массового обслуживания с Марковскими входными потоками (по каждому сенсору).

    Класс реализует вычисление характеристик СМО для моделей, в которых
    входной поток задаётся марковским процессом поступления заявок (MAP),
    а состояние системы дополнительно зависит от числа сенсоров.
    Для вычислений используются матрицы D0 и D1, формируемые построителем
    MultiSensorMAPServerMatrixBuilder.

    Attributes:
        D0 (np.ndarray): Матрица переходов без поступления заявки.
        D1 (np.ndarray): Матрица переходов с поступлением заявки.
        DDD (np.ndarray): Суммарная матрица переходов MAP-процесса.
    """

    def calculate_avg_system_length(
        self, probabilities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Вычисляет среднее число заявок в буфере для каждого датчика в каждый момент времени.

        Returns:
            NDArray[np.float64]:
                Массив формы [time_count, sensor_count], где result[t, i] -
                среднее число заявок в буфере i-го датчика в момент времени t.

        Raises:
            TypeError: Если параметры системы не являются MAPSystemParams.
        """
        logger.info("Вычисление среднего числа заявок в буфере для каждого датчика")

        if not isinstance(self.params.base_params, MAPSystemParams):
            logger.error("Базовые параметры не являются экземпляром MAPSystemParams")
            raise TypeError("Базовые параметры должны быть экземпляром MAPSystemParams")

        n = self.params.base_params.max_customers
        m = self.params.base_params.sensor_count

        buffer_sizes = np.maximum(np.arange(n) - 1, 0)

        avg_buffer_by_sensor = np.sum(
            probabilities.reshape(
                probabilities.shape[0] // m, m, probabilities.shape[1]
            )
            * buffer_sizes[:, np.newaxis, np.newaxis],
            axis=0,
        ).T

        logger.success(
            "Среднее число заявок в буфере для каждого датчика успешно вычислено"
        )

        return cast(NDArray[np.float64], avg_buffer_by_sensor)

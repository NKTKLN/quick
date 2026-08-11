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

    def calculate_unweighted_buffer_occupancy(
        self, probabilities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""Вычисляет невзвешенное заполнение буфера по каждому датчику.

        Для расчёта ``P_uns(t)`` используется выражение

        .. math::

            N_{b,i}^{(uns)}(t)=\sum_{k=1}^{N}P(k+1,i,t),

        то есть без домножения вероятностей уровней на ``k``.

        Returns:
            NDArray[np.float64]: Массив формы ``[time_count, sensor_count]``.

        Raises:
            TypeError: Если параметры системы не являются MAPSystemParams.
        """
        logger.info("Вычисление невзвешенного N_b(t) по датчикам для P_uns(t)")

        if not isinstance(self.params.base_params, MAPSystemParams):
            logger.error("Базовые параметры не являются экземпляром MAPSystemParams")
            raise TypeError("Базовые параметры должны быть экземпляром MAPSystemParams")

        phase_count = self.params.base_params.sensor_count
        probabilities_by_level = probabilities.reshape(
            probabilities.shape[0] // phase_count,
            phase_count,
            probabilities.shape[1],
        )
        buffer_probabilities = probabilities_by_level[2:]
        unweighted_by_sensor = np.sum(buffer_probabilities, axis=0).T

        logger.success("Невзвешенное N_b(t) по датчикам для P_uns(t) успешно вычислено")
        return cast(NDArray[np.float64], unweighted_by_sensor)

    def calculate_avg_system_length(
        self, probabilities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""Вычисляет ``N_b(t)`` отдельно для каждой MAP-фазы (датчика).

        Для каждого датчика применяется выражение

        .. math::

            N_{b,i}(t)=\sum_{k=1}^{N}kP(k+1,i,t).

        Returns:
            NDArray[np.float64]:
                Массив формы ``[time_count, sensor_count]``, где
                ``result[t, i]`` — вклад i-го датчика в ``N_b(t)``.

        Raises:
            TypeError: Если параметры системы не являются MAPSystemParams.
        """
        logger.info("Вычисление N_b(t) по датчикам с множителем k")

        if not isinstance(self.params.base_params, MAPSystemParams):
            logger.error("Базовые параметры не являются экземпляром MAPSystemParams")
            raise TypeError("Базовые параметры должны быть экземпляром MAPSystemParams")

        phase_count = self.params.base_params.sensor_count
        probabilities_by_level = probabilities.reshape(
            probabilities.shape[0] // phase_count,
            phase_count,
            probabilities.shape[1],
        )
        buffer_probabilities = probabilities_by_level[2:]
        buffer_weights = np.arange(
            1,
            buffer_probabilities.shape[0] + 1,
            dtype=np.float64,
        )[:, np.newaxis, np.newaxis]

        avg_buffer_by_sensor = np.sum(
            buffer_weights * buffer_probabilities,
            axis=0,
        ).T

        logger.success("N_b(t) по датчикам с множителем k успешно вычислено")

        return cast(NDArray[np.float64], avg_buffer_by_sensor)

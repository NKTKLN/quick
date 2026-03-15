"""Содержит реализацию поведения системы массового обслуживания с MAP-потоком.

Модуль определяет класс MAPSystemBehavior, который расширяет базовый интерфейс
поведения СМО для случая входного потока типа MAP. Класс отвечает за вычисление
интенсивности поступления заявок и среднего числа заявок в системе на основе
матриц переходов MAP-процесса и вектора вероятностей состояний.

Основная сущность:
    MAPSystemBehavior — класс поведения СМО с MAP-потоком, использующий
    матричное представление процесса поступления заявок.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from app.domain.models.base import SystemParams
from app.services.matrix_builders.map import MAPServerMatrixBuilder
from app.services.systems_behavior.base import BaseSystemBehavior


class MAPSystemBehavior(BaseSystemBehavior):
    """Класс поведения системы массового обслуживания с MAP-потоком.

    Класс реализует вычисление характеристик СМО для моделей, в которых
    входной поток задаётся марковским процессом поступления заявок (MAP).
    Для этого используются матрицы D0 и D1, формируемые построителем
    MAPServerMatrixBuilder.

    Attributes:
        transition_matrix (MAPServerMatrixBuilder): Построитель матриц MAP-процесса.
        D0 (np.ndarray): Матрица переходов без поступления заявки.
        D1 (np.ndarray): Матрица переходов с поступлением заявки.
        DDD (np.ndarray): Суммарная матрица переходов MAP-процесса.
    """

    def __init__(self, params: SystemParams) -> None:
        """Инициализирует систему массового обслуживания с заданными параметрами.

        Args:
            params (ServerParams): Параметры СМО.
        """
        super().__init__(params)
        self.transition_matrix = MAPServerMatrixBuilder(self.params.base_params)
        self.D0 = self.transition_matrix._d_0_matrix_generator()
        self.D1 = self.transition_matrix._d_1_matrix_generator()
        self.DDD = self.D0 + self.D1

    def _calculate_teta(self) -> None:
        """Вычисляет стационарный вектор вероятностей состояний MAP-процесса.

        Метод формирует систему линейных уравнений на основе матрицы переходов
        MAP-процесса и сохраняет результат во внутреннем атрибуте `teta_vec`.

        Raises:
            ValueError: Если детерминант базовой матрицы равен нулю и решение
            системы невозможно.
        """
        n = self.DDD.shape[0]

        base_matrix = np.vstack([self.DDD.T[: n - 1], np.ones(n)])
        det_base = np.linalg.det(base_matrix)

        if np.isclose(det_base, 0):
            raise ValueError(
                "Детерминант базовой матрицы равен нулю, решение невозможно."
            )

        self.teta_vec = np.zeros(n, dtype=np.float64)
        for i in range(n):
            modified = base_matrix.copy()
            modified[: n - 1, i] = 0
            self.teta_vec[i] = np.linalg.det(modified) / det_base

    @property
    def lambda_rate(self) -> float:
        """Возвращает интенсивность поступления заявок.

        Raises:
            ValueError: Если параметры имеют тип MAPSystemParams.
        """
        self._calculate_teta()
        ones = np.ones((self.D1.shape[0], 1))
        lam_vector = self.teta_vec @ self.D1
        lam = lam_vector @ ones
        return lam.item()

    def calculate_avg_system_length(
        self, probabilities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Вычисляет среднее число заявок в системе.

        Returns:
            NDArray[float64]: Среднее число заявок в системе.
        """
        logger.info("Вычисление среднего числа заявок в системе")

        n = self.params.base_params.max_customers

        N_b = np.sum(
            np.sum(
                probabilities[n * 2 :].reshape(n - 2, n, probabilities.shape[-1]),
                axis=1,
            )
            * np.arange(1, n - 1)[:, np.newaxis],
            axis=0,
        )

        logger.success("Среднее число заявок в системе вычислено")
        return N_b

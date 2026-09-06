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

from quick.domain.params import MAPSystemParams
from quick.domain.params.system import SystemParams
from quick.services.matrix_builders.map import MultiSensorMAPServerMatrixBuilder
from quick.services.systems_behavior.base import BaseSystemBehavior


class MultiSensorMAPSystemBehavior(BaseSystemBehavior):
    """Класс поведения многосенсорной системы массового обслуживания с Марковскими входными потоками.

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

    def __init__(self, params: SystemParams) -> None:
        """Инициализирует систему массового обслуживания с заданными параметрами.

        Args:
            params (ServerParams): Параметры СМО.

        Raises:
            TypeError: Если базовые параметры системы не являются MAPSystemParams.
        """
        super().__init__(params)

        self.base_params = self.params.base_params

        if not isinstance(self.base_params, MAPSystemParams):
            logger.error("Базовые параметры не являются экземпляром MAPSystemParams")
            raise TypeError("Базовые параметры должны быть экземпляром MAPSystemParams")

        self.transition_matrix = MultiSensorMAPServerMatrixBuilder(
            cast(MAPSystemParams, self.params.base_params)
        )
        self.D0 = self.transition_matrix._d_0_matrix_generator()
        self.D1 = self.transition_matrix._d_1_matrix_generator()
        self.DDD = self.D0 + self.D1

    @property
    def sensor_index(self) -> int | None:
        """Номер датчика, по состояниям которого ведётся расчёт.

        Returns:
            int | None: Номер MAP-фазы в индивидуальном режиме или ``None``
                в агрегированном.
        """
        return self.params.calculation_params.sensor_index

    def select_phases(
        self, probabilities_by_level: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""Оставляет в массиве вероятностей только состояния выбранного датчика.

        В агрегированном режиме массив возвращается без изменений и
        последующее суммирование по оси фаз даёт
        :math:`\sum_{i=0}^{M-1}P(k,i,t)`. В индивидуальном режиме остаётся
        единственный столбец, поэтому то же суммирование вырождается в
        :math:`P(k,j,t)` — формулы характеристик при этом не меняются.

        Args:
            probabilities_by_level (NDArray[np.float64]): Вероятности формы
                ``[level, phase, time]``.

        Returns:
            NDArray[np.float64]: Вероятности формы ``[level, phase, time]``,
                где ось фаз содержит либо все датчики, либо один выбранный.
        """
        if self.sensor_index is None:
            return probabilities_by_level

        return probabilities_by_level[:, self.sensor_index : self.sensor_index + 1, :]

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

    def calculate_unweighted_buffer_occupancy(
        self, probabilities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""Возвращает заполнение буфера для обратной совместимости API.

        Исторически этот метод действительно возвращал невзвешенную сумму
        вероятностей уровней. В актуальной математической модели формула (14)
        требует множитель ``k``. Поэтому теперь метод делегирует вычисление
        корректному :meth:`calculate_avg_system_length` и возвращает

        .. math::

            N_b(t)=\sum_{k=1}^{N}k\sum_{i=0}^{M-1}P(k+1,i,t).

        Новому коду следует использовать ``calculate_avg_system_length``
        напрямую; имя сохранено, чтобы не ломать внешние вызовы.

        Returns:
            NDArray[np.float64]: Среднее число заявок в буфере.
        """
        logger.warning(
            "calculate_unweighted_buffer_occupancy устарел: "
            "используется взвешенное N_b(t) согласно формуле (14)."
        )
        return self.calculate_avg_system_length(probabilities)

    def calculate_avg_system_length(
        self, probabilities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""Вычисляет среднее число заявок в буфере ``N_b(t)``.

        Используется взвешенная сумма вероятностей состояний:

        .. math::

            N_b(t)=\sum_{k=1}^{N}k\sum_{i=0}^{M-1}P(k+1,i,t).

        Returns:
            NDArray[float64]: Среднее число заявок в буфере во времени.

        Raises:
            TypeError: Если параметры системы не являются MAPSystemParams.
        """
        logger.info("Вычисление N_b(t) с множителем k")

        if not isinstance(self.params.base_params, MAPSystemParams):
            logger.error("Базовые параметры не являются экземпляром MAPSystemParams")
            raise TypeError("Базовые параметры должны быть экземпляром MAPSystemParams")

        level_count = self.params.base_params.max_customers
        phase_count = self.params.base_params.sensor_count
        time_count = probabilities.shape[-1]

        # Уровни 0 и 1 соответствуют пустой системе и одной заявке на
        # обслуживании. Уровень k + 1, k = 1, ..., N, означает наличие k
        # заявок в буфере, поэтому вероятность этого уровня умножается на k.
        buffer_probabilities = self.select_phases(
            probabilities[phase_count * 2 :].reshape(
                level_count - 2,
                phase_count,
                time_count,
            )
        )
        buffer_weights = np.arange(
            1,
            level_count - 1,
            dtype=np.float64,
        )[:, np.newaxis, np.newaxis]

        N_b = cast(
            NDArray[np.float64],
            np.sum(buffer_weights * buffer_probabilities, axis=(0, 1)),
        )

        logger.success("N_b(t) с множителем k успешно вычислено")
        return N_b

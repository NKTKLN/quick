"""Модуль параметров систем массового обслуживания с Марковскими входными потоками.

Реализует класс MAPSystemParams, который описывает параметры системы массового
обслуживания с марковским модулированным пуассоновским входным потоком (MAP) и
возможностью ухода заявок.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import BaseSystemParams


@dataclass
class MAPSystemParams(BaseSystemParams):
    """Базовые параметры СМО с Марковскими входными потоками и уходом нетерпеливых заявок.

    Описывает систему с марковским модулированным пуассоновским входным потоком,
    а также оттоком нетерпеливых заявок. Включает матрицы интенсивностей переходов.

    Attributes:
        lambda_rate (NDArray[np.float64]): Интенсивность поступления заявок (λ > 0).
        p_rate (NDArray[np.float64]): Матрица вероятностей переходов
            ``p^(0)`` без генерации заявки.
        q_rate (NDArray[np.float64]): Матрица вероятностей переходов
            ``p^(1)`` с генерацией заявки.
        sensor_count (int): Количество фаз MAP-процесса / датчиков (M).
        Остальные параметры наследуются от BaseSystemParams.
    """

    lambda_rate: NDArray[np.float64]
    p_rate: NDArray[np.float64]
    q_rate: NDArray[np.float64]
    sensor_count: int

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Raises:
            ValueError: При некорректных параметрах.
        """
        super().validate()
        self._validate_phases()
        self._validate_matrix_shapes()
        self._validate_transition_probabilities()

        # В статье задано ν < μ. Для построения предельного сечения ν/μ = 1
        # приложение допускает равенство, но значения ν > μ считаются вне
        # области применимости используемой модели.
        if self.nu_rate > self.mu_rate:
            raise ValueError(
                "Для MAP/M/1/N-модели требуется ν <= μ "
                "(в статье используется строгое условие ν < μ)."
            )

    def _validate_phases(self) -> None:
        """Проверяет число MAP-фаз и согласованный с ним вектор λ.

        Raises:
            ValueError: При некорректных параметрах.
        """
        if self.sensor_count <= 0:
            raise ValueError("Количество MAP-фаз (датчиков) должно быть положительным.")

        if self.lambda_rate.ndim != 1:
            raise ValueError("Массив интенсивностей λ_i должен быть одномерным.")

        if self.lambda_rate.shape[0] != self.sensor_count:
            raise ValueError(
                "Количество интенсивностей λ_i должно совпадать с числом "
                f"MAP-фаз: {self.lambda_rate.shape[0]} != {self.sensor_count}."
            )

    def _validate_matrix_shapes(self) -> None:
        """Проверяет размерности матриц переходов p^(0) и p^(1).

        Raises:
            ValueError: При некорректных параметрах.
        """
        if self.p_rate.shape != self.q_rate.shape:
            raise ValueError(
                "Матрицы p^(0) и p^(1) должны иметь одинаковую размерность."
            )
        if self.p_rate.ndim != 2 or self.p_rate.shape[0] != self.p_rate.shape[1]:
            raise ValueError(
                "Матрицы p^(0) и p^(1) должны быть двумерными и квадратными."
            )
        if self.p_rate.shape != (self.sensor_count, self.sensor_count):
            raise ValueError(
                "Размер матриц p^(0), p^(1) должен совпадать с числом MAP-фаз: "
                f"ожидалось {(self.sensor_count, self.sensor_count)}, "
                f"получено {self.p_rate.shape}."
            )

    def _validate_transition_probabilities(self) -> None:
        """Проверяет значения матриц переходов и условие нормировки.

        Raises:
            ValueError: При некорректных параметрах.
        """
        if np.any((self.p_rate < 0) | (self.p_rate > 1)):
            raise ValueError(
                "Вероятности переходов без генерации заявки p^(0) должны "
                "находиться в диапазоне [0, 1]."
            )
        if np.any((self.q_rate < 0) | (self.q_rate > 1)):
            raise ValueError(
                "Вероятности переходов с генерацией заявки p^(1) должны "
                "находиться в диапазоне [0, 1]."
            )
        if np.sum(self.q_rate) == 0:
            raise ValueError(
                "Матрица переходов с генерацией заявки p^(1) не может быть нулевой."
            )

        if not np.allclose(np.diag(self.p_rate), 0.0, atol=1e-12, rtol=0.0):
            raise ValueError(
                "Для MAP-процесса переход в ту же фазу без генерации заявки "
                "невозможен: диагональ p^(0) должна быть нулевой."
            )

        row_sums = np.sum(self.p_rate + self.q_rate, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-10, rtol=0.0):
            raise ValueError(
                "Для каждой MAP-фазы должно выполняться условие нормировки "
                "sum p^(0) + sum p^(1) = 1. Получены суммы: "
                f"{np.array2string(row_sums, precision=12)}."
            )

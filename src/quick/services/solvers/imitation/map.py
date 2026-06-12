"""Имитационный решатель для MAP-СМО с уходами заявок.

Содержит класс `MAPImitationProbabilitySolver`, реализующий Монте-Карло
моделирование MAP-системы с времезависимыми (LOCF) интенсивностями.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain.params import ImitationSystemParams, MAPSystemParams
from quick.services.matrix_builders import MultiSensorMAPServerMatrixBuilder
from quick.services.solvers.imitation.base import BaseImitationProbabilitySolver
from quick.services.solvers.imitation.utils import TimeSeriesMAPRateProvider


class MAPImitationProbabilitySolver(BaseImitationProbabilitySolver):
    """Имитационный (Монте-Карло) решатель для MAP-СМО с LOCF-параметрами."""

    def __init__(self, system_params: ImitationSystemParams) -> None:
        """Создаёт решатель Монте-Карло для MAP-СМО.

        Args:
            system_params (ImitationSystemParams): Объединённые параметры системы.

        Raises:
            TypeError: Если параметры системы не являются MAPSystemParams.
        """
        super().__init__(system_params)

        if not isinstance(self.base_params, MAPSystemParams):
            logger.error("Параметры не являются экземпляром MAPSystemParams")
            raise TypeError("Параметры должны быть экземпляром MAPSystemParams")

        self.rate_provider = TimeSeriesMAPRateProvider(
            self.base_params,
            self.system_params.time_series_params,
        )

        self.k_max = self.base_params.max_customers
        self.m = self.base_params.lambda_rate.shape[0]
        self.num_levels = self.k_max

    @property
    def num_states(self) -> int:
        """Количество состояний системы."""
        if self._num_states is None:
            self._num_states = self.num_levels * self.m
        return self._num_states

    def _get_q_row(self, time_point: float) -> NDArray[np.float64]:
        """Возвращает генератор Q(t) для текущего времени.

        Args:
            time_point (float): Текущий момент времени.

        Returns:
            NDArray[np.float64]: Матрица коэффициентов для СМО.

        Raises:
            TypeError: Если параметры системы не являются MAPSystemParams.
        """
        if not isinstance(self.base_params, MAPSystemParams):
            logger.error("Параметры не являются экземпляром MAPSystemParams")
            raise TypeError("Параметры должны быть экземпляром MAPSystemParams")

        lambda_t = self.rate_provider.get_lambda_rate(time_point)
        mu_t = self.rate_provider.get_mu_rate(time_point)
        nu_t = self.rate_provider.get_nu_rate(time_point)

        matrix_builder = MultiSensorMAPServerMatrixBuilder(
            params=MAPSystemParams(
                mu_t,
                nu_t,
                lambda_t,
                self.base_params.max_customers,
                self.base_params.p_rate,
                self.base_params.q_rate,
                self.base_params.sensor_count,
            ),
            debug_logs=True,
        )

        return matrix_builder.build().T

    def _get_transitions(
        self,
        state_id: int,
        time_point: float,
    ) -> list[tuple[float, int]]:
        """Возвращает возможные переходы из заданного состояния в момент времени.

        Args:
            state_id (int): Текущий идентификатор состояния системы.
            time_point (float): Текущий момент времени.

        Returns:
            list[tuple[float, int]]: Список возможных переходов, где каждый элемент -
                это кортеж вида (rate, next_state):
                    - rate (float): интенсивность перехода,
                    - next_state (int): состояние, в которое осуществляется переход.
        """
        logger.debug(f"Вычисление переходов: {state_id=}, {time_point=:.4f}")

        q_row = self._get_q_row(time_point)
        transitions: list[tuple[float, int]] = []

        for next_state in range(self.num_states):
            if next_state == state_id:
                continue

            rate = float(q_row[state_id, next_state])

            if rate > 0:
                transitions.append((rate, next_state))

        if not transitions:
            logger.debug("Нет доступных переходов из текущего состояния")

        logger.debug(f"Итого переходов: {len(transitions)} для {state_id=}")

        return transitions

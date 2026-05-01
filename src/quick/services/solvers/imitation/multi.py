"""Имитационный решатель для многоканальной СМО с уходами заявок.

Содержит класс `MultiServerImitationProbabilitySolver`, реализующий
Монте-Карло моделирование системы с времезависимыми (LOCF) интенсивностями.
"""

from loguru import logger

from quick.domain.models import ImitationSystemParams
from quick.domain.models.system_params import MultiSystemParams
from quick.services.solvers.imitation.base import BaseImitationProbabilitySolver
from quick.services.solvers.imitation.utils import TimeSeriesRateProvider


class MultiServerImitationProbabilitySolver(BaseImitationProbabilitySolver):
    """Имитационный (Монте-Карло) решатель для многоканальной СМО с LOCF-параметрами."""

    def __init__(self, system_params: ImitationSystemParams) -> None:
        """Создаёт решатель Монте-Карло для многоканальной СМО.

        Args:
            system_params (SystemParams): Объединённые параметры системы.

        Raises:
            TypeError: Если параметры системы не являются MultiSystemParams.
        """
        super().__init__(system_params)

        if not isinstance(self.base_params, MultiSystemParams):
            logger.error("Параметры не являются экземпляром MultiSystemParams")
            raise TypeError("Параметры должны быть экземпляром MultiSystemParams")

        self.rate_provider = TimeSeriesRateProvider(
            self.base_params, self.system_params.time_series_params
        )

        self.max_state: int = (
            self.base_params.max_customers + self.base_params.processor_count
        )

    @property
    def num_states(self) -> int:
        """Количество состояний системы."""
        if self._num_states is None:
            self._num_states = self.max_state + 1
        return self._num_states

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
            list[tuple[float, int]]: Список возможных переходов, где каждый элемент
                это кортеж вида (rate, next_state):
                    - rate (float): интенсивность перехода,
                    - next_state (int): состояние, в которое осуществляется переход.

        Raises:
            ValueError: Если переданное некорректное состояние системы
            TypeError: Если параметры системы не являются MultiSystemParams.
        """
        logger.debug(f"Вычисление переходов: {state_id=}, {time_point=:.4f}")

        if not 0 <= state_id <= self.max_state:
            logger.error(f"Передано некорректное состояние системы: {state_id}")
            raise ValueError(f"Некорректное состояние системы: {state_id}")

        if not isinstance(self.base_params, MultiSystemParams):
            logger.error("Параметры не являются экземпляром MultiSystemParams")
            raise TypeError("Параметры должны быть экземпляром MultiSystemParams")

        lam = self.rate_provider.get_lambda_rate(time_point)
        mu = self.rate_provider.get_mu_rate(time_point)
        nu = self.rate_provider.get_nu_rate(time_point)

        logger.debug(f"Интенсивности: {lam=:.4f}, {mu=:.4f}, {nu=:.4f}")

        in_service = min(state_id, self.base_params.processor_count)
        in_queue = max(0, state_id - self.base_params.processor_count)

        logger.debug(f"Состояние: {in_service=}, {in_queue=}")

        transitions: list[tuple[float, int]] = []

        birth_rate = lam if state_id < self.max_state else 0.0
        if birth_rate > 0.0:
            transitions.append((float(birth_rate), state_id + 1))
            logger.debug(
                f"Добавлен переход (приход): rate={birth_rate:.4f}, "
                f"{state_id} -> {state_id + 1}"
            )

        death_rate = mu * in_service + nu * in_queue
        if state_id > 0 and death_rate > 0.0:
            transitions.append((float(death_rate), state_id - 1))
            logger.debug(
                f"Добавлен переход (уход): rate={death_rate:.4f}, "
                f"{state_id} -> {state_id - 1}"
            )

        if not transitions:
            logger.debug("Нет доступных переходов из текущего состояния")

        logger.debug(f"Итого переходов: {len(transitions)} для {state_id=}")

        return transitions

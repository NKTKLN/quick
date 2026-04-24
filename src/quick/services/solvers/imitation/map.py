# TODO


import random

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain.models import ImitationSystemParams
from quick.domain.models.system_params import MAPSystemParams
from quick.utils.progress import Progress

from .utils import TimeSeriesMAPRateProvider


# %%
class MultiSensorMAPGeneratorBuilder:
    """Строит Q_row для многосенсорной MAP-СМО."""

    def __init__(
        self,
        max_customers: int,
        sensor_count: int,
        p_rate: NDArray[np.float64],
        q_rate: NDArray[np.float64],
    ) -> None:
        self.max_customers = max_customers
        self.sensor_count = sensor_count
        self.p_rate = p_rate
        self.q_rate = q_rate

    def build_q_row(
        self,
        lambda_rate: NDArray[np.float64],
        mu_rate: float,
        nu_rate: float,
    ) -> NDArray[np.float64]:
        m = lambda_rate.shape[0]
        k_max = self.max_customers
        levels = k_max + 2

        d00 = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                if i == j:
                    d00[i, j] = -lambda_rate[i]
                else:
                    d00[i, j] = lambda_rate[i] * self.p_rate[i, j]

        d11 = np.diag(lambda_rate) @ self.q_rate

        d0 = d00.T
        d1 = d11.T
        eye_m = np.eye(m, dtype=np.float64)

        q_col = np.zeros((levels, levels, m, m), dtype=np.float64)

        def down_rate(level: int) -> float:
            service = min(level, self.sensor_count) * mu_rate
            abandonment = max(level - self.sensor_count, 0) * nu_rate
            return service + abandonment

        for level in range(levels):
            if level < k_max:
                q_col[level + 1, level] = d1

            if level > 0:
                q_col[level - 1, level] = down_rate(level) * eye_m

            diag_block = d0.copy()
            if level == k_max:
                diag_block = diag_block + d1
            diag_block = diag_block - down_rate(level) * eye_m
            q_col[level, level] = diag_block

        q_row = q_col.transpose(1, 3, 0, 2).reshape(levels * m, levels * m)

        for i in range(q_row.shape[0]):
            for j in range(q_row.shape[1]):
                if i != j and q_row[i, j] < 0 and abs(q_row[i, j]) < 1e-12:
                    q_row[i, j] = 0.0

        for i in range(q_row.shape[0]):
            offdiag_sum = np.sum(q_row[i, :]) - q_row[i, i]
            q_row[i, i] = -offdiag_sum

        return q_row


class MAPImitationProbabilitySolver:
    """Имитационный решатель для MAP-СМО с LOCF-параметрами."""

    def __init__(
        self,
        system_params: ImitationSystemParams,
    ) -> None:
        # TODO

        self.system_params = system_params
        self.sim_config = system_params.simulation_params

        self.sim_config.validate()
        self.system_params.validate()

        self.base_params = system_params.base_params
        self.transient_params = system_params.transient_params
        self.rate_provider = TimeSeriesMAPRateProvider(
            self.base_params, self.system_params.time_series_params
        )

        if not isinstance(self.base_params, MAPSystemParams):
            logger.error("Параметры являются экземпляром MAPSystemParams")
            raise TypeError("Параметры не должны быть экземпляром MAPSystemParams")

        self.time_array: NDArray[np.float64] = self.transient_params.time_array
        self.initial_probabilities: NDArray[np.float64] = (
            self.transient_params.initial_probabilities
        )

        self.k_max = self.base_params.max_customers
        self.m = self.base_params.lambda_rate.shape[0]
        self.num_levels = self.k_max
        self.num_states = self.num_levels * self.m

        self.generator_builder = MultiSensorMAPGeneratorBuilder(
            self.base_params.max_customers,
            self.base_params.sensor_count,
            self.base_params.p_rate,
            self.base_params.q_rate,
        )

        self._rng = random.Random(self.sim_config.seed)

    def _sample_initial_state(self) -> int:
        """Сэмплирует начальное состояние по initial_probabilities."""
        cumulative = np.cumsum(self.initial_probabilities)
        u = self._rng.random()
        idx = int(np.searchsorted(cumulative, u, side="right"))
        return min(idx, self.num_states - 1)

    def _get_q_row(self, time_point: float) -> NDArray[np.float64]:
        """Возвращает генератор Q(t) для текущего времени."""
        lambda_t = self.rate_provider.get_lambda_rate(time_point)
        mu_t = self.rate_provider.get_mu_rate(time_point)
        nu_t = self.rate_provider.get_nu_rate(time_point)

        return self.generator_builder.build_q_row(
            lambda_rate=lambda_t,
            mu_rate=mu_t,
            nu_rate=nu_t,
        )

    def _get_transitions(
        self,
        state_id: int,
        time_point: float,
    ) -> list[tuple[float, int]]:
        """Возвращает возможные переходы из состояния в формате (rate, next_state)."""
        q_row = self._get_q_row(time_point)
        transitions: list[tuple[float, int]] = []

        for next_state in range(self.num_states):
            if next_state == state_id:
                continue
            rate = float(q_row[state_id, next_state])
            if rate > 0:
                transitions.append((rate, next_state))

        return transitions

    def _simulate_single_trajectory(self) -> NDArray[np.int32]:
        """Моделирует одну траекторию процесса на всём интервале time_array."""
        times = self.time_array
        t_max = float(times[-1])
        t_count = times.shape[0]

        states_over_time = np.zeros(t_count, dtype=np.int32)

        current_state = self._sample_initial_state()
        current_time = float(times[0])
        t_index = 0

        while t_index < t_count and times[t_index] <= current_time:
            states_over_time[t_index] = current_state
            t_index += 1

        while current_time < t_max:
            transitions = self._get_transitions(current_state, current_time)
            total_rate = sum(rate for rate, _ in transitions)

            if total_rate <= 0.0:
                while t_index < t_count:
                    states_over_time[t_index] = current_state
                    t_index += 1
                break

            dt = self._rng.expovariate(total_rate)
            next_time = current_time + dt

            while t_index < t_count and times[t_index] <= min(next_time, t_max):
                states_over_time[t_index] = current_state
                t_index += 1

            if next_time > t_max:
                current_time = t_max
                break

            u = self._rng.random() * total_rate
            cumulative = 0.0
            next_state = current_state

            for rate, candidate_state in transitions:
                cumulative += rate
                if u <= cumulative:
                    next_state = candidate_state
                    break

            current_state = next_state
            current_time = next_time

        while t_index < t_count:
            states_over_time[t_index] = current_state
            t_index += 1

        return states_over_time

    def calculate(self) -> NDArray[np.float64]:
        """Запускает Монте-Карло имитацию и возвращает матрицу вероятностей.

        Returns:
            NDArray[np.float64]:
                Матрица размера (len(time_array), num_states):
                probability[t, s] = P(X(time_array[t]) = s).
        """
        logger.info(
            "Старт MAP-имитации: %s траекторий, состояний %s, временных точек %s",
            self.sim_config.trajectories,
            self.num_states,
            self.time_array.shape[0],
        )

        t_count = self.time_array.shape[0]
        counts = np.zeros((t_count, self.num_states), dtype=np.float64)

        for run in Progress.wrap(
            range(self.sim_config.trajectories), description="Вычисление траекторий"
        ):
            states_over_time = self._simulate_single_trajectory()

            for t_idx in range(t_count):
                s = int(states_over_time[t_idx])
                counts[t_idx, s] += 1.0

            if (run + 1) % max(1, self.sim_config.trajectories // 10) == 0:
                logger.info(
                    "Выполнено траекторий: %s / %s",
                    run + 1,
                    self.sim_config.trajectories,
                )

        probability = counts / float(self.sim_config.trajectories)

        logger.info("Имитация завершена.")
        return probability.T

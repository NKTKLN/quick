# TODO


import random

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain.models import ImitationSystemParams
from quick.domain.models.system_params.multi import MultiSystemParams
from quick.utils.progress import Progress

from .utils import TimeSeriesRateProvider


class MultiServerImitationProbabilitySolver:
    """Имитационный (Монте-Карло) решатель для многоканальной СМО с уходами заявок.

    Модель:
        - Состояние i — число заявок в системе: обслуживаемые + в очереди.
        - Возможные состояния: 0, 1, ..., K, где
              K = max_customers + processor_count.
        - Переходы:
            • При i < K возможен приход: i → i + 1 с интенсивностью λ(t).
            • При i > 0 возможны:
                – завершения обслуживания: μ(t) * min(i, m),
                – уходы из очереди (нетерпение): ν(t) * max(0, i - m).

        Суммарная интенсивность уменьшения состояния на 1:
            μ(t) * min(i, m) + ν(t) * max(0, i - m).

    Attributes:
        system_params (SystemParams): Параметры системы (базовые + переходные).
        sim_config (SimulationConfig): Конфигурация Монте-Карло (траектории, seed).
        rate_provider (BaseRateProvider): Источник времезависимых λ(t), μ(t), ν(t).
        base_params (BaseSystemParams): Базовые параметры системы.
        transient_params (TransientSystemParams): Параметры переходного режима.
        time_array (NDArray[np.float64]): Временные точки, в которых фиксируется N(t).
        initial_probabilities (NDArray[np.float64]): Начальное распределение состояний.
        max_state (int): Максимальное состояние (K).
        num_states (int): Количество состояний K+1.
    """

    def __init__(
        self,
        system_params: ImitationSystemParams,
    ) -> None:
        """Создаёт решатель Монте-Карло для многоканальной СМО.

        Args:
            system_params (SystemParams):
                Объединённые параметры системы (базовые и переходные).
            sim_config (SimulationConfig):
                Количество траекторий и seed генератора случайных чисел.
            rate_provider (Optional[BaseRateProvider]):
                Источник интенсивностей λ(t), μ(t), ν(t).
                Если None → используется ConstantRateProvider.

        Raises:
            TypeError: Если параметры системы не являются MultiSystemParams.
        """
        self.system_params = system_params
        self.system_params.validate()

        self.base_params = system_params.base_params
        self.transient_params = system_params.transient_params
        self.sim_config = system_params.simulation_params

        self.time_array: NDArray[np.float64] = self.transient_params.time_array
        self.initial_probabilities: NDArray[np.float64] = (
            self.transient_params.initial_probabilities
        )

        if not isinstance(self.base_params, MultiSystemParams):
            logger.error("Параметры являются экземпляром MultiSystemParams")
            raise TypeError("Параметры не должны быть экземпляром MultiSystemParams")

        self.max_state: int = (
            self.base_params.max_customers + self.base_params.processor_count
        )
        self.num_states: int = self.max_state + 1

        self.rate_provider = TimeSeriesRateProvider(
            self.base_params, self.system_params.time_series_params
        )

        self._rng = random.Random(self.sim_config.seed)

    def _sample_initial_state(self) -> int:
        """Сэмплирует начальное состояние по initial_probabilities.

        Returns:
            int: Начальное состояние (целое число от 0 до num_states − 1).
        """
        cumulative = np.cumsum(self.initial_probabilities)
        u = self._rng.random()
        idx = int(np.searchsorted(cumulative, u, side="right"))
        return min(idx, self.num_states - 1)

    def _simulate_single_trajectory(self) -> NDArray[np.int32]:
        """Моделирует одну траекторию процесса N(t) на всём интервале time_array.

        События моделируются методом прямой эмуляции процесса:
        - генерируется экспоненциальное время до следующего события,
        - выбирается тип события (приход/уход),
        - состояние обновляется,
        - состояние записывается для всех контрольных точек времени до события.

        Returns:
            NDArray[np.int32]:
                Массив длины len(time_array), где i-й элемент — значение N(t_i).

        # TODO
        """
        times = self.time_array
        T = times.shape[0]
        t_max = float(times[-1])

        if not isinstance(self.base_params, MultiSystemParams):
            logger.error("Параметры являются экземпляром MultiSystemParams")
            raise TypeError("Параметры не должны быть экземпляром MultiSystemParams")

        states_over_time = np.zeros(T, dtype=np.int32)

        current_state = self._sample_initial_state()
        current_time = float(times[0])
        t_index = 0

        # Заполнение начального состояния
        while t_index < T and times[t_index] <= current_time:
            states_over_time[t_index] = current_state
            t_index += 1

        # Цикл событий
        while current_time < t_max:
            in_service = min(current_state, self.base_params.processor_count)
            in_queue = max(0, current_state - self.base_params.processor_count)

            lam = self.rate_provider.get_lambda_rate(current_time)
            mu = self.rate_provider.get_mu_rate(current_time)
            nu = self.rate_provider.get_nu_rate(current_time)

            birth_rate = lam if current_state < self.max_state else 0.0
            service_rate_total = mu * in_service
            abandon_rate_total = nu * in_queue
            death_rate_total = service_rate_total + abandon_rate_total

            total_rate = birth_rate + death_rate_total

            # Если событий больше не будет
            if total_rate <= 0.0:
                while t_index < T:
                    states_over_time[t_index] = current_state
                    t_index += 1
                break

            # Время до следующего события
            dt = self._rng.expovariate(total_rate)
            next_time = current_time + dt

            # Записываем состояние для контрольных точек
            while t_index < T and times[t_index] <= min(next_time, t_max):
                states_over_time[t_index] = current_state
                t_index += 1

            # Если следующее событие за пределами интереса
            if next_time > t_max:
                current_time = t_max
                break

            # Тип события
            u = self._rng.random() * total_rate
            if u < birth_rate:
                if current_state < self.max_state:
                    current_state += 1
            else:
                if current_state > 0:
                    current_state -= 1

            current_time = next_time

        # Заполнение остатка
        while t_index < T:
            states_over_time[t_index] = current_state
            t_index += 1

        return states_over_time

    def calculate(self) -> NDArray[np.float64]:
        """Запускает Монте-Карло имитацию и возвращает матрицу вероятностей.

        Выполняется sim_config.trajectories траекторий,
        для каждой строится N(t) для всех временных точек.
        Далее оценки вероятностей получаются как относительные частоты.

        Returns:
            NDArray[np.float64]:
                Матрица размера (len(time_array), num_states):
                probability[t, s] = P(N(time_array[t]) = s).
        """
        logger.info(
            f"Старт имитации: {self.sim_config.trajectories} траекторий, "
            f"состояний {self.num_states}, временных точек {self.time_array.shape[0]}",
        )

        T = self.time_array.shape[0]
        S = self.num_states
        counts = np.zeros((T, S), dtype=np.float64)

        for run in Progress.wrap(
            range(self.sim_config.trajectories), description="Вычисление траекторий"
        ):
            states_over_time = self._simulate_single_trajectory()

            for t_idx in range(T):
                s = int(states_over_time[t_idx])
                counts[t_idx, s] += 1.0

            if (run + 1) % max(1, self.sim_config.trajectories // 10) == 0:
                logger.info(
                    f"Выполнено траекторий: {run + 1} / {self.sim_config.trajectories}",
                )

        probability = counts / float(self.sim_config.trajectories)

        logger.info("Имитация завершена.")

        return probability.T

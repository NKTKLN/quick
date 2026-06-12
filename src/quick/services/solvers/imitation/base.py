"""Базовый модуль имитационного решателя вероятностей для СМО.

Содержит класс `BaseImitationProbabilitySolver`, который определяет интерфейс и
общие компоненты для Монте-Карло моделирования систем массового обслуживания (СМО)
с возможными уходами заявок. Конкретные реализации должны наследовать этот класс
и реализовывать методы генерации траекторий и расчёта вероятностей.
"""

import random
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain.params import ImitationSystemParams
from quick.services.solvers.base import BasicProbabilitySolver
from quick.utils.progress import Progress


class BaseImitationProbabilitySolver(BasicProbabilitySolver, ABC):
    """Базовый класс имитационного решателя вероятностей для СМО.

    Задает общий интерфейс и базовые атрибуты для реализации имитационного
    (Монте-Карло) моделирования систем массового обслуживания (СМО).
    Поддерживает работу с переходными режимами и начальными распределениями состояний.
    """

    def __init__(self, system_params: ImitationSystemParams) -> None:
        """Инициализирует базовый имитационный решатель.

        Args:
            system_params (ImitationSystemParams): Объединённые параметры системы.

        Raises:
            ValueError: Если параметры системы некорректны.
        """
        self.system_params = system_params
        self.system_params.validate()

        self.base_params = system_params.base_params
        self.transient_params = system_params.transient_params
        self.sim_config = system_params.simulation_params

        if self.transient_params is None:
            logger.error("Параметр transient_params обязательн в переходном режиме")
            raise ValueError("transient_params обязательны в переходном режиме")

        self.time_array: NDArray[np.float64] = self.transient_params.time_array
        self.initial_probabilities: NDArray[np.float64] = (
            self.transient_params.initial_probabilities
        )

        self._rng = random.Random(self.sim_config.seed)  # noqa: S311

        self._num_states: int | None = None

    @property
    @abstractmethod
    def num_states(self) -> int:
        """Количество состояний системы."""
        raise NotImplementedError

    def _sample_initial_state(self) -> int:
        """Сэмплирует начальное состояние системы.

        Returns:
            int: Начальное состояние системы.

        Raises:
            ValueError: Если количество состояний не задано.
        """
        logger.debug("Сэмплирование начального состояния...")

        cumulative = np.cumsum(self.initial_probabilities)
        u = self._rng.random()
        idx = int(np.searchsorted(cumulative, u, side="right"))
        state = min(idx, self.num_states - 1)

        logger.debug(f"Начальное состояние выбрано: {state}")
        return state

    @abstractmethod
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
            list[tuple[float, int]]: Список возможных переходов, где каждый элемент —
                это кортеж вида (rate, next_state):
                    - rate (float): интенсивность перехода,
                    - next_state (int): состояние, в которое осуществляется переход.
        """
        raise NotImplementedError

    def _simulate_single_trajectory(self) -> NDArray[np.int32]:
        """Моделирует одну траекторию процесса N(t) на всём интервале time_array.

        Returns:
            NDArray[np.int32]: Массив длины len(time_array),
                где i-й элемент это значение N(t_i).
        """
        logger.debug("Начинаем моделирование одной траектории...")

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

        step = 0
        while current_time < t_max:
            transitions = self._get_transitions(current_state, current_time)
            total_rate = sum(rate for rate, _ in transitions)

            if total_rate <= 0.0:
                logger.debug(
                    "Нет доступных переходов (total_rate=0), "
                    "заполняем остаток траектории"
                )
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

            logger.debug(
                f"Шаг {step}: состояние {current_state} -> {next_state} "
                f"в момент времени {next_time:.4f}"
            )

            current_state = next_state
            current_time = next_time
            step += 1

        while t_index < t_count:
            states_over_time[t_index] = current_state
            t_index += 1

        logger.debug("Моделирование одной траектории завершено")
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

        logger.debug("Начинаем генерацию траекторий...")

        for run in Progress.wrap(
            range(self.sim_config.trajectories), description="Вычисление траекторий"
        ):
            states_over_time = self._simulate_single_trajectory()

            for t_idx in range(T):
                s = int(states_over_time[t_idx])
                counts[t_idx, s] += 1.0

            logger.debug(f"Траектория {run + 1} обработана")

            if (run + 1) % max(1, self.sim_config.trajectories // 10) == 0:
                logger.info(
                    f"Выполнено траекторий: {run + 1} / {self.sim_config.trajectories}",
                )

        logger.debug("Все траектории сгенерированы, вычисляем вероятности...")

        probability = counts / float(self.sim_config.trajectories)

        logger.info("Имитация завершена.")

        return probability.T

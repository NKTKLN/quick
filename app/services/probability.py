"""Модуль для моделирования систем массового обслуживания с нетерпеливыми заявками.

Содержит классы для расчёта вероятностных характеристик однолинейных и многолинейных СМО
с использованием матричных методов. Поддерживает вычисления как с обычной точностью
(numpy), так и с повышенной точностью (mpmath).

Основные классы:
    - BaseProbabilitySystem: абстрактный базовый класс для всех решателей
    - SingleServerSystem: реализация для однолинейной СМО
    - MultiServerSystem: реализация для многолинейной СМО

Основные функции:
    - Построение матриц переходов между состояниями системы
    - Вычисление собственных значений матрицы переходов
    - Решение системы уравнений для нахождения вероятностей состояний
    - Расчёт динамики вероятностей во времени
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import mpmath as mp  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig

from app.domain import (
    CalculationType,
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
    MultiServerParams,
    SingleServerParams,
)
from app.services.matrix_generators import (
    MultiServerMatrixBuilder,
    SingleServerMatrixBuilder,
)
from app.services.solver import (
    MergedProbabilitySolver,
    MpmathProbabilitySolver,
    NumpyProbabilitySolver,
)

# Настройка логирования для отслеживания работы системы
logger = logging.getLogger(__name__)


class BaseProbabilitySystem(ABC):
    """Абстрактный базовый класс для моделирования систем массового обслуживания (СМО).

    Предоставляет общий интерфейс и базовую реализацию для расчёта вероятностных
    характеристик СМО с нетерпеливыми заявками. Классы-наследники должны реализовать
    специфичную логику построения матриц переходов для конкретных типов систем.

    Основные функции:
        - Управление процессом расчёта вероятностей состояний системы
        - Координация работы компонентов (построение матриц, решение уравнений)
        - Предоставление единого интерфейса для различных типов СМО

    Методы:
        - calculate(): основной метод для выполнения полного расчёта
        - _build_transition_matrix(): абстрактный метод построения матрицы переходов
        - _compute_eigenvalues(): вычисление собственных значений матрицы
        - _solve_probability_system(): решение системы уравнений для вероятностей
    """

    def __init__(
        self,
        params: Any,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует систему массового обслуживания с заданными параметрами.

        Args:
            params: Объект параметров системы, содержащий:
                   - Интенсивности потоков (входящий, обслуживания, ухода)
                   - Структурные параметры системы (количество каналов, емкость очереди)
                   - Другие специфичные параметры конкретной СМО
            config: Конфигурация вычислений
        """
        self.params = params
        self.config = config

    def calculate(self) -> NDArray[np.float64]:
        """Выполняет полный расчёт вероятностей состояний системы.

        Процесс расчёта включает:
        1. Построение матрицы переходов системы
        2. Вычисление собственных значений матрицы
        3. Решение системы уравнений для вероятностей
        4. Построение итоговой матрицы вероятностей состояний

        Returns:
            NDArray[np.float64]: Матрица вероятностей размерностью (N+1)x(M+1),
                               где N — ёмкость системы, M — число источников.
                               P[i,j] — вероятность состояния с i заявками в системе
                               и j занятыми источниками.
        """
        transition_matrix = self._build_transition_matrix()
        eigenvalues, xsi_matrix = self._compute_eigenvalues(transition_matrix)
        probability_matrix = self._solve_probability_system(
            transition_matrix, eigenvalues, xsi_matrix
        )
        return probability_matrix

    @abstractmethod
    def _build_transition_matrix(self) -> NDArray[np.float64]:
        """Абстрактный метод построения матрицы переходов между состояниями СМО.

        Returns:
            NDArray[np.float64]: Матрица переходов между состояниями системы.
        """
        pass

    def _compute_eigenvalues(
        self, transition_matrix: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[Any, Any]:
        """Вычисляет собственные значения матрицы переходов.

        Args:
            transition_matrix: Матрица переходов между состояниями системы.

        Returns:
            tuple: Кортеж из двух элементов:
                - Массив собственных значений матрицы
                - Матрица собственных векторов
        """
        if self.config.calculation_type in [
            CalculationType.MPMATH,
            CalculationType.MERGED,
        ] and isinstance(
            self.config, MpmathComputationConfig | MergedComputationConfig
        ):
            with mp.workdps(self.config.precision):
                mp_matrix = mp.matrix(transition_matrix.tolist())
                eigenvalues, xsi_matrix = mp.eig(mp_matrix)
            return eigenvalues, xsi_matrix

        eigenvalues, xsi_matrix = eig(transition_matrix)
        return eigenvalues, xsi_matrix

    def _solve_probability_system(
        self,
        transition_matrix: NDArray[np.float64],
        eigenvalues: NDArray[np.float64] | Any,
        xsi_matrix: NDArray[np.float64] | Any,
    ) -> NDArray[np.float64]:
        """Решает систему уравнений для нахождения стационарных вероятностей.

        Args:
            transition_matrix: Матрица переходов системы.
            eigenvalues: Собственные значения матрицы переходов.
            xsi_matrix: Матрица собственных векторов.

        Returns:
            NDArray[np.float64]: Матрица стационарных вероятностей состояний системы.
        """
        prob_solver: (
            NumpyProbabilitySolver | MpmathProbabilitySolver | MergedProbabilitySolver
        )
        match self.config.calculation_type:
            case CalculationType.NUMPY if isinstance(
                self.config, ComputationConfig
            ):
                prob_solver = NumpyProbabilitySolver(
                    self.params, transition_matrix, eigenvalues, self.config
                )
            case CalculationType.MPMATH if isinstance(
                self.config, MpmathComputationConfig
            ):
                prob_solver = MpmathProbabilitySolver(
                    self.params, transition_matrix, eigenvalues, self.config
                )
            case CalculationType.MERGED if isinstance(
                self.config, MergedComputationConfig
            ):
                prob_solver = MergedProbabilitySolver(
                    self.params, transition_matrix, eigenvalues, self.config
                )
            case _:
                raise ValueError("Unsupported calculation type")

        m_matrix = prob_solver.generate_m_matrix(xsi_matrix)
        return prob_solver.generate_p_matrix(m_matrix)


class SingleServerSystem(BaseProbabilitySystem):
    """Класс для моделирования однолинейной СМО с нетерпеливыми заявками.

    Реализует расчёт вероятностных характеристик системы с одним обслуживающим прибором.
    """

    def __init__(
        self,
        params: SingleServerParams,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует систему массового обслуживания с заданными параметрами.

        Args:
            params: Объект SingleServerParams, содержащий параметры системы:
                    - lambda_rate: интенсивность входящего потока
                    - mu_rate: интенсивность обслуживания
                    - nu_rate: интенсивность ухода заявок из очереди
                    - channel_count: количество каналов обслуживания
                    - queue_capacity: максимальная длина очереди
            config: Конфигурация вычислений
        """
        super().__init__(params, config)

    def _build_transition_matrix(self) -> NDArray[np.float64]:
        """Строит матрицу переходов системы массового обслуживания.

        Returns:
            NDArray[np.float64]: Матрица переходов между состояниями системы.
        """
        transition_matrix = SingleServerMatrixBuilder(self.params).build()
        return transition_matrix


class MultiServerSystem(BaseProbabilitySystem):
    """Класс для моделирования многолинейной СМО с нетерпеливыми заявками.

    Реализует расчёт вероятостей системы с несколькими обслуживающими приборами.
    """

    def __init__(
        self,
        params: MultiServerParams,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует систему массового обслуживания с заданными параметрами.

        Args:
            params: Объект MultiServerParams, содержащий параметры системы:
                    - lambda_rate: интенсивность входящего потока
                    - mu_rate: интенсивность обслуживания
                    - nu_rate: интенсивность ухода заявок из очереди
                    - channel_count: количество каналов обслуживания
                    - queue_capacity: максимальная длина очереди
                    - processor_count: количество обслуживающих приборов в системе
            config: Конфигурация вычислений
        """
        super().__init__(params, config)

    def _build_transition_matrix(self) -> NDArray[np.float64]:
        """Строит матрицу переходов многолинейной системы массового обслуживания.

        Returns:
            NDArray[np.float64]: Матрица переходов между состояниями системы.
        """
        transition_matrix = MultiServerMatrixBuilder(self.params).build()
        return transition_matrix

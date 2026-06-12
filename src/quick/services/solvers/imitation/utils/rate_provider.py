"""Модуль поставщиков интенсивностей для СМО.

Реализует классы BaseRateProvider, TimeSeriesRateProvider и
TimeSeriesMAPRateProvider, которые задают интерфейс и реализации источников
интенсивностей поступления, обслуживания и ухода заявок.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from quick.domain.params import (
    BaseSystemParams,
    MAPSystemParams,
    TimeSeriesBaseSystemParams,
)

LambdaRateT = TypeVar("LambdaRateT", float, NDArray[np.float64])


class BaseRateProvider[LambdaRateT: (float, NDArray[np.float64])](ABC):
    """Абстрактный базовый класс источника интенсивностей λ(t), μ(t), ν(t).

    Определяет универсальный интерфейс, от которого должны наследоваться
    все поставщики времезависимых параметров СМО.
    """

    @abstractmethod
    def get_lambda_rate(self, time_point: float) -> LambdaRateT:
        """Возвращает интенсивность поступления λ(t).

        Args:
            time_point (float): Момент времени, для которого требуется λ.

        Returns:
            LambdaRateT: Значение интенсивности λ(time_point).
        """
        raise NotImplementedError

    @abstractmethod
    def get_mu_rate(self, time_point: float) -> float:
        """Возвращает интенсивность обслуживания μ(t).

        Args:
            time_point (float): Момент времени, для которого требуется μ.

        Returns:
            float: Значение интенсивности μ(time_point).
        """
        raise NotImplementedError

    @abstractmethod
    def get_nu_rate(self, time_point: float) -> float:
        """Возвращает интенсивность ухода из очереди ν(t).

        Args:
            time_point (float): Момент времени, для которого требуется ν.

        Returns:
            float: Значение интенсивности ν(time_point).
        """
        raise NotImplementedError


class TimeSeriesRateProvider(BaseRateProvider[float]):
    """Источник интенсивностей λ(t), μ(t), ν(t), основанный на временных рядах.

    Attributes:
        base_params (BaseSystemParams): Базовые параметры системы.
        lambda_series (Optional[TimeSeries]): Временной ряд λ(t), если задан.
        mu_series (Optional[TimeSeries]): Временной ряд μ(t), если задан.
        nu_series (Optional[TimeSeries]): Временной ряд ν(t), если задан.
    """

    def __init__(
        self,
        base_params: BaseSystemParams,
        time_series_params: TimeSeriesBaseSystemParams | None = None,
    ) -> None:
        """Создаёт поставщика интенсивностей на основе временных рядов.

        Args:
            base_params (BaseSystemParams):
                Базовые параметры, содержащие значения интенсивностей по умолчанию.
            time_series_params (TimeSeriesBaseSystemParams):
                Базовые параметры, зывисымые от времени.

        Raises:
            ValueError: Если временные ряды заданы, но невалидны.
        """
        base_params.validate()
        self.base_params = base_params

        if time_series_params is not None:
            time_series_params.validate()

        self.time_series_params = time_series_params

    def get_lambda_rate(self, time_point: float) -> float:
        """Возвращает значение λ(t) в заданный момент времени.

        Args:
            time_point (float): Момент времени, для которого требуется значение интенсивности.

        Returns:
            float: Значение λ(t) из временного ряда или константа из base_params,
                если временной ряд не задан.
        """
        if (
            self.time_series_params is None
            or self.time_series_params.lambda_rate is None
        ):
            return float(self.base_params.lambda_rate)
        return float(self.time_series_params.lambda_rate.value_at(time_point))

    def get_mu_rate(self, time_point: float) -> float:
        """Возвращает значение μ(t) в заданный момент времени.

        Args:
            time_point (float): Момент времени, для которого требуется значение интенсивности.

        Returns:
            float: Значение μ(t) из временного ряда или константа из base_params,
                если временной ряд не задан.
        """
        if self.time_series_params is None or self.time_series_params.mu_rate is None:
            return float(self.base_params.mu_rate)
        return float(self.time_series_params.mu_rate.value_at(time_point))

    def get_nu_rate(self, time_point: float) -> float:
        """Возвращает значение ν(t) в заданный момент времени.

        Args:
            time_point (float): Момент времени, для которого требуется значение интенсивности.

        Returns:
            float: Значение ν(t) из временного ряда или константа из base_params,
                если временной ряд не задан.
        """
        if self.time_series_params is None or self.time_series_params.nu_rate is None:
            return float(self.base_params.nu_rate)
        return float(self.time_series_params.nu_rate.value_at(time_point))


class TimeSeriesMAPRateProvider(BaseRateProvider[NDArray[np.float64]]):
    """Поставщик времезависимых параметров lambda(t), mu(t), nu(t)."""

    def __init__(
        self,
        base_params: MAPSystemParams,
        time_series_params: TimeSeriesBaseSystemParams | None = None,
    ) -> None:
        """Создаёт поставщика интенсивностей на основе временных рядов.

        Args:
            base_params (MAPSystemParams):
                Базовые параметры, содержащие значения интенсивностей по умолчанию.
            time_series_params (TimeSeriesBaseSystemParams):
                Базовые параметры, зывисымые от времени.

        Raises:
            ValueError: Если временные ряды заданы, но невалидны.
        """
        base_params.validate()
        self.base_params = base_params

        if time_series_params is not None:
            time_series_params.validate()

        self.time_series_params = time_series_params

    def get_lambda_rate(self, time_point: float) -> NDArray[np.float64]:
        """Возвращает значение λ(t) в виде массива для MAP-потока.

        Args:
            time_point (float): Момент времени, для которого требуется значение интенсивности.

        Returns:
            NDArray[np.float64]: Матрица (или вектор) λ(t), полученная из временного ряда
                или скопированная из base_params, если временной ряд не задан.
        """
        if (
            self.time_series_params is None
            or self.time_series_params.lambda_rate is None
        ):
            return self.base_params.lambda_rate
        return cast(
            NDArray[np.float64],
            self.time_series_params.lambda_rate.value_at(time_point),
        )

    def get_mu_rate(self, time_point: float) -> float:
        """Возвращает значение μ(t) в заданный момент времени.

        Args:
            time_point (float): Момент времени, для которого требуется значение интенсивности.

        Returns:
            float: Значение μ(t) из временного ряда или константа из base_params,
                если временной ряд не задан.
        """
        if self.time_series_params is None or self.time_series_params.mu_rate is None:
            return float(self.base_params.mu_rate)
        return float(self.time_series_params.mu_rate.value_at(time_point))

    def get_nu_rate(self, time_point: float) -> float:
        """Возвращает значение ν(t) в заданный момент времени.

        Args:
            time_point (float): Момент времени, для которого требуется значение интенсивности.

        Returns:
            float: Значение ν(t) из временного ряда или константа из base_params,
                если временной ряд не задан.
        """
        if self.time_series_params is None or self.time_series_params.nu_rate is None:
            return float(self.base_params.nu_rate)
        return float(self.time_series_params.nu_rate.value_at(time_point))

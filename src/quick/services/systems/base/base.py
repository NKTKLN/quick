"""Модуль, содержащий абстрактный базовый класс для серверных систем массового обслуживания (СМО).

Включает в себя определение класса BaseServerSystem с основными методами и свойствами
для расчёта вероятностей состояний, интенсивности входного потока, коэффициентов загрузки
и других базовых характеристик СМО.
"""

from abc import ABC, abstractmethod

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain import (
    ComputationConfig,
)
from quick.domain.params import StabilityParams, SystemParams
from quick.services.stability import (
    StabilityResult,
    calculate_stability_series,
    evaluate_stability,
)
from quick.services.systems_behavior import BaseSystemBehavior


class BaseServerSystem(ABC):
    """Абстрактный базовый класс серверной системы массового обслуживания (СМО).

    Обеспечивает общий интерфейс и базовые свойства для вычисления вероятностей
    состояний, интенсивности поступления заявок и производных характеристик
    системы массового обслуживания.
    """

    def __init__(
        self,
        params: SystemParams,
        system_behavior: type[BaseSystemBehavior],
        config: ComputationConfig | None = None,
    ) -> None:
        """Инициализирует систему массового обслуживания с заданными параметрами.

        Args:
            params (ServerParams): Параметры СМО.
            system_behavior (BaseSystemBehavior): Зависимости СМО.
            config (ComputationConfig | None, optional): Конфигурация вычислений.
        """
        self.params = params
        self.config = config
        self.system_behavior = system_behavior(self.params)
        self._probabilities: NDArray[np.float64] | None = None
        self._lambda_rate: float | NDArray[np.float64] | None = None

        logger.debug(
            f"СМО {self.__class__.__name__} инициализирована с параметрами: "
            f"{params.__class__.__name__=}, {config.__class__.__name__=}"
        )

    @property
    def beta(self) -> float:
        """Отношение интенсивности ухода заявок к интенсивности обслуживания.

        Returns:
            float: Значение коэффициента beta = nu_rate / mu_rate.
        """
        return self.params.base_params.nu_rate / self.params.base_params.mu_rate

    @property
    def rho(self) -> float | NDArray[np.float64]:
        """Коэффициент загрузки системы.

        Returns:
            float | NDArray[np.float64] : Значение коэффициента загрузки rho = lambda_rate / mu_rate.
        """
        return self.lambda_rate / self.params.base_params.mu_rate

    @property
    def stability_params(self) -> StabilityParams:
        """Параметры оценки устойчивости системы.

        Returns:
            StabilityParams: Критический уровень характеристики и границы
                областей устойчивости.
        """
        return self.params.stability_params

    @abstractmethod
    def calculate_stability_base_metric(self) -> NDArray[np.float64]:
        """Возвращает характеристику ``a(t)``, по которой оценивается устойчивость.

        Конкретная величина зависит от типа системы и от настройки
        ``stability_params.metric``.

        Returns:
            NDArray[np.float64]: Значения ``a(t)``.
        """
        raise NotImplementedError

    def calculate_stability_coefficient(self) -> NDArray[np.float64]:
        r"""Вычисляет мгновенный коэффициент устойчивости ``K_уст(t)``.

        .. math::

            K_{уст}(t)=\frac{a(t)}{a_{кр}}.

        Значения ниже единицы отмечают моменты, в которые система не
        выдерживает требуемого качества обслуживания.

        Returns:
            NDArray[np.float64]: Значения ``K_уст(t)``.
        """
        logger.info("Вычисление коэффициента устойчивости K_уст(t)")

        coefficient = calculate_stability_series(
            self.calculate_stability_base_metric(),
            self.stability_params.critical_level,
        )

        logger.success("Коэффициент устойчивости K_уст(t) успешно вычислен")
        return coefficient

    def evaluate_stability_components(self) -> list[StabilityResult]:
        """Оценивает устойчивость по каждой компоненте характеристики.

        При покомпонентном расчёте характеристика имеет форму
        ``[время, датчик]``, и устойчивость определена для каждого датчика
        отдельно: общего числа тут быть не может, потому что критический
        уровень предъявляется к каждому потоку. В обычном расчёте список
        состоит из единственного элемента.

        Returns:
            list[StabilityResult]: Оценки устойчивости по компонентам в
                порядке их следования в характеристике.

        Raises:
            ValueError: Если расчёт выполнен не в переходном режиме и
                временная ось отсутствует.
        """
        if self.params.transient_params is None:
            raise ValueError(
                "Оценка устойчивости определена только для переходного "
                "режима: без временной оси интеграл по времени не задан."
            )

        values = np.asarray(self.calculate_stability_base_metric(), dtype=np.float64)

        if values.ndim == 1:
            values = values[:, np.newaxis]

        time_array = self.params.transient_params.time_array

        return [
            evaluate_stability(
                values=values[:, component],
                time_array=time_array,
                params=self.stability_params,
            )
            for component in range(values.shape[1])
        ]

    def evaluate_stability(self) -> StabilityResult:
        """Оценивает устойчивость системы за переходный интервал.

        Returns:
            StabilityResult: Интегральный коэффициент устойчивости, запас
                ``R``, длительность переходного процесса и качественная
                оценка.

        Raises:
            ValueError: Если расчёт выполнен не в переходном режиме, а также
                если характеристика разложена по датчикам: тогда единой
                оценки не существует и следует использовать
                ``evaluate_stability_components()``.
        """
        results = self.evaluate_stability_components()

        if len(results) != 1:
            raise ValueError(
                "Характеристика разложена по датчикам: единой оценки "
                "устойчивости не существует, используйте "
                "evaluate_stability_components()."
            )

        return results[0]

    @abstractmethod
    def calculate_probabilities(self) -> None:
        """Выполняет расчет вектора вероятностей состояний СМО.

        Метод должен быть реализован в подклассах, в соответствии с конкретной
        моделью СМО. Результат расчёта должен сохраняться во внутреннем
        поле `_probabilities`.
        """
        raise NotImplementedError

    @property
    def lambda_rate(self) -> float | NDArray[np.float64]:
        """Ленивая загрузка: возвращает интенсивность поступления заявок.

        Raises:
            ValueError: Если параметры имеют тип MAPSystemParams.
        """
        if self._lambda_rate is None:
            self._lambda_rate = self.system_behavior.lambda_rate
        return self._lambda_rate

    @property
    def probabilities(self) -> NDArray[np.float64]:
        """Ленивая загрузка: возвращает вероятности состояний или инициирует их расчёт.

        Returns:
            NDArray[np.float64]: Вектор вероятностей состояний системы.
        """
        if self._probabilities is None:
            self.calculate_probabilities()

        if self._probabilities is None:
            raise RuntimeError("Вероятности не были вычислены.")

        return self._probabilities

    @abstractmethod
    def calculate(self) -> dict[str, NDArray[np.float64]]:
        """Универсальный метод вычислений.

        Returns:
            dict[str, NDArray[np.float64]]: Результат вычислений.
        """
        raise NotImplementedError

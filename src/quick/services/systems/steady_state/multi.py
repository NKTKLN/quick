"""Модуль численного решения стационарных вероятностей для многоканальной СМО.

Содержит реализацию класса MultiServerSteadyStateSystem, который рассчитывает
стационарные вероятности состояний многоканальной СМО методом аналитических формул.
"""

from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.special import factorial

from quick.domain.models.system_params import MultiSystemParams
from quick.services.systems.base import BaseMultiServerSystem


class MultiServerSteadyStateSystem(BaseMultiServerSystem):
    """Класс для расчёта стационарных вероятностей состояний многоканальной СМО.

    Реализует вычисление вероятностей на основе аналитических выражений,
    учитывая максимальное число заявок в системе, количество процессоров,
    и параметры нагрузки.
    """

    def _calculate_p_0(self, eps: float = 1e-16, max_m: int = 10000) -> float:
        """Вычисляет начальную стационарную вероятность состояния 0 (P_0).

        Args:
            eps (float, optional): Порог для прекращения суммы. По умолчанию 1e-16.
            max_m (int, optional): Максимальное число итераций для ряда.
                По умолчанию 10000.

        Returns:
            float: Вероятность отсутствия заявок в системе (P_0).
        """
        n = self.params.base_params.max_customers

        k = np.arange(n + 1)
        sum1 = np.sum((self.rho**k) / factorial(k))

        sum2 = 0.0
        for m in range(1, max_m + 1):
            beta_terms = n + self.beta * np.arange(1, m + 1)
            denom = np.prod(beta_terms)
            term = (self.rho**m) / denom
            sum2 += term
            if term < eps:
                logger.debug(f"Ряд P_0 сошёлся на итерации {m} с членом {term}")
                break

        term_factor = (self.rho**n) / factorial(n)
        P_0 = cast(float, 1.0 / (sum1 + term_factor * sum2))
        logger.debug(f"Вычислено P_0: {P_0}")
        return P_0

    def _calculate_p_n(self, P_0: float) -> NDArray[np.float64]:
        """Вычисляет вероятности состояний с количеством заявок от 1 до n.

        Args:
            P_0 (float): Вероятность состояния 0.

        Returns:
            np.ndarray: Вектор вероятностей состояний от 1 до n.
        """
        n = self.params.base_params.max_customers
        k = np.arange(1, n + 1)
        P_n = (self.rho**k / factorial(k)) * P_0
        logger.debug(f"Вычислены вероятности P_n для n=1..{n}")
        return P_n

    def _calculate_p_m(self, P_n: float) -> NDArray[np.float64]:
        """Вычисляет вероятности состояний с количеством заявок свыше n.

        Args:
            P_n (float): Вероятность состояния с n заявками (конец предыдущего массива).

        Returns:
            np.ndarray: Вектор вероятностей состояний для m процессоров.

        Raises:
            ValueError: Если параметры системы не являются MultiSystemParams.
        """
        if not isinstance(self.params.base_params, MultiSystemParams):
            logger.error("Базовые параметры не являются экземпляром MultiSystemParams")
            raise ValueError(
                "Базовые параметры должны быть экземпляром MultiSystemParams"
            )

        n = self.params.base_params.max_customers
        m = self.params.base_params.processor_count

        P_m = np.zeros(m)
        for j in range(1, m + 1):
            beta_terms = n + self.beta * np.arange(1, j + 1)
            P_m[j - 1] = ((self.rho**j) / np.prod(beta_terms)) * P_n
        logger.debug(f"Вычислены вероятности P_m для m=1..{m}")
        return cast(NDArray[np.float64], P_m)

    def calculate_probabilities(self) -> None:
        """Вычисляет стационарные вероятности состояний СМО."""
        logger.info("Начат расчёт стационарных вероятностей состояний СМО")

        P_0 = self._calculate_p_0()
        P_n = self._calculate_p_n(P_0)
        P_m = self._calculate_p_m(P_n[-1])

        P = np.hstack((P_0, P_n, P_m))
        total = np.sum(P)
        if not np.isclose(total, 1.0, atol=1e-10):
            logger.warning(f"Сумма вероятностей = {total}, выполняется перенормировка")
            P /= total

        logger.success("Полный расчёт стационарных вероятностей завершён успешно")
        self._probabilities = P.reshape(-1, 1)

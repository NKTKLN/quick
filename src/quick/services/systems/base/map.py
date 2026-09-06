"""Базовые расчёты характеристик СМО с MAP-входным потоком.

Модуль содержит общий класс для стационарных и переходных MAP-СМО. На
основании вероятностей состояний вычисляются показатель заполнения буфера
``N_b(t)``, интенсивность входного потока ``lambda(t)``, вероятности отказа,
ухода, потерь и обслуживания, а также пропускная способность.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from quick.domain.enums import StabilityMetric
from quick.domain.params import MAPSystemParams
from quick.services.systems_behavior import MultiSensorMAPSystemBehavior

from .base import BaseServerSystem


class BaseMAPServerSystem(BaseServerSystem):
    r"""Абстрактный базовый класс серверной СМО с MAP-входным потоком.

    Класс поддерживает два режима расчёта характеристик:

    * агрегированный — суммирование по всем датчикам,
      :math:`\sum_{i=0}^{M-1}P(k,i,t)`;
    * индивидуальный — расчёт только по состояниям выбранного датчика
      ``j``, то есть :math:`P(k,j,t)`.

    Режим задаётся полем ``calculation_params.sensor_index``: ``None``
    соответствует агрегированному расчёту. Формулы характеристик при этом
    не дублируются: индивидуальный режим сужает ось MAP-фаз до одного
    столбца, и то же самое суммирование по этой оси даёт нужное слагаемое.
    """

    @property
    def sensor_index(self) -> int | None:
        """Номер датчика, по состояниям которого ведётся расчёт.

        Returns:
            int | None: Номер MAP-фазы в индивидуальном режиме или ``None``
                в агрегированном.
        """
        return self.params.calculation_params.sensor_index

    def _selected_phase_indices(self) -> NDArray[np.intp]:
        """Возвращает номера MAP-фаз, участвующих в расчёте.

        Returns:
            NDArray[np.intp]: Все номера фаз в агрегированном режиме либо
                единственный выбранный номер в индивидуальном.

        Raises:
            TypeError: Если параметры системы не являются MAPSystemParams.
        """
        if not isinstance(self.params.base_params, MAPSystemParams):
            logger.error("Базовые параметры не являются экземпляром MAPSystemParams")
            raise TypeError("Базовые параметры должны быть экземпляром MAPSystemParams")

        if self.sensor_index is None:
            return np.arange(self.params.base_params.sensor_count, dtype=np.intp)

        return np.array([self.sensor_index], dtype=np.intp)

    def _reshape_map_probabilities(self) -> NDArray[np.float64]:
        """Преобразует вероятности к форме ``[уровень, фаза, время]``.

        В индивидуальном режиме ось фаз сужается до выбранного датчика,
        поэтому последующее суммирование по ней возвращает ``P(k, j, t)``.

        Returns:
            NDArray[np.float64]: Массив вероятностей формы
                ``[num_macro_states, len(selected_phases), time_count]``.

        Raises:
            TypeError: Если параметры системы не являются MAPSystemParams.
            ValueError: Если число состояний не кратно числу фаз MAP.
        """
        if not isinstance(self.params.base_params, MAPSystemParams):
            logger.error("Базовые параметры не являются экземпляром MAPSystemParams")
            raise TypeError("Базовые параметры должны быть экземпляром MAPSystemParams")

        phase_count = self.params.base_params.sensor_count
        probabilities = np.asarray(self.probabilities, dtype=np.float64)

        if probabilities.ndim == 1:
            probabilities = probabilities[:, np.newaxis]

        if probabilities.ndim != 2:
            raise ValueError(
                "Матрица вероятностей должна иметь форму [state, time]: "
                f"получено {probabilities.shape}."
            )

        if probabilities.shape[0] % phase_count != 0:
            raise ValueError(
                "Размер вектора вероятностей не кратен числу фаз MAP: "
                f"{probabilities.shape[0]} % {phase_count} != 0"
            )

        probabilities_by_level = probabilities.reshape(
            probabilities.shape[0] // phase_count,
            phase_count,
            probabilities.shape[1],
        )

        return probabilities_by_level[:, self._selected_phase_indices(), :]

    def _calculate_initial_expected_customers(self) -> float:
        r"""Вычисляет :math:`\mathbb{E}N(t_0)` по начальному распределению.

        Для модели из статьи уровень ``k`` двумерной цепи Маркова означает,
        что в системе находится ``k`` заявок: одна может обслуживаться, а
        остальные находятся в буфере. Поэтому

        .. math::

            \mathbb{E}N(t_0)=
            \sum_{k=1}^{N+1} k\sum_{i=0}^{M-1}P(k,i,t_0).

        В стационарном режиме начальное распределение не используется, и
        метод возвращает ``0``.

        Returns:
            float: Ожидаемое число заявок в системе в начальный момент.
        """
        transient_params = self.params.transient_params
        if transient_params is None:
            return 0.0

        if not isinstance(self.params.base_params, MAPSystemParams):
            raise TypeError("Базовые параметры должны быть экземпляром MAPSystemParams")

        phase_count = self.params.base_params.sensor_count
        initial = np.asarray(
            transient_params.initial_probabilities,
            dtype=np.float64,
        ).reshape(-1, phase_count)
        initial = initial[:, self._selected_phase_indices()]
        level_weights = np.arange(initial.shape[0], dtype=np.float64)[:, np.newaxis]
        return float(np.sum(level_weights * initial))

    def _cumulative_integral(
        self,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Численно вычисляет интеграл от начального момента до каждой точки.

        Интегрирование выполняется методом трапеций по временному массиву,
        заданному для переходного режима. Первая точка интеграла равна нулю.

        Args:
            values (NDArray[np.float64]): Значения интегрируемой функции.
                Первая ось должна соответствовать времени.

        Returns:
            NDArray[np.float64]: Накопленный интеграл той же формы.

        Raises:
            ValueError: Если переходные параметры отсутствуют или размерность
                временной оси не совпадает с первой осью ``values``.
        """
        transient_params = self.params.transient_params
        if transient_params is None:
            raise ValueError("Интеграл по времени определён только в переходном режиме")

        time_array = np.asarray(transient_params.time_array, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)

        if values.shape[0] != time_array.shape[0]:
            raise ValueError(
                "Первая ось интегрируемого массива должна совпадать с временной: "
                f"{values.shape[0]} != {time_array.shape[0]}."
            )

        return np.asarray(
            cumulative_trapezoid(values, time_array, axis=0, initial=0),
            dtype=np.float64,
        )

    def calculate_input_intensity(self) -> NDArray[np.float64]:
        r"""Вычисляет нестационарную интенсивность входного потока ``lambda(t)``.

        Используется выражение

        .. math::

            \lambda(t)=\sum_{k=0}^{N+1}\mathbf p_k(t)D_1\mathbf e,

        где ``p_k(t)`` — строка вероятностей MAP-фаз на уровне ``k``.

        Returns:
            NDArray[np.float64]: Значения ``lambda(t)`` для всех временных точек.

        Raises:
            TypeError: Если объект поведения системы не содержит матрицу ``D1``.
        """
        logger.info("Вычисление нестационарной интенсивности входного потока lambda(t)")

        probabilities = self._reshape_map_probabilities()
        d1_matrix = getattr(self.system_behavior, "D1", None)

        if d1_matrix is None:
            raise TypeError("Для расчёта lambda(t) требуется матрица D1 MAP-процесса")

        d1_matrix = np.asarray(d1_matrix, dtype=np.float64)
        ones = np.ones(d1_matrix.shape[1], dtype=np.float64)

        # alpha_i(t) = sum_k P(k, i, t) — текущее распределение MAP-фаз.
        phase_probabilities = np.sum(probabilities, axis=0)

        # Для каждой исходной фазы i величина [D1 * e]_i равна интенсивности
        # поступлений, генерируемых из этой фазы.
        arrival_rate_by_phase = (d1_matrix @ ones)[self._selected_phase_indices()]
        input_intensity = np.einsum(
            "it,i->t",
            phase_probabilities,
            arrival_rate_by_phase,
        )

        if np.any(input_intensity <= 0.0):
            logger.warning(
                "Для некоторых временных точек lambda(t) <= 0; "
                "накопленный знаменатель P_uns(t) может быть нулевым "
                "в начальной точке."
            )

        logger.success("Нестационарная интенсивность lambda(t) успешно вычислена")
        return cast(NDArray[np.float64], input_intensity)

    def calculate_phase_distribution(self) -> NDArray[np.float64]:
        r"""Вычисляет распределение вероятностей MAP-фаз ``alpha_i(t)``.

        .. math::

            \alpha_i(t)=\sum_{k=0}^{N+1}P(k,i,t).

        Величина показывает, как во времени перераспределяется вклад
        отдельных датчиков во входной поток, и является множителем в
        выражении для ``lambda(t)``.

        Returns:
            NDArray[np.float64]: Массив формы ``[time_count, sensor_count]``.
        """
        logger.info("Вычисление распределения MAP-фаз alpha_i(t)")

        phase_probabilities = np.sum(self._reshape_map_probabilities(), axis=0).T

        logger.success("Распределение MAP-фаз успешно вычислено")
        return cast(NDArray[np.float64], phase_probabilities)

    def calculate_service_flow_intensity(self) -> NDArray[np.float64]:
        r"""Вычисляет интенсивность потока обслуженных заявок ``v_serv(t)``.

        .. math::

            v_{serv}(t)=\mu\sum_{k=1}^{N+1}\sum_{i=0}^{M-1}P(k,i,t).

        Прибор занят во всех состояниях, кроме пустой системы, поэтому
        фактический поток обслуживания равен ``mu``, взвешенной на
        вероятность занятости прибора. В отличие от параметра ``mu``,
        эта величина меняется во времени.

        Суммирование по уровням записано явно, а не как дополнение до
        единицы: в индивидуальном режиме вероятности выбранного датчика
        в сумме дают не единицу, а его долю во входном потоке.

        Returns:
            NDArray[np.float64]: Значения ``v_serv(t)``.
        """
        logger.info("Вычисление интенсивности потока обслуживания v_serv(t)")

        probabilities = self._reshape_map_probabilities()
        busy_probability = np.sum(probabilities[1:], axis=(0, 1))
        service_flow_intensity = self.params.base_params.mu_rate * busy_probability

        logger.success("Интенсивность потока обслуживания успешно вычислена")
        return cast(NDArray[np.float64], service_flow_intensity)

    def calculate_loss_flow_intensity(self) -> NDArray[np.float64]:
        r"""Вычисляет интенсивность потока ушедших заявок ``v_loss(t)``.

        .. math::

            v_{loss}(t)=\upsilon N_b(t)
            =\upsilon\sum_{k=1}^{N}k\sum_{i=0}^{M-1}P(k+1,i,t).

        Это формула (14) статьи. В актуальной редакции множитель ``k``
        обязателен: каждая из ``k`` заявок, находящихся в буфере, может уйти
        с интенсивностью ``nu``.

        Returns:
            NDArray[np.float64]: Значения ``v_loss(t)``. При покомпонентном
                расчёте массив имеет форму ``[time_count, sensor_count]``.

        Raises:
            TypeError: Если поведение системы не является MAP-поведением.
        """
        logger.info("Вычисление интенсивности потока ухода v_loss(t)")

        if not isinstance(self.system_behavior, MultiSensorMAPSystemBehavior):
            raise TypeError("Для расчёта v_loss(t) требуется поведение MAP-системы")

        buffer_occupancy = np.asarray(
            self.system_behavior.calculate_avg_system_length(self.probabilities),
            dtype=np.float64,
        )
        loss_flow_intensity = self.params.base_params.nu_rate * buffer_occupancy

        logger.success("Интенсивность потока ухода успешно вычислена")
        return loss_flow_intensity

    def calculate_avg_system_length(self) -> NDArray[np.float64]:
        r"""Вычисляет среднее число заявок в буфере ``N_b(t)``.

        Используется выражение

        .. math::

            N_b(t)=\sum_{k=1}^{N}k\sum_{i=0}^{M-1}P(k+1,i,t).

        Returns:
            NDArray[np.float64]: Значения ``N_b(t)``.
        """
        return self.system_behavior.calculate_avg_system_length(self.probabilities)

    def calculate_rejection_probability(self) -> NDArray[np.float64]:
        r"""Вычисляет вероятность отказа ``P_fail(t)``.

        .. math::

            P_{fail}(t)=\sum_{i=0}^{M-1}P(N+1,i,t).

        Returns:
            NDArray[np.float64]: Вероятность отказа.
        """
        logger.info("Вычисление вероятности отказа P_fail(t)")

        probabilities = self._reshape_map_probabilities()
        rejection_probability = np.sum(probabilities[-1], axis=0)

        logger.success("Вероятность отказа успешно вычислена")
        return cast(NDArray[np.float64], rejection_probability)

    def calculate_quit_probability(self) -> NDArray[np.float64]:
        r"""Вычисляет вероятность ухода нетерпеливых заявок ``P_uns(t)``.

        В переходном режиме реализована формула (16) статьи:

        .. math::

            P_{uns}(t)=
            \frac{\int_{t_0}^{t}v_{loss}(u)\,du}
            {\mathbb{E}N(t_0)+\int_{t_0}^{t}\lambda(u)\,du},

        где

        .. math::

            v_{loss}(t)=\upsilon
            \sum_{k=1}^{N}k\sum_{i=0}^{M-1}P(k+1,i,t),

        а ``lambda(t)`` вычисляется по формуле (15). Если временная сетка
        начинается не с нуля, её первая точка трактуется как ``t_0``.

        В стационарном режиме используется предельное значение отношения
        интегралов:

        .. math::

            P_{uns}=\frac{v_{loss}}{\lambda}.

        Returns:
            NDArray[np.float64]: Вероятность ухода заявки из системы.

        Raises:
            TypeError: Если поведение системы не является MAP-поведением.
        """
        logger.info("Вычисление вероятности ухода P_uns(t)")

        loss_flow_intensity = self.calculate_loss_flow_intensity()
        input_intensity = self.calculate_input_intensity()

        if self.params.transient_params is not None:
            numerator = self._cumulative_integral(loss_flow_intensity)
            denominator_base = (
                self._calculate_initial_expected_customers()
                + self._cumulative_integral(input_intensity)
            )
        else:
            # Для стационарного распределения отношение накопленных потоков
            # стремится к отношению их постоянных интенсивностей.
            numerator = loss_flow_intensity
            denominator_base = input_intensity

        # В режиме «все датчики по отдельности» числитель раскладывается по
        # датчикам, а знаменатель остаётся общим. Поэтому сумма компонент
        # P_uns,i(t) совпадает с агрегированной вероятностью ухода.
        denominator: NDArray[np.float64]
        if numerator.ndim == 2:
            denominator = denominator_base[:, np.newaxis]
        else:
            denominator = denominator_base

        quit_probability = np.full_like(
            numerator,
            np.nan,
            dtype=np.float64,
        )
        np.divide(
            numerator,
            denominator,
            out=quit_probability,
            where=~np.isclose(denominator, 0.0),
        )

        # При пустой системе в t=t0 формула даёт 0/0. Физически к этому
        # моменту ещё ни одна заявка не ушла, поэтому принимается правый
        # предел P_uns(t0)=0. Если знаменатель нулевой, а числитель нет,
        # значение остаётся NaN как признак некорректной декомпозиции.
        zero_over_zero = np.isclose(denominator, 0.0) & np.isclose(numerator, 0.0)
        quit_probability[zero_over_zero] = 0.0

        logger.success("Вероятность ухода успешно вычислена")
        return quit_probability

    def calculate_loss_probability(self) -> NDArray[np.float64]:
        r"""Вычисляет результирующую вероятность потерь ``P_loss(t)``.

        .. math::

            P_{loss}(t)=P_{fail}(t)+P_{uns}(t).

        Returns:
            NDArray[np.float64]: Вероятность потерь.
        """
        logger.info("Вычисление вероятности потерь P_loss(t)")

        rejection_probability = self.calculate_rejection_probability()
        quit_probability = self.calculate_quit_probability()

        if quit_probability.ndim == 2 and rejection_probability.ndim == 1:
            rejection_probability = rejection_probability[:, np.newaxis]

        loss_probability = rejection_probability + quit_probability

        logger.success("Вероятность потерь успешно вычислена")
        return loss_probability

    def calculate_service_probability(self) -> NDArray[np.float64]:
        r"""Вычисляет вероятность обслуживания ``P_serv(t)``.

        .. math::

            P_{serv}(t)=1-P_{loss}(t).

        Returns:
            NDArray[np.float64]: Вероятность обслуживания.
        """
        logger.info("Вычисление вероятности обслуживания P_serv(t)")

        service_probability = 1.0 - self.calculate_loss_probability()

        logger.success("Вероятность обслуживания успешно вычислена")
        return service_probability

    def calculate_throughput(self) -> NDArray[np.float64]:
        r"""Вычисляет пропускную способность ``A(t)``.

        Для переходного режима используется текущая нестационарная
        интенсивность входного MAP-потока ``lambda(t)``:

        .. math::

            A(t)=\left(1-P_{loss}(t)\right)\lambda(t).

        Returns:
            NDArray[np.float64]: Пропускная способность системы.
        """
        logger.info("Вычисление пропускной способности A(t) через lambda(t)")

        service_probability = np.asarray(
            self.calculate_service_probability(),
            dtype=np.float64,
        )
        input_intensity = self.calculate_input_intensity()

        # При покомпонентном расчёте вероятность обслуживания имеет форму
        # [time, sensor]. Суммарная lambda(t) применяется к каждому вкладу;
        # такая декомпозиция используется только для визуального анализа.
        intensity_multiplier: NDArray[np.float64]
        if service_probability.ndim == 2:
            intensity_multiplier = input_intensity[:, np.newaxis]
        else:
            intensity_multiplier = input_intensity

        throughput = service_probability * intensity_multiplier

        logger.success("Пропускная способность через lambda(t) успешно вычислена")
        return throughput

    def calculate_stability_base_metric(self) -> NDArray[np.float64]:
        r"""Возвращает характеристику ``a(t)`` для оценки устойчивости.

        Для MAP-системы это либо вероятность обслуживания
        :math:`a(t)=1-P_{loss}(t)`, либо пропускная способность
        :math:`A(t)` — в зависимости от ``stability_params.metric``.

        Returns:
            NDArray[np.float64]: Значения ``a(t)``.
        """
        if self.stability_params.metric == StabilityMetric.THROUGHPUT:
            return self.calculate_throughput()

        return self.calculate_service_probability()

    def calculate(self) -> dict[str, NDArray[np.float64]]:
        """Вычисляет вероятности состояний и производные характеристики.

        Returns:
            dict[str, NDArray[np.float64]]: Результат вычислений.
        """
        calculations: dict[str, Callable[[], NDArray[np.float64]]] = {
            "probability": lambda: self.probabilities,
            "throughput": self.calculate_throughput,
            "avg_system_length": self.calculate_avg_system_length,
            "rejection_probability": self.calculate_rejection_probability,
            "loss_probability": self.calculate_loss_probability,
            "service_probability": self.calculate_service_probability,
            "quit_probability": self.calculate_quit_probability,
            "input_intensity": self.calculate_input_intensity,
            "service_flow_intensity": self.calculate_service_flow_intensity,
            "loss_flow_intensity": self.calculate_loss_flow_intensity,
            "phase_distribution": self.calculate_phase_distribution,
            "stability_coefficient": self.calculate_stability_coefficient,
        }

        logger.info("Запуск расчёта характеристик СМО")

        result = {
            metric_name: calculation()
            for metric_name, calculation in calculations.items()
        }

        logger.success("Расчёт успешно завершён")
        return result

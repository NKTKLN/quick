"""Pydantic-схемы запросов и ответов FastAPI-бекенда.

Описывает структуру входных данных для запуска моделирования СМО
и структуру результатов, возвращаемых клиенту.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class SystemTypeName(StrEnum):
    """Тип системы массового обслуживания."""

    MULTI = "multi"
    MAP = "map"


class SystemModeName(StrEnum):
    """Режим функционирования системы."""

    TRANSIENT = "transient"
    STEADY = "steady"


class CalculationMethodName(StrEnum):
    """Метод расчёта."""

    ANALYTICAL = "analytical"
    NUMERICAL = "numerical"
    IMITATION = "imitation"


class CalculationEngineName(StrEnum):
    """Движок вычислений."""

    NUMPY = "numpy"
    MPMATH = "mpmath"
    MERGED = "merged"


class ComputationConfigSchema(BaseModel):
    """Конфигурация вычислений.

    Attributes:
        calculation_method (CalculationMethodName): Метод расчёта.
        calculation_engine (CalculationEngineName): Движок вычислений
            (используется только для аналитического метода).
        precision (int): Точность вычислений в знаках после запятой
            (для движков mpmath и merged).
        tolerance (int): Допустимая погрешность в знаках после запятой
            (для движка merged).
        disable_cache (bool): Флаг отключения кэширования вычислений.
        compute_only_last_state (bool): Флаг вычисления только последнего состояния.
    """

    calculation_method: CalculationMethodName = CalculationMethodName.ANALYTICAL
    calculation_engine: CalculationEngineName = CalculationEngineName.NUMPY
    precision: int = Field(default=50, ge=1, le=512)
    tolerance: int = Field(default=50, ge=1, le=512)
    disable_cache: bool = False
    compute_only_last_state: bool = False


class MAPParamsSchema(BaseModel):
    """Параметры MAP-процесса для систем типа MAP.

    Attributes:
        p_rate (list[list[float]]): Матрица интенсивностей обслуживания.
        q_rate (list[list[float]]): Матрица интенсивностей поступления.
        sensor_count (int): Количество сенсоров в системе.
    """

    p_rate: list[list[float]]
    q_rate: list[list[float]]
    sensor_count: int = Field(ge=1)


class TransientParamsSchema(BaseModel):
    """Параметры переходного режима.

    Attributes:
        time_array (list[float]): Массив временных точек расчёта.
        initial_probabilities (list[float]): Начальные вероятности состояний.
    """

    time_array: list[float]
    initial_probabilities: list[float]


class TimeSeriesSchema(BaseModel):
    """Временной ряд значения параметра.

    Attributes:
        times (list[float]): Массив времён измерений (неубывающий).
        values (list[float]): Массив значений в соответствующие моменты времени.
    """

    times: list[float]
    values: list[float]


class TimeSeriesParamsSchema(BaseModel):
    """Зависящие от времени параметры СМО для имитационного метода.

    Attributes:
        lambda_rate (TimeSeriesSchema | None): Временной ряд интенсивности поступления.
        mu_rate (TimeSeriesSchema | None): Временной ряд интенсивности обслуживания.
        nu_rate (TimeSeriesSchema | None): Временной ряд интенсивности ухода.
    """

    lambda_rate: TimeSeriesSchema | None = None
    mu_rate: TimeSeriesSchema | None = None
    nu_rate: TimeSeriesSchema | None = None


class SimulationParamsSchema(BaseModel):
    """Параметры имитационного моделирования.

    Attributes:
        trajectories (int): Количество траекторий Монте-Карло.
        seed (int | None): Начальное значение для ГПСЧ.
    """

    trajectories: int = Field(default=10_000, ge=1)
    seed: int | None = None


class SimulateRequest(BaseModel):
    """Запрос на запуск моделирования СМО.

    Attributes:
        system_type (SystemTypeName): Тип системы.
        system_mode (SystemModeName): Режим функционирования.
        lambda_rate (float | list[float]): Интенсивность поступления заявок (λ).
            Скаляр для MULTI, список по сенсорам для MAP.
        mu_rate (float): Интенсивность обслуживания заявок (μ).
        nu_rate (float): Интенсивность дополнительных процессов (ν).
        max_customers (int): Максимальное количество заявок в системе.
        processor_count (int | None): Количество каналов (для MULTI).
        map_params (MAPParamsSchema | None): Параметры MAP-процесса (для MAP).
        transient_params (TransientParamsSchema | None): Параметры переходного режима.
        config (ComputationConfigSchema): Конфигурация вычислений.
        simulation_params (SimulationParamsSchema | None): Параметры имитации.
        time_series_params (TimeSeriesParamsSchema | None): Параметры,
            зависящие от времени (для имитационного метода).
        per_sensor (bool): Выполнять расчёт отдельно для каждого сенсора (для MAP).
    """

    system_type: SystemTypeName
    system_mode: SystemModeName
    lambda_rate: float | list[float]
    mu_rate: float
    nu_rate: float
    max_customers: int = Field(ge=1)
    processor_count: int | None = Field(default=None, ge=1)
    map_params: MAPParamsSchema | None = None
    transient_params: TransientParamsSchema | None = None
    config: ComputationConfigSchema = Field(default_factory=ComputationConfigSchema)
    simulation_params: SimulationParamsSchema | None = None
    time_series_params: TimeSeriesParamsSchema | None = None
    per_sensor: bool = False


class SimulateResponse(BaseModel):
    """Результат моделирования СМО.

    Attributes:
        system_type (SystemTypeName): Тип системы.
        system_mode (SystemModeName): Режим функционирования.
        probabilities (list): Вероятности состояний системы.
            Для переходного режима — двумерный массив (состояние × время),
            для стационарного — одномерный массив.
        time_array (list[float] | None): Массив временных точек
            (только для переходного режима).
        metrics (dict | None): Метрики производительности системы
            (только для переходного режима).
    """

    system_type: SystemTypeName
    system_mode: SystemModeName
    probabilities: list
    time_array: list[float] | None = None
    metrics: dict | None = None


class MetaResponse(BaseModel):
    """Справочник допустимых значений перечислений API.

    Attributes:
        system_types (dict[str, str]): Типы систем и их описания.
        system_modes (dict[str, str]): Режимы функционирования и их описания.
        calculation_methods (dict[str, str]): Методы расчёта и их описания.
        calculation_engines (dict[str, str]): Движки вычислений и их описания.
    """

    system_types: dict[str, str]
    system_modes: dict[str, str]
    calculation_methods: dict[str, str]
    calculation_engines: dict[str, str]

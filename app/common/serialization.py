"""Модуль для сериализации и десериализации объектов с поддержкой mpmath и numpy.

Включает класс PickleSerializer, который обеспечивает корректное сохранение и
восстановление сложных числовых структур, таких как mpmath.matrix и numpy.ndarray
с элементами mpf, учитывая заданную точность вычислений.
"""

import pickle
from typing import Any, Optional

from mpmath import matrix, mpf, re, workdps  # type: ignore[import-untyped]
from numpy import array, ndarray

from app.common.safe_unpickler import safe_loads


class PickleSerializer:
    """Сериализатор и десериализатор для объектов с mpmath и numpy.

    Обеспечивает конвертацию сложных числовых типов в сериализуемый формат и
    обратное восстановление с учётом точности вычислений.
    """

    def _deserialize_obj(self, obj: Any, precision: Optional[int] = None) -> Any:
        """Рекурсивно десериализует объекты с учётом точности mpmath.

        Args:
            obj (Any): Десериализуемый объект.
            precision (Optional[int]): Точность вычислений mpmath (default=50).

        Returns:
            Any: Объект с восстановленными mpmath и numpy типами.
        """
        if precision is None:
            precision = 50

        with workdps(precision):

            def deserialize_nested(data: Any) -> Any:
                """Рекурсивно преобразует вложенные списки в объекты mpmath.mpf.

                Args:
                    data (Any): Входные данные, представляющие числа или списки.

                Returns:
                    Any: Объект mpf либо список объектов mpf с рекурсивной обработкой.
                """
                if isinstance(data, list):
                    return [deserialize_nested(item) for item in data]
                return mpf(data)

            if isinstance(obj, dict):
                obj_type = obj.get("__type__")
                if obj_type == "mpmath.matrix":
                    data = obj["data"]
                    return matrix(deserialize_nested(data))
                elif obj_type == "NDArray[mpmath.mpf]":
                    data = obj["data"]
                    return array(deserialize_nested(data), dtype=object)
            if isinstance(obj, list):
                return [self._deserialize_obj(item, precision) for item in obj]
            if isinstance(obj, dict):
                return {k: self._deserialize_obj(v, precision) for k, v in obj.items()}
            return obj

    def _serialize_obj(self, obj: Any) -> Any:
        """Рекурсивно сериализует объекты mpmath и numpy в сериализуемый формат.

        Args:
            obj (Any): Объект для сериализации.

        Returns:
            Any: Структура, пригодная для сериализации pickle.
        """

        def serialize_nested(data: Any) -> Any:
            """Рекурсивно сериализует вложенные списки, преобразуя элементы в строки.

            Args:
                data (Any): Входные данные - числа или вложенные списки.

            Returns:
                Any: Сериализованная структура с элементами в виде строк.
            """
            if isinstance(data, list):
                return [serialize_nested(item) for item in data]
            return str(re(data))

        if isinstance(obj, matrix):
            return {"__type__": "mpmath.matrix", "data": serialize_nested(obj.tolist())}

        if isinstance(obj, ndarray):
            if obj.dtype != object:
                return obj
            return {
                "__type__": "NDArray[mpmath.mpf]",
                "data": serialize_nested(obj.tolist()),
            }

        if isinstance(obj, list):
            return [self._serialize_obj(item) for item in obj]

        if isinstance(obj, tuple):
            return tuple(self._serialize_obj(item) for item in obj)

        if isinstance(obj, dict):
            return {k: self._serialize_obj(v) for k, v in obj.items()}

        return obj

    def dump_key_to_pickle(self, data: dict[str, Any]) -> bytes:
        """Сериализует словарь с преобразованием в pickle байты.

        Args:
            data (dict[str, Any]): Словарь для сериализации.

        Returns:
            bytes: Сериализованные байты pickle.
        """
        serialized = {k: self._serialize_obj(v) for k, v in data.items()}
        return pickle.dumps(serialized)

    def dump_to_pickle(self, obj: Any) -> bytes:
        """Сериализует произвольный объект в pickle байты.

        Args:
            obj (Any): Объект для сериализации.

        Returns:
            bytes: Сериализованные байты pickle.
        """
        serialized = self._serialize_obj(obj)
        return pickle.dumps(serialized)

    def load_result_from_pickle(
        self, obj: bytes, precision: Optional[int] = None
    ) -> Any:
        """Десериализует объект из pickle байт с восстановлением типов и точности.

        Args:
            obj (bytes): Байты pickle для загрузки.
            precision (Optional[int]): Точность mpmath при десериализации.

        Returns:
            Any: Восстановленный объект с корректными типами.
        """
        loaded = safe_loads(obj)
        return self._deserialize_obj(loaded, precision)

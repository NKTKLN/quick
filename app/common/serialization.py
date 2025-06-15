from pickle import dumps, loads
from typing import Any, Optional

from mpmath import matrix, mpf, re, workdps
from numpy import array, ndarray


class PickleSerializer:
    def _deserialize_obj(self, obj: Any, precision: Optional[int] = None) -> Any:
        if precision is None:
            precision = 50

        with workdps(precision):

            def deserialize_nested(data):
                if isinstance(data, list):
                    return [deserialize_nested(item) for item in data]
                return mpf(data)

            if isinstance(obj, dict):
                if obj.get("__type__") == "mpmath.matrix":
                    data = obj["data"]
                    deserialized_data = deserialize_nested(data)
                    return matrix(deserialized_data)
                elif obj.get("__type__") == "NDArray[mpmath.mpf]":
                    data = obj["data"]
                    deserialized_data = deserialize_nested(data)
                    return array(deserialized_data, dtype=object)
            elif isinstance(obj, list):
                return [self._deserialize_obj(item, precision) for item in obj]
            elif isinstance(obj, dict):
                return {k: self._deserialize_obj(v, precision) for k, v in obj.items()}
            return obj

    def _serialize_obj(self, obj: Any) -> Any:
        def serialize_nested(data):
            if isinstance(data, list):
                return [serialize_nested(item) for item in data]
            return str(re(data))

        if isinstance(obj, matrix):
            data = obj.tolist()
            return {"__type__": "mpmath.matrix", "data": serialize_nested(data)}

        if isinstance(obj, ndarray):
            if obj.dtype != object:
                return obj
            data = obj.tolist()
            return {"__type__": "NDArray[mpmath.mpf]", "data": serialize_nested(data)}

        if isinstance(obj, list):
            return [self._serialize_obj(item) for item in obj]

        if isinstance(obj, tuple):
            return tuple(self._serialize_obj(item) for item in obj)

        if isinstance(obj, dict):
            return {k: self._serialize_obj(v) for k, v in obj.items()}

        return obj

    def dump_key_to_pickle(self, data: dict[str, Any]) -> bytes:
        serialized_data = {k: self._serialize_obj(v) for k, v in data.items()}
        return dumps(serialized_data)

    def dump_to_pickle(self, obj: Any) -> bytes:
        serialized_obj = self._serialize_obj(obj)
        return dumps(serialized_obj)

    def load_result_from_pickle(
        self, obj: bytes, precision: Optional[int] = None
    ) -> Any:
        loaded = loads(obj)
        return self._deserialize_obj(loaded, precision)

"""Модуль безопасной десериализации данных с использованием ограниченного набора типов.

Предоставляет класс SafeUnpickler и функцию safe_loads, которые ограничивают
десериализацию только допустимыми типами данных, предотвращая выполнение потенциально
вредоносного кода при работе с pickle. Поддерживаются только базовые типы Python.
"""

import io
import pickle
from typing import Any

from loguru import logger


class SafeUnpickler(pickle.Unpickler):
    """Безопасный unpickler с ограничением на допустимые классы.

    Разрешает только загрузку объектов из белого списка допустимых типов для
    предотвращения выполнения произвольного кода при десериализации.
    """

    # Разрешённые классы в формате (модуль, имя)
    ALLOWED_CLASSES = {
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "list"),
        ("builtins", "dict"),
        ("builtins", "tuple"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "str"),
        ("builtins", "bool"),
        ("builtins", "complex"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "dtype"),
        ("mpmath.ctx_mp_python", "mpf"),
        ("mpmath.ctx_mp_python", "mpc"),
    }

    def find_class(self, module: str, name: str) -> Any:
        """Переопределённый метод поиска классов с ограничением по белому списку.

        Args:
            module (str): Имя модуля.
            name (str): Имя класса.

        Returns:
            Any: Разрешённый класс.

        Raises:
            pickle.UnpicklingError: Если класс не разрешён.
        """
        if (module, name) not in self.ALLOWED_CLASSES:
            logger.error(f"Попытка загрузить запрещённый класс: {module}.{name}")
            raise pickle.UnpicklingError(
                f"Попытка загрузить запрещённый класс: {module}.{name}"
            )

        logger.debug(f"Разрешён класс для загрузки: {module}.{name}")
        return super().find_class(module, name)


def safe_loads(data: bytes) -> object:
    """Безопасно загружает объект из pickle-байтов с использованием SafeUnpickler.

    Args:
        data (bytes): Сериализованные pickle-данные.

    Returns:
        object: Десериализованный объект из безопасных классов.

    Raises:
        pickle.UnpicklingError: При попытке загрузки запрещённого типа.
    """
    logger.debug("Начало безопасной десериализации данных")
    result = SafeUnpickler(io.BytesIO(data)).load()
    logger.debug("Безопасная десериализация завершена успешно")
    return result

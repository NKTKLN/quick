"""Точка входа FastAPI-бекенда программного комплекса QUICK.

Создаёт приложение FastAPI, инициализирует логирование и кэш DuckDB,
подключает маршруты API и эндпоинт проверки состояния.

Запуск:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from quick.db import DuckDBClient
from quick.utils import setup_logger

from .routes import router


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Инициализирует ресурсы приложения при старте.

    Args:
        _app (FastAPI): Экземпляр приложения (не используется).

    Yields:
        None: Управление приложению на время его работы.
    """
    setup_logger()
    _ = DuckDBClient()
    yield


app = FastAPI(
    title="QUICK API",
    description=(
        "REST API программного комплекса для моделирования и анализа "
        "медицинских информационно-измерительных систем, представляемых "
        "в виде СМО с нетерпеливыми заявками."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    """Проверка состояния сервиса.

    Returns:
        dict[str, str]: Статус сервиса.
    """
    return {"status": "ok"}

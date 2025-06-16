# ===== Стадия 1: Сборщик =====
FROM python:3.13-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.0.0

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl

# Установка Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

RUN poetry self add poetry-plugin-export

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

# Экспортируем зависимости в requirements.txt для pip, собираем wheel
RUN poetry export --without-hashes --format=requirements.txt > requirements.txt && \
    pip wheel --no-deps --wheel-dir /wheels -r requirements.txt

RUN apt-get purge -y build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

# ===== Стадия 2: Финальная =====
FROM python:3.13-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DUCKDB_PATH=cache/data.duck_db

# Создаем непривилегированного пользователя
RUN groupadd -g 10000 shrimp && \
    useradd -m -u 10000 -g shrimp shrimp

USER shrimp
WORKDIR /app

ENV PATH="/home/shrimp/.local/bin:${PATH}"

# Копируем зависимости и requirements из builder
COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements.txt ./requirements.txt

# Устанавливаем зависимости из wheel
RUN pip install --user --no-index --find-links=/wheels -r requirements.txt

COPY --chown=shrimp:shrimp . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py"]
CMD ["--server.port=8501", "--server.address=0.0.0.0"]

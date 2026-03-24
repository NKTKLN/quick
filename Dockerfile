# ===== Stage 1: Assembler =====
FROM python:3.13-slim AS builder

# Базовое окружение Python + настройки uv + виртуальное окружение в /opt/venv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Установка бинарника uv (быстрый резолвер/установщик зависимостей)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Сначала копируем только файлы зависимостей (лучше кешируются слои)
COPY pyproject.toml uv.lock ./

# Создаём виртуальное окружение + устанавливаем прод-зависимости (по lock-файлу, без dev)
RUN uv venv /opt/venv \
    && uv sync --frozen --no-dev --group web --no-install-project

# Копируем исходный код после зависимостей для эффективного кеширования
COPY . .

# Устанавливаем пакет в виртуальное окружение
RUN uv sync --frozen --no-dev --group web

# ===== Stage 2: Final =====
FROM python:3.13-slim AS final

# Окружение Python для рантайма; используем заранее собранное виртуальное окружение
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Зависимости только для выполнения (чтобы образ был меньше)
RUN apt-get update && apt-get install --no-install-recommends -y \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Создаём непривилегированного пользователя (фиксированный UID/GID для удобства в Kubernetes)
RUN groupadd -g 10000 shrimp && \
    useradd -m -u 10000 -g shrimp shrimp

WORKDIR /app

# Переносим виртуальное окружение и приложение из этапа сборки
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Исправляем владельца, чтобы непривилегированный пользователь мог читать и запускать всё
RUN chown -R shrimp:shrimp /app /opt/venv

USER shrimp

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Запуск модуля как entrypoint
ENTRYPOINT ["streamlit", "run", "web/app.py"]
CMD ["--server.port=8501", "--server.address=0.0.0.0"]

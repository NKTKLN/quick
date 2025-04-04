# 📊 Impatient Queue System

**Impatient Queue System** — это приложение для моделирования математической системы массового обслуживания (СМО) с нетерпеливыми заявками в переходном режиме.

## 📦 Зависимости

- [Python](https://www.python.org/downloads/).
- [Poetry](https://python-poetry.org/docs/#installation).
- [Docker](https://docs.docker.com/get-docker/).

## 📂 Структура проекта

- **app**: Основной код приложения.
  - main.py: Точка входа в приложение.
  - system.py: Логика математической модели.
  - plot.py: Построение графиков.
  - logger.py: Настройка логирования.
  - args_parser.py: Парсинг аргументов командной строки.
- **Dockerfile**: Конфигурация для сборки Docker-образа.
- **`.dockerignore`**: Исключения для файлов, не попадающих в Docker-образ.
- **`pyproject.toml`**: Зависимости проекта.

## 🛠️ Установка и запуск

### 💻 Локальный запуск

1. Убедитесь, что у вас установлен Python 3.12 или выше.
2. Установите зависимости:
   ```bash
   poetry install --no-root
   ```
3. Запустите приложение:
   ```bash
   python -m app.main
   ```

### 🐳 Запуск в Docker

1. Соберите Docker-образ:
   ```bash
   docker build -t impatient-queue-system .
   ```
2. Запустите контейнер:
   ```bash
   docker run --rm -it impatient-queue-system
   ```

## 📝 Аргументы командной строки

- `--disable-logging`: Отключить логирование.
- `--log-level`: Уровень логирования (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
- `--log-file`: Имя файла для записи логов (по умолчанию логи выводятся в консоль).

Пример запуска с аргументами:
```bash
python -m app.main --log-level DEBUG --log-file logs.txt
```

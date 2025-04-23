# 📊 Impatient Queue System

**Impatient Queue System** — это приложение для моделирования систем массового обслуживания (СМО) с нетерпеливыми заявками. Оно позволяет анализировать вероятностные характеристики системы, включая интенсивности потока заявок, обслуживания и ухода, а также визуализировать результаты в виде графиков.

## 📦 Зависимости

- [Python](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/#installation)
- [Docker](https://docs.docker.com/get-docker/)

## 📂 Структура проекта

- **app**: Основной код приложения.
  - `main.py`: Точка входа в приложение.
  - `impatient_queue_system.py`: Логика математической модели.
  - `plot.py`: Построение графиков.
  - `logger.py`: Настройка логирования.
  - `args_parser.py`: Парсинг аргументов командной строки.
  - `parameters.py`: Определение параметров системы.
  - `matrix_generators.py`: Генерация матриц для расчетов.
  - `probability_solver.py`: Вычисление вероятностей.
  - `throughput_queue_system.py`: Анализ пропускной способности.
- **Dockerfile**: Конфигурация для сборки Docker-образа.
- **.dockerignore**: Исключения для файлов, не попадающих в Docker-образ.
- **pyproject.toml**: Зависимости проекта.
- **Taskfile.yml**: Сценарии для автоматизации задач.

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

### ⚙️ Использование Taskfile

Для упрощения работы с проектом можно использовать `Taskfile.yml`. Убедитесь, что у вас установлен [Task](https://taskfile.dev/).

1. Установите зависимости:

   ```bash
   task install
   ```

2. Запустите приложение:

   ```bash
   task run
   ```

## 📝 Аргументы командной строки

- `--disable-logging`: Отключить логирование.
- `--log-level`: Уровень логирования (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
- `--log-file`: Имя файла для записи логов (по умолчанию логи выводятся в консоль).
- `--save-plot`: Сохранить график вероятностей в формате PNG.

Пример запуска с аргументами:

```bash
python -m app.main --log-level DEBUG --log-file logs.txt --save-plot
```

## 📜 Лицензия

Этот проект распространяется под лицензией MIT. Подробнее см. в файле [LICENSE](./LICENSE).

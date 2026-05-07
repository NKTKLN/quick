"""Содержит настройки визуализации метрик системы для построения графиков.

Константа PLOT_SETTINGS задаёт параметры отображения для каждой метрики:
    - заголовок графика;
    - подпись оси Y;
    - цвета линий.

Используется при визуализации результатов моделирования, чтобы унифицировать
оформление графиков для различных характеристик СМО.
"""

PLOT_SETTINGS = {
    "throughput": {
        "title_text": "Пропускная способность системы",
        "yaxis_title": "Заявки/ед. времени",
        "line_colors": [
            "#1f77b4",
            "#4fa3d9",
            "#6baed6",
            "#9ecae1",
            "#c6dbef",
        ],
    },
    "avg_buffer_length": {
        "title_text": "Среднее число заявок в буфере",
        "yaxis_title": "Каналы",
        "line_colors": [
            "#ff7f0e",
            "#ff9f4a",
            "#ffb870",
            "#ffd1a3",
            "#ffe2c6",
        ],
    },
    "avg_system_length": {
        "title_text": "Среднее число заявок в системе",
        "yaxis_title": "Заявки",
        "line_colors": [
            "#9467bd",
            "#a984d6",
            "#c2a5e2",
            "#d4bff0",
            "#e7d9fa",
        ],
    },
    "absolute_throughput": {
        "title_text": "Абсолютная пропускная способность",
        "yaxis_title": "Заявки/ед. времени",
        "line_colors": [
            "#2ca02c",
            "#4cc04c",
            "#74d674",
            "#a1e3a1",
            "#c7f0c7",
        ],
    },
    "relative_throughput": {
        "title_text": "Относительная пропускная способность",
        "yaxis_title": "Доля от входящего потока",
        "line_colors": [
            "#d62728",
            "#e25555",
            "#ec7d7d",
            "#f3a7a7",
            "#f9d0d0",
        ],
    },
    "rejection_probability": {
        "title_text": "Вероятность отказа",
        "yaxis_title": "Вероятность",
        "line_colors": [
            "#8c564b",
            "#a06d63",
            "#b78a82",
            "#cfaaa5",
            "#e5cbc8",
        ],
    },
    "loss_probability": {
        "title_text": "Вероятность потери пакетов в момент времени",
        "yaxis_title": "Вероятность",
        "line_colors": [
            "#17becf",
            "#46cfdd",
            "#72dce7",
            "#9de9ef",
            "#c9f5f7",
        ],
    },
    "service_probability": {
        "title_text": "Вероятность обслуживания заявок",
        "yaxis_title": "Вероятность",
        "line_colors": [
            "#e377c2",
            "#ea95d0",
            "#f0b2dd",
            "#f6d0ea",
            "#fbe8f5",
        ],
    },
    "quit_probability": {
        "title_text": "Вероятность ухода заявки из системы",
        "yaxis_title": "Вероятность",
        "line_colors": [
            "#7f7f7f",
            "#9a9a9a",
            "#b5b5b5",
            "#d1d1d1",
            "#ebebeb",
        ],
    },
}

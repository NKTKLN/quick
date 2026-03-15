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
        "line_colors": ["#1f77b4"],
    },
    "avg_buffer_length": {
        "title_text": "Среднее число заявок в буфере",
        "yaxis_title": "Каналы",
        "line_colors": ["#ff7f0e"],
    },
    "avg_system_length": {
        "title_text": "Среднее число заявок в системе",
        "yaxis_title": "Заявки",
        "line_colors": ["#9467bd"],
    },
    "absolute_throughput": {
        "title_text": "Абсолютная пропускная способность",
        "yaxis_title": "Заявки/ед. времени",
        "line_colors": ["#2ca02c"],
    },
    "relative_throughput": {
        "title_text": "Относительная пропускная способность",
        "yaxis_title": "Доля от входящего потока",
        "line_colors": ["#d62728"],
    },
    "rejection_probability": {
        "title_text": "Вероятность отказа",
        "yaxis_title": "Вероятность",
        "line_colors": ["#8c564b"],
    },
    "loss_probability": {
        "title_text": "Вероятность потери пакетов в момент времени",
        "yaxis_title": "Вероятность",
        "line_colors": ["#17becf"],
    },
    "service_probability": {
        "title_text": "Вероятность обслуживания заявок",
        "yaxis_title": "Вероятность",
        "line_colors": ["#e377c2"],
    },
    "quit_probability": {
        "title_text": "Вероятность ухода заявки из системы",
        "yaxis_title": "Вероятность",
        "line_colors": ["#7f7f7f"],
    },
}

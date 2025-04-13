import matplotlib.pyplot as plt
import numpy as np


def plot_probabilities(p_matrix: np.ndarray, time_array: np.ndarray) -> None:
    """
    Строит график вероятностей для каждой строки матрицы p и их суммы.

    :param p_matrix: матрица вероятностей размерности (NxT), где N - количество состояний, T - количество временных точек
    :param time_array: массив времени, соответствующий временным точкам T
    """
    plt.figure(figsize=(12, 8))
    
    num_states = p_matrix.shape[0]  # Количество состояний (строк в p_matrix)

    d = np.zeros(num_states)

    # Построение графиков для каждой строки матрицы p
    for i in range(num_states):
        if p_matrix[i].max() > 1:
            d[i] = p_matrix[i].max()
            plt.plot(time_array, p_matrix[i], label=f'p{i + 1}', color="red", linewidth=2)
            continue
        if p_matrix[i].min() < 0:
            d[i] = p_matrix[i].min()
            plt.plot(time_array, p_matrix[i], label=f'p{i + 1}', color="blue", linewidth=2)
            continue
        plt.plot(time_array, p_matrix[i], label=f'p{i + 1}', linewidth=2)

    print(d)
    print(d.argmax())

    # Построение графика суммы всех вероятностей
    plt.plot(time_array, p_matrix.sum(axis=0), label='Total', color='black', linewidth=2, linestyle='--')

    # Настройка графика
    plt.title('Plot of Probabilities and Total', fontsize=16, fontweight='bold')
    plt.xlabel('Time (t)', fontsize=14)
    plt.ylabel('Values', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

    # Отображение графика
    plt.show()

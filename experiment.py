"""
experiment.py
Запуск экспериментов, тестирование и визуализация:
- Тестовые функции (Sphere, Rastrigin, Rosenbrock, Ackley)
- Запуск одиночных и множественных экспериментов
- Оптимизатор параметров ParameterOptimizer
- Анализ результатов
- Поддержка параллельных вычислений
"""

import numpy as np
import multiprocessing as mp
from typing import Callable, Dict, List, Tuple, Optional
from enum import Enum

import matplotlib.pyplot as plt


from config import ES_HHA_Config
from core import ES_HHA
from run_pv_extraction import plot_operator_usage, plot_fitness_convergence, plot_pool_usage_dynamics

# ТЕСТОВЫЕ ФУНКЦИИ

def sphere_function(x: np.ndarray) -> float:
    return np.sum(x ** 2)


def rastrigin_function(x: np.ndarray) -> float:
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock_function(x: np.ndarray) -> float:
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def ackley_function(x: np.ndarray) -> float:
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / len(x))) - \
        np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.e


def get_test_function(func_name: str, dimensions: int):
    """Возвращает функцию, границы и глобальный оптимум заданной размерности"""
    if func_name == "sphere":
        func = sphere_function
        bounds = (-100, 100)
        optimum = np.zeros(dimensions)
    elif func_name == "rastrigin":
        func = rastrigin_function
        bounds = (-5.12, 5.12)
        optimum = np.zeros(dimensions)
    elif func_name == "rosenbrock":
        func = rosenbrock_function
        bounds = (-2.048, 2.048)
        optimum = np.ones(dimensions)
    elif func_name == "ackley":
        func = ackley_function
        bounds = (-32, 32)
        optimum = np.zeros(dimensions)
    else:
        raise ValueError(f"Unknown function: {func_name}")
    return func, bounds[0], bounds[1], optimum



# ТОЧКА ВХОДА


if __name__ == "__main__":

    print("=" * 60)
    print("ES-HHA: Evolutionary Status Guided Hyper-Heuristic Algorithm")
    print("=" * 60)
    print(f"CPU cores available: {mp.cpu_count()}")
    print()

    # настройка эксперимента

    # Выбор тестовой функции
    TEST_FUNCTION = "sphere"  # sphere, rastrigin, rosenbrock, ackley

    # Основные параметры оптимизации
    POPULATION_SIZE = 500
    DIMENSIONS = 200
    MAX_FES = 500000
    USE_PARALLEL = False
    N_WORKERS = 4

    # тестовые
    objective_func, lb, ub, optimum = get_test_function(TEST_FUNCTION, DIMENSIONS)

    config = ES_HHA_Config(
        population_size=POPULATION_SIZE,
        dimensions=DIMENSIONS,
        max_FEs=MAX_FES,
        use_parallel=USE_PARALLEL,
        n_workers=N_WORKERS,
        lb_init=lb,
        ub_init=ub,
        lb_opt=lb,
        ub_opt=ub,
        test_function_name=TEST_FUNCTION,
        global_optimum=optimum,
        verbose=True,
        log_interval=5
    )


    print(f"\nFunction: {TEST_FUNCTION}")
    print(f"Population: {POPULATION_SIZE}, Dims: {DIMENSIONS}, MaxFEs: {MAX_FES}")
    print(f"Parallel: {'ON' if USE_PARALLEL else 'OFF'}, workers: {N_WORKERS}")
    print("\nStarting optimization...")

    optimizer = ES_HHA(objective_func, config)
    results = optimizer.optimize()

    if True:
        if results.get('fitness_history'):
            best_iter = np.argmin(results['fitness_history'])
            best_val = results['fitness_history'][best_iter]
            plot_fitness_convergence(results['fitness_history'], best_val, best_iter)
        if results.get('llh_usage') and results.get('pool_usage'):
            plot_operator_usage(results['llh_usage'], results['pool_usage'], title=f"{TEST_FUNCTION} Operator Usage")
        if results.get('pool_usage_history'):
            plot_pool_usage_dynamics(results['pool_usage_history'], title=f"{TEST_FUNCTION} Pool Usage Dynamics")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best fitness: {results['best_fitness']:.6e}")
    print(f"Execution time: {results['execution_time']:.2f} sec")
    print(f"Total iterations: {results['iterations']}")
    if results['distance_to_optimum_history']:
        print(f"Final distance to optimum: {results['distance_to_optimum_history'][-1]:.6f}")


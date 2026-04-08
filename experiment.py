import numpy as np
import time
import json
import os
from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import asdict
from enum import Enum

from config import ES_HHA_Config, ParameterChromosome
from core import ES_HHA

def sphere_function(x: np.ndarray) -> float:
    return np.sum(x ** 2)


def rastrigin_function(x: np.ndarray) -> float:
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock_function(x: np.ndarray) -> float:
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def ackley_function(x: np.ndarray) -> float:
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / len(x))) - \
        np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.e


def get_test_function(func_name: str):
    if func_name == "sphere":
        return sphere_function, -100.0, 100.0, np.zeros(10)
    elif func_name == "rastrigin":
        return rastrigin_function, -5.12, 5.12, np.zeros(10)
    elif func_name == "rosenbrock":
        return rosenbrock_function, -2.048, 2.048, np.ones(10)
    elif func_name == "ackley":
        return ackley_function, -32.0, 32.0, np.zeros(10)
    else:
        raise ValueError(f"Unknown function: {func_name}")




if __name__ == "__main__":

    print("=" * 60)
    print("ES-HHA: Evolutionary Status Guided Hyper-Heuristic Algorithm")
    print("=" * 60)
    print(f"CPU cores available: {mp.cpu_count()}")
    print()


    TEST_FUNCTION = "rastrigin"  


    POPULATION_SIZE = 200
    DIMENSIONS = 10
    MAX_FES = 100000
    USE_PARALLEL = True
    N_WORKERS = mp.cpu_count()

    objective_func, lb, ub, optimum = get_test_function(TEST_FUNCTION)

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

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best fitness: {results['best_fitness']:.6e}")
    print(f"Execution time: {results['execution_time']:.2f} sec")
    print(f"Total iterations: {results['iterations']}")
    if results['distance_to_optimum_history']:
        print(f"Final distance to optimum: {results['distance_to_optimum_history'][-1]:.6f}")

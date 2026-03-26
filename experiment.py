"""
experiment.py
Запуск экспериментов, тестирование и визуализация:
- Тестовые функции (Sphere, Rastrigin, Rosenbrock, Ackley)
- Запуск одиночных и множественных экспериментов
- Оптимизатор параметров ParameterOptimizer
- Анализ результатов
"""

import numpy as np
import time
import json
import os
from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import asdict
from enum import Enum

from config import ES_HHA_Config, ParameterChromosome
from core import ES_HHA


# ============================================================================
# ТЕСТОВЫЕ ФУНКЦИИ
# ============================================================================

class TestFunction(Enum):
    SPHERE = "sphere"
    RASTRIGIN = "rastrigin"
    ROSENBROCK = "rosenbrock"
    ACKLEY = "ackley"


def get_test_function(func_name: str) -> Tuple[Callable, np.ndarray, float, Tuple[float, float]]:
    """
    Получение тестовой функции по имени

    Возвращает:
        - функция оптимизации
        - глобальный оптимум (координаты)
        - значение в оптимуме
        - границы поиска
    """
    test_functions = {
        "sphere": {
            "func": lambda x: np.sum(x ** 2),
            "optimum": np.zeros(10),
            "optimum_value": 0.0,
            "bounds": (-100, 100),
            "description": "Сфера (простая, унимодальная)"
        },
        "rastrigin": {
            "func": lambda x: 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)),
            "optimum": np.zeros(10),
            "optimum_value": 0.0,
            "bounds": (-5.12, 5.12),
            "description": "Растригин (сложная, мультимодальная)"
        },
        "rosenbrock": {
            "func": lambda x: np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2),
            "optimum": np.ones(10),
            "optimum_value": 0.0,
            "bounds": (-2.048, 2.048),
            "description": "Розенброк (овражная)"
        },
        "ackley": {
            "func": lambda x: -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / len(x))) -
                              np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.e,
            "optimum": np.zeros(10),
            "optimum_value": 0.0,
            "bounds": (-32, 32),
            "description": "Экли (много локальных минимумов)"
        }
    }

    if func_name not in test_functions:
        raise ValueError(f"Неизвестная функция: {func_name}. Доступны: {list(test_functions.keys())}")

    func_info = test_functions[func_name]
    return (func_info["func"],
            func_info["optimum"],
            func_info["optimum_value"],
            func_info["bounds"])


def run_experiment(config: ES_HHA_Config) -> dict:
    """Запуск одного эксперимента"""
    test_func, optimum, opt_value, bounds = get_test_function(config.test_function_name)

    config.global_optimum = optimum
    config.lb_init = bounds[0]
    config.ub_init = bounds[1]
    config.lb_opt = bounds[0]
    config.ub_opt = bounds[1]

    optimizer = ES_HHA(test_func, config)
    results = optimizer.optimize()

    return results


def run_multiple_experiments(config: ES_HHA_Config, n_runs: int = 30,
                             save_results: bool = True) -> Tuple[Dict, List[Dict]]:
    """
    Запуск множественных экспериментов для статистического анализа

    Аргументы:
        config: конфигурация алгоритма
        n_runs: количество запусков
        save_results: сохранять ли результаты в файл

    Возвращает:
        статистика и список всех результатов
    """
    print(f"\n{'=' * 60}")
    print(f"RUNNING {n_runs} EXPERIMENTS FOR {config.test_function_name.upper()}")
    print(f"{'=' * 60}")

    all_results = []
    best_fitnesses = []

    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")

        np.random.seed(run + 42)

        results = run_experiment(config)
        all_results.append(results)
        best_fitnesses.append(results['best_fitness'])

        print(f"Run {run + 1} best fitness: {results['best_fitness']:.6e}")

    best_fitnesses = np.array(best_fitnesses)
    stats = {
        'mean': float(np.mean(best_fitnesses)),
        'std': float(np.std(best_fitnesses)),
        'min': float(np.min(best_fitnesses)),
        'max': float(np.max(best_fitnesses)),
        'median': float(np.median(best_fitnesses)),
        'q25': float(np.percentile(best_fitnesses, 25)),
        'q75': float(np.percentile(best_fitnesses, 75))
    }

    print(f"\n{'=' * 60}")
    print(f"STATISTICS OVER {n_runs} RUNS")
    print(f"{'=' * 60}")
    print(f"Mean fitness: {stats['mean']:.6e}")
    print(f"Std fitness: {stats['std']:.6e}")
    print(f"Min fitness: {stats['min']:.6e}")
    print(f"Max fitness: {stats['max']:.6e}")
    print(f"Median fitness: {stats['median']:.6e}")
    print(f"Q25: {stats['q25']:.6e}")
    print(f"Q75: {stats['q75']:.6e}")

    if save_results:
        _save_experiment_results(config, all_results, stats, n_runs)

    return stats, all_results


def _save_experiment_results(config: ES_HHA_Config, all_results: List[Dict],
                             stats: Dict, n_runs: int):
    """Сохранение результатов экспериментов"""
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # Сохранение конфигурации
    config_filename = f"{results_dir}/config_{config.test_function_name}_{timestamp}.json"
    config.save_to_file(config_filename)

    def convert_to_serializable(obj):
        if obj is None:
            return None
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(key): convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    # Сохранение подробных результатов
    results_summary = {
        'config': asdict(config),
        'n_runs': n_runs,
        'statistics': stats,
        'all_best_fitnesses': [float(f) for f in [r['best_fitness'] for r in all_results]],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'runs_details': []
    }

    for i, res in enumerate(all_results):
        run_info = {
            'run': i + 1,
            'best_fitness': float(res['best_fitness']),
            'total_FEs': int(res['total_FEs']),
            'iterations': int(res['iterations']),
            'execution_time': float(res['execution_time']),
            'final_distance_to_optimum': float(res['distance_to_optimum_history'][-1]) if res[
                'distance_to_optimum_history'] else None,
            'llh_usage': {str(k): int(v) for k, v in res['llh_usage'].items()},
            'pool_usage': {str(k): int(v) for k, v in res['pool_usage'].items()}
        }
        results_summary['runs_details'].append(run_info)

    results_summary = convert_to_serializable(results_summary)

    results_filename = f"{results_dir}/results_{config.test_function_name}_{timestamp}.json"
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    # Сохранение CSV
    csv_filename = f"{results_dir}/results_{config.test_function_name}_{timestamp}.csv"
    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write("run,best_fitness,distance_to_optimum,iterations,execution_time\n")
        for i, res in enumerate(all_results):
            distance = res['distance_to_optimum_history'][-1] if res['distance_to_optimum_history'] else 0
            f.write(
                f"{i + 1},{res['best_fitness']:.6e},{distance:.6f},{res['iterations']},{res['execution_time']:.2f}\n")

    print(f"\n✅ Results saved to: {results_filename}")
    print(f"✅ Config saved to: {config_filename}")
    print(f"✅ CSV results saved to: {csv_filename}")


# ============================================================================
# ОПТИМИЗАТОР ПАРАМЕТРОВ (ГЕНЕТИЧЕСКИЙ АЛГОРИТМ)
# ============================================================================

class ParameterOptimizer:
    """
    Оптимизатор параметров ES-HHA с использованием генетического алгоритма

    Позволяет автоматически находить оптимальные параметры конфигурации
    """

    def __init__(self,
                 objective_function: Callable,
                 base_config: ES_HHA_Config,
                 population_size: int = 20,
                 generations: int = 10,
                 n_runs_per_evaluation: int = 3):

        self.objective_function = objective_function
        self.base_config = base_config
        self.population_size = population_size
        self.generations = generations
        self.n_runs_per_evaluation = n_runs_per_evaluation

        self.population = []
        self.fitness_history = []
        self.best_chromosome = None
        self.best_fitness = float('inf')

    def evaluate_chromosome(self, chromosome: ParameterChromosome) -> float:
        """Оценка качества хромосомы (параметров)"""
        config = ES_HHA_Config(
            population_size=self.base_config.population_size,
            dimensions=self.base_config.dimensions,
            max_FEs=self.base_config.max_FEs,

            w1=chromosome.w1,
            fdc_threshold=chromosome.fdc_threshold,
            pd_threshold=chromosome.diversity_threshold,
            F_exploitation=chromosome.F_exploitation,
            F_exploration=chromosome.F_exploration,
            R_exploitation=chromosome.R_exploitation,
            R_exploration=chromosome.R_exploration,
            Cr_binomial=chromosome.Cr_binomial,
            Cr_exponential=chromosome.Cr_exponential,
            R_adaptation_rate=chromosome.R_adaptation_rate,
            p_best=chromosome.p_best,

            shake_intensity=chromosome.shake_intensity,
            shake_threshold_improvements=chromosome.shake_threshold_improvements,
            shake_threshold_fdc=chromosome.shake_threshold_fdc,
            shake_threshold_pd=chromosome.shake_threshold_pd,
            late_stage_threshold=chromosome.late_stage_threshold,
            late_stage_diversity=chromosome.late_stage_diversity,

            lb_init=self.base_config.lb_init,
            ub_init=self.base_config.ub_init,
            lb_opt=self.base_config.lb_opt,
            ub_opt=self.base_config.ub_opt,
            constraint_method=self.base_config.constraint_method,
            constraint_steepness=self.base_config.constraint_steepness,
            verbose=False,
            detailed_log=False,
            test_function_name=self.base_config.test_function_name,
            global_optimum=self.base_config.global_optimum
        )

        fitnesses = []
        for run in range(self.n_runs_per_evaluation):
            np.random.seed(run + 100)
            optimizer = ES_HHA(self.objective_function, config)
            results = optimizer.optimize()
            fitnesses.append(results['best_fitness'])

        mean_fitness = np.mean(fitnesses)
        std_penalty = np.std(fitnesses) * 0.1
        final_fitness = mean_fitness + std_penalty

        return final_fitness

    def initialize_population(self):
        """Инициализация популяции хромосом"""
        self.population = []
        for i in range(self.population_size):
            if i == 0:
                chromosome = ParameterChromosome()  # Дефолтные параметры
            else:
                chromosome = ParameterChromosome.create_random()
            self.population.append(chromosome)

    def tournament_selection(self, k: int = 3) -> ParameterChromosome:
        """Турнирная селекция"""
        tournament = np.random.choice(len(self.population), k, replace=False)
        best_idx = tournament[0]
        best_fitness = self.fitnesses[best_idx]

        for idx in tournament[1:]:
            if self.fitnesses[idx] < best_fitness:
                best_idx = idx
                best_fitness = self.fitnesses[idx]

        return self.population[best_idx]

    def run(self) -> Tuple[ParameterChromosome, List[Dict]]:
        """Запуск эволюции параметров"""
        print("\n" + "=" * 60)
        print("PARAMETER EVOLUTION STARTED")
        print("=" * 60)
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.generations}")
        print(f"Runs per evaluation: {self.n_runs_per_evaluation}")

        self.initialize_population()

        for generation in range(self.generations):
            print(f"\n--- Generation {generation + 1}/{self.generations} ---")

            self.fitnesses = []
            for i, chromosome in enumerate(self.population):
                print(f"  Evaluating chromosome {i + 1}/{self.population_size}...", end="")
                fitness = self.evaluate_chromosome(chromosome)
                self.fitnesses.append(fitness)
                print(f" fitness = {fitness:.4e}")

                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_chromosome = chromosome
                    print(f" ✨ New best! Fitness: {fitness:.4e}")

            mean_fitness = np.mean(self.fitnesses)
            std_fitness = np.std(self.fitnesses)
            min_fitness = np.min(self.fitnesses)
            print(f"\n  Generation stats:")
            print(f"    Mean fitness: {mean_fitness:.4e}")
            print(f"    Std fitness: {std_fitness:.4e}")
            print(f"    Min fitness: {min_fitness:.4e}")
            print(f"    Best fitness overall: {self.best_fitness:.4e}")

            self.fitness_history.append({
                'generation': generation + 1,
                'mean': mean_fitness,
                'std': std_fitness,
                'min': min_fitness,
                'best': self.best_fitness
            })

            # Создание нового поколения
            if generation < self.generations - 1:
                new_population = [self.best_chromosome]
                temperature = 1.0 - (generation / self.generations)

                while len(new_population) < self.population_size:
                    if np.random.random() < 0.7:
                        parent1 = self.tournament_selection()
                        parent2 = self.tournament_selection()
                        child = parent1.crossover(parent2)
                        child = child.mutate(temperature)
                    else:
                        child = self.best_chromosome.mutate(temperature * 0.5)

                    new_population.append(child)

                self.population = new_population

        print("\n" + "=" * 60)
        print("PARAMETER EVOLUTION COMPLETED")
        print("=" * 60)
        print(f"Best fitness achieved: {self.best_fitness:.4e}")
        print("\nBest chromosome found:")
        for key, value in self.best_chromosome.to_dict().items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        return self.best_chromosome, self.fitness_history


def run_with_parameter_evolution():
    """Запуск ES-HHA с предварительной эволюцией параметров"""
    print("\n" + "=" * 60)
    print("ES-HHA WITH PARAMETER EVOLUTION")
    print("=" * 60)

    base_config = ES_HHA_Config(
        population_size=100,
        dimensions=10,
        max_FEs=10000,
        test_function_name="rastrigin",
        verbose=False
    )

    test_func, optimum, opt_value, bounds = get_test_function("rastrigin")
    base_config.global_optimum = optimum
    base_config.lb_init = bounds[0]
    base_config.ub_init = bounds[1]
    base_config.lb_opt = bounds[0]
    base_config.ub_opt = bounds[1]

    print("\n[STAGE 1] Evolution des paramètres...")
    param_optimizer = ParameterOptimizer(
        objective_function=test_func,
        base_config=base_config,
        population_size=10,
        generations=5,
        n_runs_per_evaluation=2
    )

    best_chromosome, evolution_history = param_optimizer.run()

    print("\n[STAGE 2] Exécution avec les meilleurs paramètres...")

    best_config = base_config
    best_config.update_from_chromosome(best_chromosome)
    best_config.verbose = True

    optimizer = ES_HHA(test_func, best_config)
    results = optimizer.optimize()

    print("\n[STAGE 3] Comparaison avec les paramètres de base...")

    base_config.verbose = False
    base_optimizer = ES_HHA(test_func, base_config)
    base_results = base_optimizer.optimize()

    print(f"\n{'=' * 60}")
    print("RÉSULTATS COMPARATIFS")
    print(f"{'=' * 60}")
    print(f"Paramètres de base: fitness = {base_results['best_fitness']:.4e}")
    print(f"Paramètres optimisés: fitness = {results['best_fitness']:.4e}")

    improvement = ((base_results['best_fitness'] - results['best_fitness']) /
                   base_results['best_fitness'] * 100) if base_results['best_fitness'] > 0 else 0
    print(f"Amélioration: {improvement:.2f}%")

    print(f"\nMeilleurs paramètres trouvés:")
    for key, value in best_chromosome.to_dict().items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    return results, best_chromosome


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ES-HHA: Evolutionary Status Guided Hyper-Heuristic Algorithm")
    print("=" * 60)

    mode = "normal"  # "normal" или "evolution"

    if mode == "evolution":
        results, best_chromosome = run_with_parameter_evolution()
    else:
        # ===== НАСТРОЙКА ЭКСПЕРИМЕНТА =====

        # Выбор тестовой функции
        TEST_FUNCTION = "rastrigin"  # sphere, rastrigin, rosenbrock, ackley

        # Основные параметры оптимизации
        POPULATION_SIZE = 100
        DIMENSIONS = 10
        MAX_FES = 10000

        # Параметры высокоуровневого компонента
        W1 = 0.5
        FDC_THRESHOLD = 0.6
        PD_THRESHOLD = 0.4

        # Параметры низкоуровневых эвристик
        F_EXPLOITATION = 0.8
        F_EXPLORATION = 0.8
        R_EXPLOITATION = 1.0
        R_EXPLORATION = 1.0
        CR_BINOMIAL = 0.1
        CR_EXPONENTIAL = 0.1
        R_ADAPTATION_RATE = 0.5
        P_BEST = 0.2

        # Параметры улучшений
        SHAKE_INTENSITY = 0.1
        SHAKE_THRESHOLD_IMPROVEMENTS = 5
        SHAKE_THRESHOLD_FDC = 0.1
        SHAKE_THRESHOLD_PD = 0.01
        LATE_STAGE_THRESHOLD = 0.7
        LATE_STAGE_DIVERSITY = 0.1

        # Метод обработки ограничений
        CONSTRAINT_METHOD = "tanh"
        CONSTRAINT_STEEPNESS = 1.0

        # Параметры логирования
        VERBOSE = True
        DETAILED_LOG = False
        LOG_INTERVAL = 10

        # Запуск нескольких экспериментов
        MULTIPLE_RUNS = True
        N_RUNS = 5

        # ===== СОЗДАНИЕ КОНФИГУРАЦИИ =====

        config = ES_HHA_Config(
            population_size=POPULATION_SIZE,
            dimensions=DIMENSIONS,
            max_FEs=MAX_FES,
            w1=W1,
            fdc_threshold=FDC_THRESHOLD,
            pd_threshold=PD_THRESHOLD,
            F_exploitation=F_EXPLOITATION,
            F_exploration=F_EXPLORATION,
            R_exploitation=R_EXPLOITATION,
            R_exploration=R_EXPLORATION,
            Cr_binomial=CR_BINOMIAL,
            Cr_exponential=CR_EXPONENTIAL,
            R_adaptation_rate=R_ADAPTATION_RATE,
            p_best=P_BEST,
            shake_intensity=SHAKE_INTENSITY,
            shake_threshold_improvements=SHAKE_THRESHOLD_IMPROVEMENTS,
            shake_threshold_fdc=SHAKE_THRESHOLD_FDC,
            shake_threshold_pd=SHAKE_THRESHOLD_PD,
            late_stage_threshold=LATE_STAGE_THRESHOLD,
            late_stage_diversity=LATE_STAGE_DIVERSITY,
            constraint_method=CONSTRAINT_METHOD,
            constraint_steepness=CONSTRAINT_STEEPNESS,
            verbose=VERBOSE,
            detailed_log=DETAILED_LOG,
            log_interval=LOG_INTERVAL,
            test_function_name=TEST_FUNCTION
        )

        # ===== ЗАПУСК ЭКСПЕРИМЕНТОВ =====

        if MULTIPLE_RUNS:
            stats, all_results = run_multiple_experiments(config, n_runs=N_RUNS)
        else:
            test_func, optimum, opt_value, bounds = get_test_function(TEST_FUNCTION)
            config.global_optimum = optimum
            config.lb_init = bounds[0]
            config.ub_init = bounds[1]
            config.lb_opt = bounds[0]
            config.ub_opt = bounds[1]

            optimizer = ES_HHA(test_func, config)
            results = optimizer.optimize()

            print("\n" + "=" * 50)
            print("FINAL RESULTS:")
            print("=" * 50)
            print(f"Best fitness: {results['best_fitness']:.6e}")
            print(f"Best solution: {results['best_solution']}")
            print(f"Total iterations: {results['iterations']}")
            print(f"Execution time: {results['execution_time']:.2f} sec")

            if 'distance_to_optimum_history' in results and results['distance_to_optimum_history']:
                print(f"Final distance to optimum: {results['distance_to_optimum_history'][-1]:.6f}")
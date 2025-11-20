import numpy as np
import math
from abc import ABC, abstractmethod
from typing import List, Callable, Dict
from collections import defaultdict
import time


# Абстрактный базовый класс для всех LLH
class LLHOperator(ABC):
    def __init__(self, name: str, F: float = 0.8):
        self.name = name
        self.F = F  # Параметр F для каждой эвристики

    @abstractmethod
    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        pass


# Конкретные классы операторов эксплуатации
class UniformLLH(LLHOperator):
    def __init__(self, F: float = 0.8):
        super().__init__("uniform", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        R = kwargs.get('R', 1.0)
        dimensions = best_solution.shape[0]
        return best_solution + R * np.random.uniform(-1, 1, dimensions)


class NormalLLH(LLHOperator):
    def __init__(self, F: float = 0.8):
        super().__init__("normal", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        R = kwargs.get('R', 1.0)
        dimensions = best_solution.shape[0]
        return best_solution + R * np.random.normal(0, 1, dimensions)


class LevyLLH(LLHOperator):
    def __init__(self, F: float = 0.8):
        super().__init__("levy", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        R = kwargs.get('R', 1.0)
        dimensions = best_solution.shape[0]
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (beta * math.gamma((1 + beta) / 2) * 2 ** ((beta - 1) / 2))) ** (1 / beta)

        u = np.random.normal(0, sigma, dimensions)
        v = np.random.normal(0, 1, dimensions)
        step = u / (np.abs(v) ** (1 / beta))

        return best_solution + R * step


class DEBest1LLH(LLHOperator):
    def __init__(self, F: float = 0.8):
        super().__init__("DE_best_1", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        n = len(population)

        # Создаем список индексов 
        if not hasattr(self, '_available_indices'):
            self._available_indices = [i for i in range(n)]

        indices = self._available_indices.copy()
        if current_index in indices:
            indices.remove(current_index)

        r1, r2 = np.random.choice(indices, 2, replace=False)

        return best_solution + self.F * (population[r1] - population[r2])


# Конкретные классы операторов исследования
class DERand1LLH(LLHOperator):
    def __init__(self, F: float = 0.8):
        super().__init__("DE_rand_1", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        n = len(population)

        # Создаем список индексов 
        if not hasattr(self, '_available_indices'):
            self._available_indices = [i for i in range(n)]

        indices = self._available_indices.copy()
        if current_index in indices:
            indices.remove(current_index)

        r1, r2, r3 = np.random.choice(indices, 3, replace=False)

        return population[r1] + self.F * (population[r2] - population[r3])


class DECurrent1LLH(LLHOperator):
    def __init__(self, F: float = 0.8):
        super().__init__("DE_cur_1", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        n = len(population)
        current_ind = population[current_index]

        # Создаем список индексов 
        if not hasattr(self, '_available_indices'):
            self._available_indices = [i for i in range(n)]

        indices = self._available_indices.copy()
        if current_index in indices:
            indices.remove(current_index)

        r1, r2 = np.random.choice(indices, 2, replace=False)

        return current_ind + self.F * (population[r1] - population[r2])


class DECurrentToBest1LLH(LLHOperator):
    def __init__(self, F: float = 0.8):
        super().__init__("DE_cur_to_best_1", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        n = len(population)
        current_ind = population[current_index]

        # Создаем список индексов 
        if not hasattr(self, '_available_indices'):
            self._available_indices = [i for i in range(n)]

        indices = self._available_indices.copy()
        if current_index in indices:
            indices.remove(current_index)

        r1, r2 = np.random.choice(indices, 2, replace=False)

        return current_ind + self.F * (best_solution - current_ind) + self.F * (population[r1] - population[r2])


class DECurrentToPBest1LLH(LLHOperator):
    def __init__(self, F: float = 0.8):
        super().__init__("DE_cur_to_pbest_1", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        n = len(population)
        current_ind = population[current_index]

        p = 0.2
        p_size = max(1, int(n * p))
        fitness = kwargs.get('fitness', None)

        if fitness is not None:
            best_indices = np.argsort(fitness)[:p_size]
            pbest = population[np.random.choice(best_indices)]
        else:
            pbest = best_solution

        # Создаем список индексов 
        if not hasattr(self, '_available_indices'):
            self._available_indices = [i for i in range(n)]

        indices = self._available_indices.copy()
        if current_index in indices:
            indices.remove(current_index)

        r1, r2 = np.random.choice(indices, 2, replace=False)

        return current_ind + self.F * (pbest - current_ind) + self.F * (population[r1] - population[r2])


# Класс для управления пулами эвристик
class LLHPoolManager:
    def __init__(self, exploitation_Fs: Dict[str, float] = None, exploration_Fs: Dict[str, float] = None):
        self.exploitation_pool = self.initialize_exploitation_pool(exploitation_Fs or {})
        self.exploration_pool = self.initialize_exploration_pool(exploration_Fs or {})

    def initialize_exploitation_pool(self, Fs: Dict[str, float]) -> List[LLHOperator]:
        pool = [
            UniformLLH(F=Fs.get('uniform', 0.8)),
            NormalLLH(F=Fs.get('normal', 0.8)),
            LevyLLH(F=Fs.get('levy', 0.8)),
            DEBest1LLH(F=Fs.get('DE_best_1', 0.8))
        ]
        return pool

    def initialize_exploration_pool(self, Fs: Dict[str, float]) -> List[LLHOperator]:
        pool = [
            DERand1LLH(F=Fs.get('DE_rand_1', 0.8)),
            DECurrent1LLH(F=Fs.get('DE_cur_1', 0.8)),
            DECurrentToBest1LLH(F=Fs.get('DE_cur_to_best_1', 0.8)),
            DECurrentToPBest1LLH(F=Fs.get('DE_cur_to_pbest_1', 0.8))
        ]
        return pool

    def get_exploitation_pool(self) -> List[LLHOperator]:
        return self.exploitation_pool

    def get_exploration_pool(self) -> List[LLHOperator]:
        return self.exploration_pool


# Основной класс ES-HHA 
class ES_HHA:
    def __init__(self, objective_function: Callable, population_size=100, dimensions=30,
                 max_FEs=30000, w1=0.5, R=1.0,
                 exploitation_Fs: Dict[str, float] = None, exploration_Fs: Dict[str, float] = None,
                 verbose=True, detailed_log=False):

        self.objective_function = objective_function
        self.population_size = population_size
        self.dimensions = dimensions
        self.max_FEs = max_FEs
        self.w1 = w1
        self.R = R
        self.verbose = verbose
        self.detailed_log = detailed_log

        # Статистика
        self.llh_usage_count = defaultdict(int)
        self.pool_usage_count = defaultdict(int)
        self.fdc_history = []
        self.pd_history = []
        self.best_fitness_history = []

        self.llh_manager = LLHPoolManager(exploitation_Fs, exploration_Fs)
        self.exploitation_pool = self.llh_manager.get_exploitation_pool()
        self.exploration_pool = self.llh_manager.get_exploration_pool()

    def initialize_population(self, lb, ub):
        population = np.random.uniform(lb, ub, (self.population_size, self.dimensions))
        fitness = np.array([self.objective_function(ind) for ind in population])
        best_idx = np.argmin(fitness)
        return population, fitness, population[best_idx], fitness[best_idx]

    def calculate_FDC(self, population, fitness, best_solution):
        distances = np.array([np.sqrt(np.sum((ind - best_solution) ** 2)) for ind in population])
        f_mean, d_mean = np.mean(fitness), np.mean(distances)
        numerator = np.sum((fitness - f_mean) * (distances - d_mean))
        denominator = np.sqrt(np.sum((fitness - f_mean) ** 2) * np.sum((distances - d_mean) ** 2))
        fdc_value = numerator / denominator if denominator != 0 else 0
        self.fdc_history.append(fdc_value)
        return fdc_value

    def calculate_PD(self, population, lb, ub):
        centroid = np.mean(population, axis=0)
        diversity_sum = np.sum([abs(population[i, j] - centroid[j]) / (ub[j] - lb[j])
                                for i in range(len(population)) for j in range(self.dimensions)])
        pd_value = diversity_sum / (len(population) * self.dimensions)
        self.pd_history.append(pd_value)
        return pd_value

    def select_LLH(self, FDC: float, PD: float) -> tuple:
        # Нормализуем FDC и PD к диапазону [0, 1] 
        FDC_norm = (FDC + 1) / 2  # FDC ∈ [-1, 1] -> [0, 1]
        PD_norm = PD  # PD уже ∈ [0, 1]

        FDC_threshold = 0.6  # Порог для "высокого" FDC
        PD_threshold = 0.4  # Порог для "высокого" PD

        FDC_high = FDC_norm > FDC_threshold
        PD_high = PD_norm > PD_threshold

        selection_info = {
            'FDC': FDC,
            'PD': PD,
            'FDC_norm': FDC_norm,
            'PD_norm': PD_norm,
            'FDC_high': FDC_high,
            'PD_high': PD_high,
            'case': None,
            'P_exploit': None,
            'pool_type': None
        }

        if FDC_high and PD_high:
            # Случай (a): FDC высокий, PD высокий -> Эксплуатация
            pool = self.exploitation_pool
            pool_type = "exploitation"
            selection_info['case'] = 'a'
            selection_info['pool_type'] = pool_type

        elif FDC_high and not PD_high:
            # Случай (b): FDC высокий, PD низкий -> Сбалансированная стратегия
            P_exploit = self.w1 * FDC_norm + (1 - self.w1) * (1 - PD_norm)
            selection_info['P_exploit'] = P_exploit
            selection_info['case'] = 'b'

            if np.random.random() < P_exploit:
                pool = self.exploitation_pool
                pool_type = "balanced_exploitation"
            else:
                pool = self.exploration_pool
                pool_type = "balanced_exploration"
            selection_info['pool_type'] = pool_type

        elif not FDC_high and PD_high:
            # Случай (c): FDC низкий, PD высокий -> Сбалансированная стратегия
            P_exploit = self.w1 * PD_norm + (1 - self.w1) * (1 - FDC_norm)
            selection_info['P_exploit'] = P_exploit
            selection_info['case'] = 'c'

            if np.random.random() < P_exploit:
                pool = self.exploitation_pool
                pool_type = "balanced_exploitation"
            else:
                pool = self.exploration_pool
                pool_type = "balanced_exploration"
            selection_info['pool_type'] = pool_type

        else:
            # Случай (d): FDC низкий, PD низкий -> Исследование
            pool = self.exploration_pool
            pool_type = "exploration"
            selection_info['case'] = 'd'
            selection_info['pool_type'] = pool_type

        selected_operator = np.random.choice(pool)

        # Обновляие статистики
        self.llh_usage_count[selected_operator.name] += 1
        self.pool_usage_count[pool_type] += 1

        return selected_operator, selection_info

    def apply_LLH(self, operator: LLHOperator, population: np.ndarray,
                  best_solution: np.ndarray, current_index: int, fitness: np.ndarray = None) -> np.ndarray:
        kwargs = {'R': self.R}
        if fitness is not None:
            kwargs['fitness'] = fitness
        return operator.apply(population, best_solution, current_index, **kwargs)

    def create_new_generation(self, population: np.ndarray, fitness: np.ndarray,
                              best_solution: np.ndarray, best_fitness: float, lb: np.ndarray, ub: np.ndarray) -> tuple:
        FDC = self.calculate_FDC(population, fitness, best_solution)
        PD = self.calculate_PD(population, lb, ub)

        new_population = []
        new_fitness = []
        iteration_llh_info = []

        # Генерируем новых индивидов и сразу вычисляем fitness
        for i in range(self.population_size):
            operator, selection_info = self.select_LLH(FDC, PD)
            new_individual = self.apply_LLH(operator, population, best_solution, i, fitness)
            new_individual = np.clip(new_individual, lb, ub)
            new_fitness_val = self.objective_function(new_individual)

            new_population.append(new_individual)
            new_fitness.append(new_fitness_val)
            iteration_llh_info.append({
                'operator': operator.name,
                'pool_type': selection_info['pool_type'],
                'fitness': new_fitness_val
            })

        new_fitness = np.array(new_fitness)
        new_population = np.array(new_population)

        # Жадный выбор между старым и новым поколением
        final_population = []
        final_fitness = []
        new_best_solution = best_solution
        new_best_fitness = best_fitness
        improvements = 0

        for i in range(self.population_size):
            if new_fitness[i] < fitness[i]:
                final_population.append(new_population[i])
                final_fitness.append(new_fitness[i])
                improvements += 1

                if new_fitness[i] < new_best_fitness:
                    new_best_solution = new_population[i].copy()
                    new_best_fitness = new_fitness[i]
            else:
                final_population.append(population[i])
                final_fitness.append(fitness[i])

        #  история лучшей приспособленности
        self.best_fitness_history.append(new_best_fitness)

        return (np.array(final_population), np.array(final_fitness),
                new_best_solution, new_best_fitness, improvements, iteration_llh_info)

    def check_stopping_criteria(self, current_FEs):
        if current_FEs >= self.max_FEs:
            return True, "Max FEs"
        return False, "Continue"

    def print_iteration_info(self, iteration, best_fitness, improvements, FDC, PD, llh_info):
        """Вывод детальной информации об итерации"""
        if not self.verbose:
            return

        if iteration % 10 == 0 or self.detailed_log:
            # Анализ использования операторов в этой итерации
            op_count = {}
            pool_count = {}
            for info in llh_info:
                op_count[info['operator']] = op_count.get(info['operator'], 0) + 1
                pool_count[info['pool_type']] = pool_count.get(info['pool_type'], 0) + 1

            print(f"\n--- Iteration {iteration} ---")
            print(f"Best fitness: {best_fitness:.6f}")
            print(f"Improvements: {improvements}/{self.population_size}")
            print(f"FDC: {FDC:.3f}, PD: {PD:.3f}")
            print(f"Operators used: {dict(op_count)}")
            print(f"Pool distribution: {dict(pool_count)}")

    def print_final_statistics(self, start_time, end_time):
        """Вывод финальной статистики"""
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)

        total_llh_uses = sum(self.llh_usage_count.values())
        total_pool_uses = sum(self.pool_usage_count.values())

        print(f"\nAlgorithm execution time: {end_time - start_time:.2f} seconds")
        print(f"Total iterations: {len(self.best_fitness_history)}")
        print(f"Final best fitness: {self.best_fitness_history[-1]:.6f}")
        print(f"Total FEs: {self.population_size * len(self.best_fitness_history)}")

        print(f"\nLLH Usage Statistics:")
        for llh, count in sorted(self.llh_usage_count.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_llh_uses) * 100
            print(f"  {llh}: {count} uses ({percentage:.1f}%)")

        print(f"\nPool Usage Statistics:")
        for pool, count in sorted(self.pool_usage_count.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_pool_uses) * 100
            print(f"  {pool}: {count} uses ({percentage:.1f}%)")

        print(f"\nFDC Statistics:")
        print(f"  Average FDC: {np.mean(self.fdc_history):.3f}")
        print(f"  Min FDC: {np.min(self.fdc_history):.3f}")
        print(f"  Max FDC: {np.max(self.fdc_history):.3f}")

        print(f"\nPD Statistics:")
        print(f"  Average PD: {np.mean(self.pd_history):.3f}")
        print(f"  Min PD: {np.min(self.pd_history):.3f}")
        print(f"  Max PD: {np.max(self.pd_history):.3f}")

        print(f"\nConvergence Analysis:")
        initial_fitness = self.best_fitness_history[0]
        final_fitness = self.best_fitness_history[-1]
        improvement = initial_fitness - final_fitness
        print(f"  Initial fitness: {initial_fitness:.6f}")
        print(f"  Final fitness: {final_fitness:.6f}")
        print(f"  Total improvement: {improvement:.6f}")

    def optimize(self, lb, ub):
        start_time = time.time()
        population, fitness, best_solution, best_fitness = self.initialize_population(lb, ub)
        current_FEs = self.population_size
        iteration = 0

        if self.verbose:
            print(f"Initial best fitness: {best_fitness:.6f}")

        while True:
            # Создаем новое поколение
            population, fitness, best_solution, best_fitness, improvements, llh_info = self.create_new_generation(
                population, fitness, best_solution, best_fitness, lb, ub)

            current_FEs += self.population_size

            # Вывод информации об итерации
            FDC = self.fdc_history[-1] if self.fdc_history else 0
            PD = self.pd_history[-1] if self.pd_history else 0
            self.print_iteration_info(iteration, best_fitness, improvements, FDC, PD, llh_info)

            stop, reason = self.check_stopping_criteria(current_FEs)
            if stop:
                break

            iteration += 1

        end_time = time.time()

        if self.verbose:
            self.print_final_statistics(start_time, end_time)

        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'total_FEs': current_FEs,
            'iterations': iteration,
            'llh_usage': dict(self.llh_usage_count),
            'pool_usage': dict(self.pool_usage_count),
            'fdc_history': self.fdc_history,
            'pd_history': self.pd_history,
            'fitness_history': self.best_fitness_history,
            'execution_time': end_time - start_time
        }


# Примеры целевых функций
def sphere_function(x):
    return np.sum(x ** 2)


def rastrigin_function(x):
    A = 10
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def rosenbrock_function(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def ackley_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + a + np.exp(1)


def cec2014_shifted_rotated_rosenbrock(x):
    n = len(x)
    result = 0
    for i in range(n - 1):
        result += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
    return result + 400


def cec2014_shifted_rastrigin(x):
    A = 10
    n = len(x)
    result = A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))
    return result + 800


if __name__ == "__main__":
    dimensions = 30

    exploitation_Fs = {
        'uniform': 0.5, 'normal': 0.6, 'levy': 0.7, 'DE_best_1': 0.8
    }

    exploration_Fs = {
        'DE_rand_1': 0.9, 'DE_cur_1': 0.8, 'DE_cur_to_best_1': 0.7, 'DE_cur_to_pbest_1': 0.6
    }

    print("=== Testing ES-HHA Algorithm with Detailed Logging ===")

   
    print("\n" + "=" * 50)
    print("SPHERE FUNCTION")
    print("=" * 50)

    es_hha_sphere = ES_HHA(
        objective_function=sphere_function,
        dimensions=dimensions,
        max_FEs=10000,  
        exploitation_Fs=exploitation_Fs,
        exploration_Fs=exploration_Fs,
        verbose=True,
        detailed_log=False 
    )
    lb = np.full(dimensions, -100)
    ub = np.full(dimensions, 100)
    results = es_hha_sphere.optimize(lb, ub)

    
    print("\n" + "=" * 50)
    print("CEC2014 ROSENBROCK FUNCTION")
    print("=" * 50)

    es_hha_cec = ES_HHA(
        objective_function=cec2014_shifted_rotated_rosenbrock,
        dimensions=dimensions,
        max_FEs=15000,
        exploitation_Fs=exploitation_Fs,
        exploration_Fs=exploration_Fs,
        verbose=True,
        detailed_log=False
    )
    lb = np.full(dimensions, -100)
    ub = np.full(dimensions, 100)
    results_cec = es_hha_cec.optimize(lb, ub)


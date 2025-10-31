import numpy as np
import math
from abc import ABC, abstractmethod
from typing import List, Callable, Dict


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

        # Создаем список индексов один раз и переиспользуем
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

        # Создаем список индексов один раз и переиспользуем
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

        # Создаем список индексов один раз и переиспользуем
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

        # Создаем список индексов один раз и переиспользуем
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

        # Создаем список индексов один раз и переиспользуем
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


# Основной класс ES-HHA с улучшенной архитектурой
class ES_HHA:
    def __init__(self, objective_function: Callable, population_size=100, dimensions=30,
                 max_FEs=30000, w1=0.5, R=1.0,
                 exploitation_Fs: Dict[str, float] = None, exploration_Fs: Dict[str, float] = None):

        self.objective_function = objective_function
        self.population_size = population_size
        self.dimensions = dimensions
        self.max_FEs = max_FEs
        self.w1 = w1
        self.R = R

        # Используем менеджер пулов эвристик
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
        return numerator / denominator if denominator != 0 else 0

    def calculate_PD(self, population, lb, ub):
        centroid = np.mean(population, axis=0)
        diversity_sum = np.sum([abs(population[i, j] - centroid[j]) / (ub[j] - lb[j])
                                for i in range(len(population)) for j in range(self.dimensions)])
        return diversity_sum / (len(population) * self.dimensions)

    def calculate_P1(self, FDC, PD):
        P1 = self.w1 * FDC + (1 - self.w1) * PD
        return max(0.1, min(0.9, P1))

    def select_LLH(self, P1) -> LLHOperator:
        if np.random.random() < P1:
            pool = self.exploitation_pool
        else:
            pool = self.exploration_pool
        return np.random.choice(pool)

    def apply_LLH(self, operator: LLHOperator, population: np.ndarray,
                  best_solution: np.ndarray, current_index: int, fitness: np.ndarray = None) -> np.ndarray:
        kwargs = {'R': self.R}
        if fitness is not None:
            kwargs['fitness'] = fitness
        return operator.apply(population, best_solution, current_index, **kwargs)

    def create_new_generation(self, population: np.ndarray, fitness: np.ndarray,
                              best_solution: np.ndarray, best_fitness: float, lb: np.ndarray, ub: np.ndarray) -> tuple:
        """Создает новое поколение вместо замены в текущем"""
        FDC = self.calculate_FDC(population, fitness, best_solution)
        PD = self.calculate_PD(population, lb, ub)
        P1 = self.calculate_P1(FDC, PD)

        new_population = []
        new_fitness = []

        # Генерируем новых индивидов и сразу вычисляем fitness
        for i in range(self.population_size):
            operator = self.select_LLH(P1)
            new_individual = self.apply_LLH(operator, population, best_solution, i, fitness)
            new_individual = np.clip(new_individual, lb, ub)
            new_fitness_val = self.objective_function(new_individual)

            new_population.append(new_individual)
            new_fitness.append(new_fitness_val)

        new_fitness = np.array(new_fitness)
        new_population = np.array(new_population)

        # Жадный выбор между старым и новым поколением
        final_population = []
        final_fitness = []
        new_best_solution = best_solution
        new_best_fitness = best_fitness

        for i in range(self.population_size):
            if new_fitness[i] < fitness[i]:
                final_population.append(new_population[i])
                final_fitness.append(new_fitness[i])

                if new_fitness[i] < new_best_fitness:
                    new_best_solution = new_population[i].copy()
                    new_best_fitness = new_fitness[i]
            else:
                final_population.append(population[i])
                final_fitness.append(fitness[i])

        return (np.array(final_population), np.array(final_fitness),
                new_best_solution, new_best_fitness)

    def check_stopping_criteria(self, current_FEs):
        if current_FEs >= self.max_FEs:
            return True, "Max FEs"
        return False, "Continue"

    def optimize(self, lb, ub):
        population, fitness, best_solution, best_fitness = self.initialize_population(lb, ub)
        current_FEs = self.population_size
        iteration = 0

        while True:
            # Создаем новое поколение
            population, fitness, best_solution, best_fitness = self.create_new_generation(
                population, fitness, best_solution, best_fitness, lb, ub)

            current_FEs += self.population_size

            stop, reason = self.check_stopping_criteria(current_FEs)
            if stop:
                break

            iteration += 1
            if iteration % 50 == 0:
                print(f"Iteration {iteration}, Best fitness: {best_fitness:.6f}")

        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'total_FEs': current_FEs,
            'iterations': iteration
        }


# Примеры целевых функций
def sphere_function(x):
    """Sphere function"""
    return np.sum(x ** 2)


def rastrigin_function(x):
    """Rastrigin function"""
    A = 10
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def rosenbrock_function(x):
    """Rosenbrock function"""
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def ackley_function(x):
    """Ackley function"""
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + a + np.exp(1)


if __name__ == "__main__":
    # Тестирование с разными целевыми функциями и параметрами F
    dimensions = 30

    # Кастомные параметры F для разных эвристик
    exploitation_Fs = {
        'uniform': 0.5,
        'normal': 0.6,
        'levy': 0.7,
        'DE_best_1': 0.8
    }

    exploration_Fs = {
        'DE_rand_1': 0.9,
        'DE_cur_1': 0.8,
        'DE_cur_to_best_1': 0.7,
        'DE_cur_to_pbest_1': 0.6
    }

    print("=== Testing ES-HHA Algorithm ===")

    # Sphere function
    print("\n=== Sphere Function ===")
    es_hha_sphere = ES_HHA(
        objective_function=sphere_function,
        dimensions=dimensions,
        exploitation_Fs=exploitation_Fs,
        exploration_Fs=exploration_Fs
    )
    lb = np.full(dimensions, -100)
    ub = np.full(dimensions, 100)
    results = es_hha_sphere.optimize(lb, ub)
    print(f"Best fitness: {results['best_fitness']:.6f}")
    print(f"Total FEs: {results['total_FEs']}")
    print(f"Iterations: {results['iterations']}")

    # Rastrigin function
    print("\n=== Rastrigin Function ===")
    es_hha_rastrigin = ES_HHA(
        objective_function=rastrigin_function,
        dimensions=dimensions,
        exploitation_Fs=exploitation_Fs,
        exploration_Fs=exploration_Fs
    )
    lb = np.full(dimensions, -5.12)
    ub = np.full(dimensions, 5.12)
    results = es_hha_rastrigin.optimize(lb, ub)
    print(f"Best fitness: {results['best_fitness']:.6f}")
    print(f"Total FEs: {results['total_FEs']}")
    print(f"Iterations: {results['iterations']}")

    # Ackley function
    print("\n=== Ackley Function ===")
    es_hha_ackley = ES_HHA(
        objective_function=ackley_function,
        dimensions=dimensions,
        exploitation_Fs=exploitation_Fs,
        exploration_Fs=exploration_Fs
    )
    lb = np.full(dimensions, -32.768)
    ub = np.full(dimensions, 32.768)
    results = es_hha_ackley.optimize(lb, ub)
    print(f"Best fitness: {results['best_fitness']:.6f}")
    print(f"Total FEs: {results['total_FEs']}")
    print(f"Iterations: {results['iterations']}")

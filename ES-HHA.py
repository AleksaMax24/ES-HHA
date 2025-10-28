import numpy as np
import math
from abc import ABC, abstractmethod
from typing import List


# Абстрактный базовый класс для всех LLH
class LLHOperator(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        pass


# Конкретные классы операторов эксплуатации
class UniformLLH(LLHOperator):
    def __init__(self):
        super().__init__("uniform")

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        R = kwargs.get('R', 1.0)
        dimensions = best_solution.shape[0]
        return best_solution + R * np.random.uniform(-1, 1, dimensions)


class NormalLLH(LLHOperator):
    def __init__(self):
        super().__init__("normal")

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        R = kwargs.get('R', 1.0)
        dimensions = best_solution.shape[0]
        return best_solution + R * np.random.normal(0, 1, dimensions)


class LevyLLH(LLHOperator):
    def __init__(self):
        super().__init__("levy")

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
    def __init__(self):
        super().__init__("DE_best_1")

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        F = kwargs.get('F', 0.8)
        n = len(population)

        indices = [i for i in range(n) if i != current_index]
        r1, r2 = np.random.choice(indices, 2, replace=False)

        return best_solution + F * (population[r1] - population[r2])


# Конкретные классы операторов исследования
class DERand1LLH(LLHOperator):
    def __init__(self):
        super().__init__("DE_rand_1")

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        F = kwargs.get('F', 0.8)
        n = len(population)

        indices = [i for i in range(n) if i != current_index]
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)

        return population[r1] + F * (population[r2] - population[r3])


class DECurrent1LLH(LLHOperator):
    def __init__(self):
        super().__init__("DE_cur_1")

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        F = kwargs.get('F', 0.8)
        n = len(population)
        current_ind = population[current_index]

        indices = [i for i in range(n) if i != current_index]
        r1, r2 = np.random.choice(indices, 2, replace=False)

        return current_ind + F * (population[r1] - population[r2])


class DECurrentToBest1LLH(LLHOperator):
    def __init__(self):
        super().__init__("DE_cur_to_best_1")

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        F = kwargs.get('F', 0.8)
        n = len(population)
        current_ind = population[current_index]

        indices = [i for i in range(n) if i != current_index]
        r1, r2 = np.random.choice(indices, 2, replace=False)

        return current_ind + F * (best_solution - current_ind) + F * (population[r1] - population[r2])


class DECurrentToPBest1LLH(LLHOperator):
    def __init__(self):
        super().__init__("DE_cur_to_pbest_1")

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        F = kwargs.get('F', 0.8)
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

        indices = [i for i in range(n) if i != current_index]
        r1, r2 = np.random.choice(indices, 2, replace=False)

        return current_ind + F * (pbest - current_ind) + F * (population[r1] - population[r2])


# Основной класс ES-HHA с новой архитектурой
class ES_HHA:
    def __init__(self, objective_function: Callable, population_size=100, dimensions=30,
                 max_FEs=30000, w1=0.5, R=1.0, F=0.8):
        self.objective_function = objective_function
        self.population_size = population_size
        self.dimensions = dimensions
        self.max_FEs = max_FEs
        self.w1 = w1
        self.R = R
        self.F = F

        self.exploitation_pool = self.initialize_exploitation_pool()
        self.exploration_pool = self.initialize_exploration_pool()

    def initialize_exploitation_pool(self) -> List[LLHOperator]:
        return [UniformLLH(), NormalLLH(), LevyLLH(), DEBest1LLH()]

    def initialize_exploration_pool(self) -> List[LLHOperator]:
        return [DERand1LLH(), DECurrent1LLH(), DECurrentToBest1LLH(), DECurrentToPBest1LLH()]

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
        kwargs = {'R': self.R, 'F': self.F}
        if fitness is not None:
            kwargs['fitness'] = fitness
        return operator.apply(population, best_solution, current_index, **kwargs)

    def greedy_selection(self, current_ind, new_ind, current_fitness, new_fitness):
        if new_fitness < current_fitness:
            return new_ind, new_fitness, True
        return current_ind, current_fitness, False

    def check_stopping_criteria(self, current_FEs):
        if current_FEs >= self.max_FEs:
            return True, "Max FEs"
        return False, "Continue"


    def optimize(self, lb, ub):
        population, fitness, best_solution, best_fitness = self.initialize_population(lb, ub)
        current_FEs = self.population_size
        iteration = 0

        while True:
            FDC = self.calculate_FDC(population, fitness, best_solution)
            PD = self.calculate_PD(population, lb, ub)
            P1 = self.calculate_P1(FDC, PD)

            for i in range(self.population_size):
                operator = self.select_LLH(P1)
                new_individual = self.apply_LLH(operator, population, best_solution, i, fitness)
                new_individual = np.clip(new_individual, lb, ub)

                new_fitness = self.objective_function(new_individual)
                current_FEs += 1

                population[i], fitness[i], improved = self.greedy_selection(
                    population[i], new_individual, fitness[i], new_fitness)

                if improved and fitness[i] < best_fitness:
                    best_solution, best_fitness = population[i].copy(), fitness[i]

            stop, reason = self.check_stopping_criteria(current_FEs)
            if stop:
                break

            iteration += 1

        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'total_FEs': current_FEs,
            'iterations': iteration
        }

# Примеры целевых функций
def sphere_function(x):
    """Sphere function"""
    return np.sum(x**2)

def rastrigin_function(x):
    """Rastrigin function"""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock_function(x):
    """Rosenbrock function"""
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley_function(x):
    """Ackley function"""
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + a + np.exp(1)


if __name__ == "__main__":
    # Тестирование с разными целевыми функциями
    dimensions = 30

    # Sphere function
    print("=== Sphere Function ===")
    es_hha_sphere = ES_HHA(objective_function=sphere_function, dimensions=dimensions)
    lb = np.full(dimensions, -100)
    ub = np.full(dimensions, 100)
    results = es_hha_sphere.optimize(lb, ub)
    print(f"Best fitness: {results['best_fitness']:.6f}")
    print(f"Total FEs: {results['total_FEs']}")

    # Rastrigin function
    print("\n=== Rastrigin Function ===")
    es_hha_rastrigin = ES_HHA(objective_function=rastrigin_function, dimensions=dimensions)
    lb = np.full(dimensions, -5.12)
    ub = np.full(dimensions, 5.12)
    results = es_hha_rastrigin.optimize(lb, ub)
    print(f"Best fitness: {results['best_fitness']:.6f}")
    print(f"Total FEs: {results['total_FEs']}")

    # Ackley function
    print("\n=== Ackley Function ===")
    es_hha_ackley = ES_HHA(objective_function=ackley_function, dimensions=dimensions)
    lb = np.full(dimensions, -32.768)
    ub = np.full(dimensions, 32.768)
    results = es_hha_ackley.optimize(lb, ub)
    print(f"Best fitness: {results['best_fitness']:.6f}")
    print(f"Total FEs: {results['total_FEs']}")

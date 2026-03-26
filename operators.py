"""
operators.py
Операторы низкого уровня (LLH) для ES-HHA:
- Эксплуатирующие операторы (локальный поиск вокруг best)
- Исследующие операторы (DE-мутации, случайные возмущения)
- Кроссоверы
- Управление пулами операторов
"""

import numpy as np
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field


# ============================================================================
# КРОССОВЕРЫ
# ============================================================================

class CrossoverOperator(ABC):
    """Базовый класс для операторов кроссовера"""

    def __init__(self, name: str, Cr: float = 0.1):
        self.name = name
        self.Cr = Cr

    @abstractmethod
    def apply(self, target: np.ndarray, trial: np.ndarray) -> np.ndarray:
        pass


class BinomialCrossover(CrossoverOperator):
    """Биномиальный кроссовер"""

    def __init__(self, Cr: float = 0.1):
        super().__init__("binomial", Cr)

    def apply(self, target: np.ndarray, trial: np.ndarray) -> np.ndarray:
        D = len(target)
        j_rand = np.random.randint(0, D)
        mask = np.random.rand(D) < self.Cr
        mask[j_rand] = True
        return np.where(mask, trial, target)


class ExponentialCrossover(CrossoverOperator):
    """Экспоненциальный кроссовер"""

    def __init__(self, Cr: float = 0.1):
        super().__init__("exponential", Cr)

    def apply(self, target: np.ndarray, trial: np.ndarray) -> np.ndarray:
        D = len(target)
        j_rand = np.random.randint(0, D)
        j = j_rand
        L = 0

        while np.random.rand() < self.Cr and L < D:
            L += 1
            j = (j + 1) % D

        offspring = target.copy()
        for k in range(L):
            idx = (j_rand + k) % D
            offspring[idx] = trial[idx]

        return offspring


# ============================================================================
# БАЗОВЫЙ КЛАСС ОПЕРАТОРА НИЗКОГО УРОВНЯ
# ============================================================================

class LLHOperator(ABC):
    """Базовый класс для операторов низкого уровня (Low-Level Heuristics)"""

    def __init__(self, name: str, F: float = 0.8):
        self.name = name
        self.F = F

    @abstractmethod
    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        pass


# ============================================================================
# ЭКСПЛУАТИРУЮЩИЕ ОПЕРАТОРЫ (ЛОКАЛЬНЫЙ ПОИСК ВОКРУГ BEST)
# ============================================================================

class UniformLLH(LLHOperator):
    """Равномерное возмущение вокруг лучшего решения"""

    def __init__(self, F: float = 0.8):
        super().__init__("uniform", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        R = kwargs.get('R', 1.0)
        dim = best_solution.shape[0]
        return best_solution + R * np.random.uniform(-1, 1, dim)


class NormalLLH(LLHOperator):
    """Нормальное возмущение вокруг лучшего решения"""

    def __init__(self, F: float = 0.8):
        super().__init__("normal", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        R = kwargs.get('R', 1.0)
        dim = best_solution.shape[0]
        return best_solution + R * np.random.normal(0, 1, dim)


class LevyLLH(LLHOperator):
    """Возмущение с использованием распределения Леви"""

    def __init__(self, F: float = 0.8):
        super().__init__("levy", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        R = kwargs.get('R', 1.0)
        dim = best_solution.shape[0]
        beta = 1.5

        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (beta * math.gamma((1 + beta) / 2) * 2 ** ((beta - 1) / 2))) ** (1 / beta)

        u = np.random.normal(0, sigma, dim)
        v = np.random.normal(0, 1, dim)
        step = u / (np.abs(v) ** (1 / beta))

        return best_solution + R * step


class DEBest1LLH(LLHOperator):
    """DE/best/1 мутационный оператор"""

    def __init__(self, F: float = 0.8, crossover: CrossoverOperator = None):
        super().__init__("DE_best_1", F)
        self.crossover = crossover if crossover else BinomialCrossover(Cr=0.1)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        n = len(population)

        indices = np.arange(n)
        mask = indices != current_index
        available_indices = indices[mask]
        r1, r2 = np.random.choice(available_indices, 2, replace=False)

        mutant = best_solution + self.F * (population[r1] - population[r2])
        trial = self.crossover.apply(best_solution, mutant)

        return trial


# ============================================================================
# ИССЛЕДУЮЩИЕ ОПЕРАТОРЫ
# ============================================================================

class UniformCurrentLLH(LLHOperator):
    """Равномерное возмущение вокруг текущего решения"""

    def __init__(self, F: float = 0.8):
        super().__init__("uniform_current", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        R = kwargs.get('R', 1.0)
        current_solution = population[current_index].copy()
        dim = current_solution.shape[0]

        return current_solution + R * np.random.uniform(-1, 1, dim)


class NormalCurrentLLH(LLHOperator):
    """Нормальное возмущение вокруг текущего решения"""

    def __init__(self, F: float = 0.8):
        super().__init__("normal_current", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        R = kwargs.get('R', 1.0)
        current_solution = population[current_index].copy()
        dim = current_solution.shape[0]

        return current_solution + R * np.random.normal(0, 1, dim)


class LevyCurrentLLH(LLHOperator):
    """Возмущение Леви вокруг текущего решения"""

    def __init__(self, F: float = 0.8):
        super().__init__("levy_current", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        R = kwargs.get('R', 1.0)
        current_solution = population[current_index].copy()
        dim = current_solution.shape[0]
        beta = 1.5

        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (beta * math.gamma((1 + beta) / 2) * 2 ** ((beta - 1) / 2))) ** (1 / beta)

        u = np.random.normal(0, sigma, dim)
        v = np.random.normal(0, 1, dim)
        step = u / (np.abs(v) ** (1 / beta))

        return current_solution + R * step


class DERand1LLH(LLHOperator):
    """DE/rand/1 мутационный оператор"""

    def __init__(self, F: float = 0.8, crossover: CrossoverOperator = None):
        super().__init__("DE_rand_1", F)
        self.crossover = crossover if crossover else BinomialCrossover(Cr=0.1)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        n = len(population)

        indices = np.arange(n)
        mask = indices != current_index
        available_indices = indices[mask]
        r1, r2, r3 = np.random.choice(available_indices, 3, replace=False)

        mutant = population[r1] + self.F * (population[r2] - population[r3])
        trial = self.crossover.apply(population[r1], mutant)

        return trial


class DECurrent1LLH(LLHOperator):
    """DE/cur/1 мутационный оператор"""

    def __init__(self, F: float = 0.8, crossover: CrossoverOperator = None):
        super().__init__("DE_cur_1", F)
        self.crossover = crossover if crossover else BinomialCrossover(Cr=0.1)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        n = len(population)
        current_ind = population[current_index]

        indices = np.arange(n)
        mask = indices != current_index
        available_indices = indices[mask]
        r1, r2 = np.random.choice(available_indices, 2, replace=False)

        mutant = current_ind + self.F * (population[r1] - population[r2])
        trial = self.crossover.apply(current_ind, mutant)

        return trial


class DECurrentToBest1LLH(LLHOperator):
    """DE/cur-to-best/1 мутационный оператор"""

    def __init__(self, F: float = 0.8, crossover: CrossoverOperator = None):
        super().__init__("DE_cur_to_best_1", F)
        self.crossover = crossover if crossover else BinomialCrossover(Cr=0.1)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        n = len(population)
        current_ind = population[current_index]

        indices = np.arange(n)
        mask = indices != current_index
        available_indices = indices[mask]
        r1, r2 = np.random.choice(available_indices, 2, replace=False)

        mutant = current_ind + self.F * (best_solution - current_ind) + self.F * (population[r1] - population[r2])
        trial = self.crossover.apply(current_ind, mutant)

        return trial


class DECurrentToPBest1LLH(LLHOperator):
    """DE/cur-to-pbest/1 мутационный оператор"""

    def __init__(self, F: float = 0.8, crossover: CrossoverOperator = None):
        super().__init__("DE_cur_to_pbest_1", F)
        self.crossover = crossover if crossover else BinomialCrossover(Cr=0.1)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        n = len(population)
        current_ind = population[current_index]

        p = kwargs.get('p_best', 0.2)
        p_size = max(1, int(n * p))
        fitness = kwargs.get('fitness', None)

        if fitness is not None:
            best_indices = np.argpartition(fitness, p_size)[:p_size]
            pbest = population[np.random.choice(best_indices)]
        else:
            pbest = best_solution

        indices = np.arange(n)
        mask = indices != current_index
        available_indices = indices[mask]
        r1, r2 = np.random.choice(available_indices, 2, replace=False)

        mutant = current_ind + self.F * (pbest - current_ind) + self.F * (population[r1] - population[r2])
        trial = self.crossover.apply(current_ind, mutant)

        return trial


# ============================================================================
# УПРАВЛЕНИЕ ПУЛАМИ ОПЕРАТОРОВ
# ============================================================================

class LLHPoolManager:
    """Управление пулами операторов низкого уровня"""

    def __init__(self, config):
        self.config = config

        self.exploitation_pool = self._create_exploitation_pool()
        self.exploration_pool = self._create_exploration_pool()

    def _create_crossover(self, crossover_type: str) -> CrossoverOperator:
        """Создание оператора кроссовера"""
        if crossover_type == 'exponential':
            cr_value = self.config.crossover_config['exponential']['Cr']
            return ExponentialCrossover(Cr=cr_value)
        else:  # binomial
            cr_value = self.config.crossover_config['binomial']['Cr']
            return BinomialCrossover(Cr=cr_value)

    def _create_exploitation_pool(self) -> List[LLHOperator]:
        """Создание пула эксплуатирующих операторов"""
        crossover = self._create_crossover('binomial')
        Fs = self.config.exploitation_Fs

        return [
            UniformLLH(F=Fs.get('uniform', self.config.F_exploitation)),
            NormalLLH(F=Fs.get('normal', self.config.F_exploitation)),
            LevyLLH(F=Fs.get('levy', self.config.F_exploitation)),
            DEBest1LLH(F=Fs.get('DE_best_1', self.config.F_exploitation), crossover=crossover)
        ]

    def _create_exploration_pool(self) -> List[LLHOperator]:
        """Создание пула исследующих операторов"""
        crossover = self._create_crossover('binomial')
        Fs = self.config.exploration_Fs

        return [
            UniformCurrentLLH(F=Fs.get('uniform_current', self.config.F_exploration)),
            NormalCurrentLLH(F=Fs.get('normal_current', self.config.F_exploration)),
            LevyCurrentLLH(F=Fs.get('levy_current', self.config.F_exploration)),
            DERand1LLH(F=Fs.get('DE_rand_1', self.config.F_exploration), crossover=crossover),
            DECurrent1LLH(F=Fs.get('DE_cur_1', self.config.F_exploration), crossover=crossover),
            DECurrentToBest1LLH(F=Fs.get('DE_cur_to_best_1', self.config.F_exploration), crossover=crossover),
            DECurrentToPBest1LLH(F=Fs.get('DE_cur_to_pbest_1', self.config.F_exploration), crossover=crossover)
        ]

    def select_weighted_operator(self, pool: List[LLHOperator]) -> LLHOperator:
        """Выбор оператора с весами из конфигурации"""
        if hasattr(self.config, 'exploration_weights') and self.config.exploration_weights is not None:
            uses_exploration_weights = any(
                op.name in self.config.exploration_weights for op in pool
            )

            if uses_exploration_weights:
                weights = []
                for op in pool:
                    if op.name in self.config.exploration_weights:
                        weights.append(self.config.exploration_weights[op.name])
                    else:
                        weights.append(1.0)

                weights = np.array(weights) / np.sum(weights)
                return np.random.choice(pool, p=weights)

        return np.random.choice(pool)
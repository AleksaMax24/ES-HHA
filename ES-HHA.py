import numpy as np
import math
from abc import ABC, abstractmethod
from typing import List, Callable, Dict, Any, Optional, Tuple
from collections import defaultdict
import time
import json
import os
from dataclasses import dataclass, asdict, field
from enum import Enum


# операторы скрещивания
class CrossoverOperator(ABC):

    def __init__(self, name: str, Cr: float = 0.1):
        self.name = name
        self.Cr = Cr

    @abstractmethod
    def apply(self, target: np.ndarray, trial: np.ndarray) -> np.ndarray:
        pass


class BinomialCrossover(CrossoverOperator):

    def __init__(self, Cr: float = 0.1):
        super().__init__("binomial", Cr)

    def apply(self, target: np.ndarray, trial: np.ndarray) -> np.ndarray:
        D = len(target)
        j_rand = np.random.randint(0, D)
        mask = np.random.rand(D) < self.Cr
        mask[j_rand] = True
        return np.where(mask, trial, target)


class ExponentialCrossover(CrossoverOperator):

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



class LLHOperator(ABC):

    def __init__(self, name: str, F: float = 0.8):
        self.name = name
        self.F = F

    @abstractmethod
    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        pass


# Эксплуатирующие операторы
class UniformLLH(LLHOperator):

    def __init__(self, F: float = 0.8):
        super().__init__("uniform", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        R = kwargs.get('R', 1.0)
        dim = best_solution.shape[0]
        return best_solution + R * np.random.uniform(-1, 1, dim)


class NormalLLH(LLHOperator):

    def __init__(self, F: float = 0.8):
        super().__init__("normal", F)

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        R = kwargs.get('R', 1.0)
        dim = best_solution.shape[0]
        return best_solution + R * np.random.normal(0, 1, dim)


class LevyLLH(LLHOperator):

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


# Исследующие операторы
class UniformCurrentLLH(LLHOperator):

    def __init__(self, F: float = 0.8):
        super().__init__("uniform_current", F)  # другое имя для отличия

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        R = kwargs.get('R', 1.0)
        current_solution = population[current_index].copy()
        dim = current_solution.shape[0]

        return current_solution + R * np.random.uniform(-1, 1, dim)


class NormalCurrentLLH(LLHOperator):

    def __init__(self, F: float = 0.8):
        super().__init__("normal_current", F)  # другое имя для отличия

    def apply(self, population: np.ndarray, best_solution: np.ndarray,
              current_index: int, **kwargs) -> np.ndarray:
        R = kwargs.get('R', 1.0)
        current_solution = population[current_index].copy()
        dim = current_solution.shape[0]

        return current_solution + R * np.random.normal(0, 1, dim)


class LevyCurrentLLH(LLHOperator):

    def __init__(self, F: float = 0.8):
        super().__init__("levy_current", F)  # другое имя для отличия

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


class LLHPoolManager:

    def __init__(self, config):
        self.config = config

        self.exploitation_pool = self._create_exploitation_pool()
        self.exploration_pool = self._create_exploration_pool()

    def _create_crossover(self, crossover_type: str) -> CrossoverOperator:
        if crossover_type == 'exponential':
            cr_value = self.config.crossover_config['exponential']['Cr']
            return ExponentialCrossover(Cr=cr_value)
        else:  # binomial
            cr_value = self.config.crossover_config['binomial']['Cr']
            return BinomialCrossover(Cr=cr_value)

    def _create_exploitation_pool(self) -> List[LLHOperator]:
        crossover = self._create_crossover('binomial')
        Fs = self.config.exploitation_Fs

        return [
            UniformLLH(F=Fs.get('uniform', self.config.F_exploitation)),
            NormalLLH(F=Fs.get('normal', self.config.F_exploitation)),
            LevyLLH(F=Fs.get('levy', self.config.F_exploitation)),
            DEBest1LLH(F=Fs.get('DE_best_1', self.config.F_exploitation), crossover=crossover)
        ]

    def _create_exploration_pool(self) -> List[LLHOperator]:
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


# тестовые функции
class TestFunction(Enum):
    SPHERE = "sphere"
    RASTRIGIN = "rastrigin"
    ROSENBROCK = "rosenbrock"
    ACKLEY = "ackley"


def get_test_function(func_name: str) -> Tuple[Callable, np.ndarray, float, Tuple[float, float]]:
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


# класс хромосомы с параметрами
@dataclass
class ParameterChromosome:

    # Веса стратегий
    w1: float = 0.5
    w2: float = 0.3

    # Параметры операторов
    F_exploitation: float = 0.8
    F_exploration: float = 0.8
    Cr_binomial: float = 0.1
    Cr_exponential: float = 0.1

    # Параметры поиска
    R_exploitation: float = 1.0
    R_exploration: float = 1.0
    R_adaptation_rate: float = 0.5  #скорость адаптации

    # Параметры разнообразия
    p_best: float = 0.2
    diversity_threshold: float = 0.4
    fdc_threshold: float = 0.6

    # Параметры "встряски"
    shake_intensity: float = 0.1
    shake_threshold_improvements: int = 5
    shake_threshold_fdc: float = 0.1
    shake_threshold_pd: float = 0.01

    # Параметры поздней оптимизации
    late_stage_threshold: float = 0.7
    late_stage_diversity: float = 0.1

    # Параметры мутации для самих параметров
    mutation_rate: float = 0.1
    mutation_strength: float = 0.2

    def mutate(self, temperature: float = 1.0) -> 'ParameterChromosome':
        new_params = {}

        for field_name in self.__dataclass_fields__:
            current_value = getattr(self, field_name)

            if field_name in ['mutation_rate', 'mutation_strength']:
                new_params[field_name] = current_value
                continue

            if np.random.random() < self.mutation_rate * temperature:
                if isinstance(current_value, float):
                    delta = np.random.randn() * self.mutation_strength * temperature
                    new_value = current_value * (1 + delta)

                    if 'threshold' in field_name or 'rate' in field_name or 'p_' in field_name:
                        new_value = np.clip(new_value, 0.01, 0.99)
                    elif 'F_' in field_name:
                        new_value = np.clip(new_value, 0.1, 2.0)
                    elif 'R_' in field_name:
                        new_value = max(0.1, new_value)
                    elif 'Cr_' in field_name:
                        new_value = np.clip(new_value, 0.0, 1.0)
                    elif 'w' in field_name:
                        new_value = np.clip(new_value, 0.0, 1.0)

                    new_params[field_name] = new_value
                elif isinstance(current_value, int):
                    delta = int(np.random.randn() * max(1, current_value * self.mutation_strength))
                    new_value = max(1, current_value + delta)
                    new_params[field_name] = new_value
                else:
                    new_params[field_name] = current_value
            else:
                new_params[field_name] = current_value

        return ParameterChromosome(**new_params)

    def crossover(self, other: 'ParameterChromosome') -> 'ParameterChromosome':
        new_params = {}

        for field_name in self.__dataclass_fields__:
            if np.random.random() < 0.5:
                new_params[field_name] = getattr(self, field_name)
            else:
                new_params[field_name] = getattr(other, field_name)

        return ParameterChromosome(**new_params)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, (np.integer, np.floating)):
                result[field_name] = float(value)
            else:
                result[field_name] = value
        return result

    @classmethod
    def create_random(cls, bounds: Dict[str, Tuple[float, float]] = None) -> 'ParameterChromosome':
        if bounds is None:
            bounds = {
                'w1': (0.1, 0.9),
                'w2': (0.1, 0.7),
                'F_exploitation': (0.3, 1.5),
                'F_exploration': (0.3, 1.5),
                'Cr_binomial': (0.05, 0.9),
                'Cr_exponential': (0.05, 0.9),
                'R_exploitation': (0.3, 2.0),
                'R_exploration': (0.3, 2.0),
                'R_adaptation_rate': (0.2, 0.8),
                'p_best': (0.05, 0.4),
                'diversity_threshold': (0.2, 0.6),
                'fdc_threshold': (0.4, 0.8),
                'shake_intensity': (0.05, 0.3),
                'shake_threshold_improvements': (2, 10),
                'shake_threshold_fdc': (0.05, 0.3),
                'shake_threshold_pd': (0.005, 0.05),
                'late_stage_threshold': (0.5, 0.9),
                'late_stage_diversity': (0.05, 0.3),
                'mutation_rate': (0.05, 0.2),
                'mutation_strength': (0.1, 0.4)
            }

        params = {}
        for field_name, (low, high) in bounds.items():
            if field_name in ['shake_threshold_improvements']:
                params[field_name] = np.random.randint(int(low), int(high) + 1)
            else:
                params[field_name] = np.random.uniform(low, high)

        return cls(**params)


# класс конфигурации
@dataclass
class ES_HHA_Config:

    population_size: int = 100
    dimensions: int = 30
    max_FEs: int = 10000

    w1: float = 0.5
    fdc_threshold: float = 0.6
    pd_threshold: float = 0.4

    F_exploitation: float = 0.8
    F_exploration: float = 0.8
    R_exploitation: float = 1.0
    R_exploration: float = 1.0
    Cr_binomial: float = 0.1
    Cr_exponential: float = 0.1
    R_adaptation_rate: float = 0.5
    p_best: float = 0.2

    shake_intensity: float = 0.1
    shake_threshold_improvements: int = 5
    shake_threshold_fdc: float = 0.1
    shake_threshold_pd: float = 0.01
    late_stage_threshold: float = 0.7
    late_stage_diversity: float = 0.1

    lb_init: float = -100.0
    ub_init: float = 100.0
    lb_opt: float = -100.0
    ub_opt: float = 100.0

    constraint_method: str = 'tanh'
    constraint_steepness: float = 1.0

    exploitation_Fs: Optional[Dict[str, float]] = None
    exploration_Fs: Optional[Dict[str, float]] = None
    crossover_config: Optional[Dict[str, Dict]] = None

    verbose: bool = True
    detailed_log: bool = False
    log_interval: int = 10
    save_history: bool = True

    global_optimum: Optional[np.ndarray] = None
    test_function_name: str = "unknown"
    exploration_weights: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.exploitation_Fs is None:
            self.exploitation_Fs = {
                'uniform': self.F_exploitation,
                'normal': self.F_exploitation,
                'levy': self.F_exploitation,
                'DE_best_1': self.F_exploitation
            }

        if self.exploration_Fs is None:
            self.exploration_Fs = {
                'uniform_current': self.F_exploration,
                'normal_current': self.F_exploration,
                'levy_current': self.F_exploration,
                'DE_rand_1': self.F_exploration,
                'DE_cur_1': self.F_exploration,
                'DE_cur_to_best_1': self.F_exploration,
                'DE_cur_to_pbest_1': self.F_exploration
            }

        if self.crossover_config is None:
            self.crossover_config = {
                'binomial': {'Cr': self.Cr_binomial},
                'exponential': {'Cr': self.Cr_exponential}
            }
        if self.exploration_weights is None:
            self.exploration_weights = {
                'uniform_current': 0.25 / 3,  # ~8.33%
                'normal_current': 0.25 / 3,  # ~8.33%
                'levy_current': 0.25 / 3,  # ~8.33%

                'DE_rand_1': 0.75 / 4,  # ~18.75%
                'DE_cur_1': 0.75 / 4,  # ~18.75%
                'DE_cur_to_best_1': 0.75 / 4,  # ~18.75%
                'DE_cur_to_pbest_1': 0.75 / 4  # ~18.75%
            }



    def update_from_chromosome(self, chromosome: ParameterChromosome):
        self.w1 = chromosome.w1
        self.fdc_threshold = chromosome.fdc_threshold
        self.pd_threshold = chromosome.diversity_threshold
        self.F_exploitation = chromosome.F_exploitation
        self.F_exploration = chromosome.F_exploration
        self.p_best = chromosome.p_best
        self.R_exploitation = chromosome.R_exploitation
        self.R_exploration = chromosome.R_exploration
        self.Cr_binomial = chromosome.Cr_binomial
        self.Cr_exponential = chromosome.Cr_exponential
        self.R_adaptation_rate = chromosome.R_adaptation_rate

        self.shake_intensity = chromosome.shake_intensity
        self.shake_threshold_improvements = chromosome.shake_threshold_improvements
        self.shake_threshold_fdc = chromosome.shake_threshold_fdc
        self.shake_threshold_pd = chromosome.shake_threshold_pd
        self.late_stage_threshold = chromosome.late_stage_threshold
        self.late_stage_diversity = chromosome.late_stage_diversity
        self.R_adaptation_rate = chromosome.R_adaptation_rate

        self.exploitation_Fs = {
            'uniform': self.F_exploitation,
            'normal': self.F_exploitation,
            'levy': self.F_exploitation,
            'DE_best_1': self.F_exploitation
        }
        self.exploration_Fs = {
            'uniform_current': self.F_exploration,
            'normal_current': self.F_exploration,
            'levy_current': self.F_exploration,
            'DE_rand_1': self.F_exploration,
            'DE_cur_1': self.F_exploration,
            'DE_cur_to_best_1': self.F_exploration,
            'DE_cur_to_pbest_1': self.F_exploration
        }
        self.crossover_config = {
            'binomial': {'Cr': self.Cr_binomial},
            'exponential': {'Cr': self.Cr_exponential}
        }

    def save_to_file(self, filename: str):
        config_dict = asdict(self)

        def convert_numpy(obj):
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
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, np.generic):
                return obj.item()
            else:
                return obj

        config_dict = convert_numpy(config_dict)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filename: str) -> 'ES_HHA_Config':
        with open(filename, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        if config_dict.get('global_optimum') is not None:
            config_dict['global_optimum'] = np.array(config_dict['global_optimum'])

        return cls(**config_dict)


class ParameterOptimizer:
    # Оптимизатор параметров ES-HHA с использованием генетического алгоритма

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
        self.population = []
        for _ in range(self.population_size):
            if _ == 0:
                chromosome = ParameterChromosome()
            else:
                chromosome = ParameterChromosome.create_random()
            self.population.append(chromosome)

    def tournament_selection(self, k: int = 3) -> ParameterChromosome:
        # Турнирная селекция
        tournament = np.random.choice(len(self.population), k, replace=False)
        best_idx = tournament[0]
        best_fitness = self.fitnesses[best_idx]

        for idx in tournament[1:]:
            if self.fitnesses[idx] < best_fitness:
                best_idx = idx
                best_fitness = self.fitnesses[idx]

        return self.population[best_idx]

    def run(self) -> Tuple[ParameterChromosome, List[float]]:

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
                    print(f" New best! Fitness: {fitness:.4e}")

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
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        return self.best_chromosome, self.fitness_history


class ES_HHA:

    def __init__(self, objective_function: Callable, config: ES_HHA_Config):
        self.objective_function = objective_function
        self.config = config

        self.llh_usage_count = defaultdict(int)
        self.pool_usage_count = defaultdict(int)

        self.fdc_history = []
        self.pd_history = []
        self.best_fitness_history = []
        self.distance_to_optimum_history = []
        self.optimum_comparison_history = []
        self.improvements_history = []
        self.operator_success_history = []

        self.llh_manager = LLHPoolManager(config)
        self.exploitation_pool = self.llh_manager.exploitation_pool
        self.exploration_pool = self.llh_manager.exploration_pool

        self.initial_R_exploitation = config.R_exploitation
        self.initial_R_exploration = config.R_exploration
        self.initial_best_fitness = None

    def initialize_population(self):
        population = np.random.uniform(
            self.config.lb_init,
            self.config.ub_init,
            (self.config.population_size, self.config.dimensions)
        )
        fitness = self._evaluate_population(population)
        best_idx = np.argmin(fitness)

        return population, fitness, population[best_idx], fitness[best_idx]

    def _evaluate_population(self, population):
        fitness = np.zeros(self.config.population_size)
        for i in range(self.config.population_size):
            individual = population[i]
            constrained = self._apply_constraints(individual)
            fitness[i] = self.objective_function(constrained)
        return fitness

    def _apply_constraints(self, x: np.ndarray) -> np.ndarray:
        lb, ub = self.config.lb_opt, self.config.ub_opt
        method = self.config.constraint_method
        steepness = self.config.constraint_steepness

        if method == 'clip':
            return np.clip(x, lb, ub)

        elif method == 'sigmoid':
            center = (lb + ub) / 2
            scale = (ub - lb) / 2
            scale = np.maximum(scale, 1e-10)
            normalized = (x - center) / scale * (steepness / 2)
            sigmoid_val = 1 / (1 + np.exp(-normalized))
            return lb + sigmoid_val * (ub - lb)

        elif method == 'tanh':
            center = (lb + ub) / 2
            scale = (ub - lb) / 2
            tanh_val = np.tanh(x * steepness)
            return center + tanh_val * scale

        else:
            return np.clip(x, lb, ub)

    def calculate_FDC(self, population, fitness, best_solution):
        if self.config.global_optimum is not None:
            optimum = self.config.global_optimum
        else:
            optimum = best_solution

        differences = population - optimum
        distances = np.sqrt(np.einsum('ij,ij->i', differences, differences))

        f_mean, d_mean = np.mean(fitness), np.mean(distances)
        f_diff = fitness - f_mean
        d_diff = distances - d_mean

        numerator = np.dot(f_diff, d_diff)
        denominator = np.sqrt(np.dot(f_diff, f_diff) * np.dot(d_diff, d_diff))
        fdc_value = numerator / denominator if denominator != 0 else 0

        self.fdc_history.append(fdc_value)
        return fdc_value

    def calculate_PD(self, population):
        centroid = np.mean(population, axis=0)
        ranges = self.config.ub_opt - self.config.lb_opt
        normalized_diff = np.abs(population - centroid) / ranges
        pd_value = np.mean(normalized_diff)

        self.pd_history.append(pd_value)
        return pd_value

    def calculate_distance_to_optimum(self, solution):
        if self.config.global_optimum is not None:
            distance = np.linalg.norm(solution - self.config.global_optimum)
            self.distance_to_optimum_history.append(distance)
            return distance
        return None

    def compare_with_optimum(self, solution, fitness):
        if self.config.global_optimum is not None:
            optimum_fitness = self.objective_function(self.config.global_optimum)
            comparison = {
                'current_fitness': fitness,
                'optimum_fitness': optimum_fitness,
                'fitness_difference': fitness - optimum_fitness,
                'distance_to_optimum': np.linalg.norm(solution - self.config.global_optimum),
                'is_optimal': np.allclose(solution, self.config.global_optimum, atol=1e-6)
            }
            self.optimum_comparison_history.append(comparison)
            return comparison
        return None

    def select_LLH(self, FDC: float, PD: float) -> tuple:
        FDC_norm = (FDC + 1) / 2
        PD_norm = PD

        FDC_high = FDC_norm > self.config.fdc_threshold
        PD_high = PD_norm > self.config.pd_threshold

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
            pool = self.exploitation_pool
            pool_type = "exploitation"
            selection_info['case'] = 'a'
            selection_info['pool_type'] = pool_type

        elif FDC_high and not PD_high:
            P_exploit = self.config.w1 * FDC_norm + (1 - self.config.w1) * (1 - PD_norm)
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
            P_exploit = self.config.w1 * PD_norm + (1 - self.config.w1) * (1 - FDC_norm)
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
            pool = self.exploration_pool
            pool_type = "exploration"
            selection_info['case'] = 'd'
            selection_info['pool_type'] = pool_type

        selected_operator = self._select_weighted_operator(pool)

        self.llh_usage_count[selected_operator.name] += 1
        self.pool_usage_count[pool_type] += 1

        return selected_operator, selection_info

    def _select_weighted_operator(self, pool: List[LLHOperator]) -> LLHOperator:

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
    def apply_LLH(self, operator: LLHOperator, population: np.ndarray,
                  best_solution: np.ndarray, current_index: int,
                  fitness: np.ndarray = None, pool_type: str = None) -> np.ndarray:
        kwargs = {
            'p_best': self.config.p_best
        }

        if pool_type and ('exploitation' in pool_type):
            kwargs['R'] = self.config.R_exploitation
        else:
            kwargs['R'] = self.config.R_exploration

        if fitness is not None:
            kwargs['fitness'] = fitness

        return operator.apply(population, best_solution, current_index, **kwargs)

    def create_new_generation(self, population: np.ndarray, fitness: np.ndarray,
                              best_solution: np.ndarray, best_fitness: float,
                              improvements: int = 0, iteration: int = 0) -> tuple:
        FDC = self.calculate_FDC(population, fitness, best_solution)
        PD = self.calculate_PD(population)

        if self.initial_best_fitness is None:
            self.initial_best_fitness = best_fitness

        if self.initial_best_fitness > 0:
            progress = 1 - (best_fitness / self.initial_best_fitness)
            self.config.R_exploitation = max(0.1, self.initial_R_exploitation *
                                             (1 - progress * self.config.R_adaptation_rate))
            self.config.R_exploration = max(0.1, self.initial_R_exploration *
                                            (1 - progress * self.config.R_adaptation_rate))

        # Механизм "встряски"
        if (abs(FDC) < self.config.shake_threshold_fdc and
                PD < self.config.shake_threshold_pd and
                improvements < self.config.shake_threshold_improvements):

            if not hasattr(self, '_original_w1'):
                self._original_w1 = self.config.w1
            self.config.w1 = 0.3
            if self.config.verbose and iteration % 10 == 0:
                print(f" Застревание! w1: {self._original_w1} → 0.3")
        else:
            if hasattr(self, '_original_w1'):
                self.config.w1 = self._original_w1
                delattr(self, '_original_w1')

        new_population = np.empty_like(population)
        iteration_llh_info = []

        for j in range(self.config.population_size):
            operator, selection_info = self.select_LLH(FDC, PD)
            new_individual = self.apply_LLH(operator, population, best_solution, j, fitness,
                                           selection_info['pool_type'])
            new_population[j] = new_individual
            iteration_llh_info.append({
                'operator': operator.name,
                'pool_type': selection_info['pool_type']
            })

        # Увеличение разнообразия в конце
        total_iterations = len(self.best_fitness_history)
        max_iterations = self.config.max_FEs / self.config.population_size

        if total_iterations > max_iterations * self.config.late_stage_threshold and best_fitness > 1.0:
            n_random = max(1, int(self.config.population_size * self.config.late_stage_diversity))
            random_individuals = np.random.uniform(
                self.config.lb_opt,
                self.config.ub_opt,
                (n_random, self.config.dimensions)
            )

            worst_indices = np.argsort(fitness)[-n_random:]
            new_population[worst_indices] = random_individuals

            if self.config.verbose and iteration % 10 == 0:
                print(f" Добавлено {n_random} случайных особей")

        new_fitness = self._evaluate_population(new_population)

        improvement_mask = new_fitness < fitness
        improvements_count = np.sum(improvement_mask)
        self.improvements_history.append(improvements_count)

        # Сохраняем информацию о том, какие операторы дали улучшения
        operator_success = defaultdict(int)
        for j, (op_info, improved) in enumerate(zip(iteration_llh_info, improvement_mask)):
            if improved:
                operator_success[op_info['operator']] += 1

        # Сохраняем в историю (создаем атрибут, если его нет)
        if not hasattr(self, 'operator_success_history'):
            self.operator_success_history = []
        self.operator_success_history.append(dict(operator_success))

        final_population = np.where(improvement_mask[:, None], new_population, population)
        final_fitness = np.where(improvement_mask, new_fitness, fitness)

        best_new_idx = np.argmin(new_fitness)
        if new_fitness[best_new_idx] < best_fitness:
            new_best_solution = new_population[best_new_idx].copy()
            new_best_fitness = new_fitness[best_new_idx]
        else:
            new_best_solution = best_solution.copy()
            new_best_fitness = best_fitness

        distance = self.calculate_distance_to_optimum(new_best_solution)
        comparison = self.compare_with_optimum(new_best_solution, new_best_fitness)
        self.best_fitness_history.append(new_best_fitness)

        return (final_population, final_fitness,
                new_best_solution, new_best_fitness, improvements_count,
                iteration_llh_info, distance, comparison)

    def print_iteration_info(self, iteration, best_fitness, improvements,
                             FDC, PD, llh_info, distance, comparison):
        if not self.config.verbose:
            return

        if iteration % self.config.log_interval == 0 or self.config.detailed_log:
            op_count = {}
            pool_count = {}
            for info in llh_info:
                op_count[info['operator']] = op_count.get(info['operator'], 0) + 1
                pool_count[info['pool_type']] = pool_count.get(info['pool_type'], 0) + 1

            print(f"\n--- Iteration {iteration} ---")
            print(f"Best fitness: {best_fitness:.6e}")
            print(f"Improvements: {improvements}/{self.config.population_size}")
            print(f"FDC: {FDC:.3f}, PD: {PD:.3f}")

            if distance is not None:
                print(f"Distance to optimum: {distance:.6f}")

            if comparison is not None:
                print(f"Fitness difference from optimum: {comparison['fitness_difference']:.6e}")
                if comparison['is_optimal']:
                    print("*** REACHED GLOBAL OPTIMUM ***")

            if self.config.population_size > 1000:
                top_ops = dict(sorted(op_count.items(), key=lambda x: x[1], reverse=True)[:3])
                top_pools = dict(sorted(pool_count.items(), key=lambda x: x[1], reverse=True)[:2])
                print(f"Top operators: {top_ops}")
                print(f"Top pools: {top_pools}")
            else:
                print(f"Operators used: {dict(op_count)}")
                print(f"Pool distribution: {dict(pool_count)}")

    def print_final_statistics(self, start_time, end_time):
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)

        total_llh_uses = sum(self.llh_usage_count.values())
        total_pool_uses = sum(self.pool_usage_count.values())
        total_iterations = len(self.best_fitness_history)
        total_fes = self.config.population_size * total_iterations

        print(f"\nAlgorithm execution time: {end_time - start_time:.2f} seconds")
        print(f"Total iterations: {total_iterations}")
        print(f"Final best fitness: {self.best_fitness_history[-1]:.6e}")
        print(f"Total FEs: {total_fes}")
        print(f"Population size: {self.config.population_size}")
        print(f"Test function: {self.config.test_function_name}")

        if self.config.global_optimum is not None and len(self.optimum_comparison_history) > 0:
            final_comparison = self.optimum_comparison_history[-1]
            print(f"\nOptimum Comparison:")
            print(f"  Global optimum fitness: {final_comparison['optimum_fitness']:.6e}")
            print(f"  Final fitness difference: {final_comparison['fitness_difference']:.6e}")
            print(f"  Final distance to optimum: {final_comparison['distance_to_optimum']:.6f}")
            print(f"  Reached global optimum: {final_comparison['is_optimal']}")

            if self.distance_to_optimum_history:
                min_distance = min(self.distance_to_optimum_history)
                print(f"  Minimum distance to optimum: {min_distance:.6f}")

        print(f"\nLLH Usage Statistics (top 5):")
        for llh, count in sorted(self.llh_usage_count.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / total_llh_uses) * 100 if total_llh_uses > 0 else 0
            print(f"  {llh}: {count} uses ({percentage:.1f}%)")

        print(f"\nPool Usage Statistics:")
        for pool, count in sorted(self.pool_usage_count.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_pool_uses) * 100 if total_pool_uses > 0 else 0
            print(f"  {pool}: {count} uses ({percentage:.1f}%)")

        print(f"\nPerformance Metrics:")
        print(f"  FEs per second: {total_fes / (end_time - start_time):.0f}")
        print(f"  Iterations per second: {total_iterations / (end_time - start_time):.2f}")

        print(f"\nConvergence Analysis:")
        initial_fitness = self.best_fitness_history[0]
        final_fitness = self.best_fitness_history[-1]
        improvement = initial_fitness - final_fitness
        print(f"  Initial fitness: {initial_fitness:.6e}")
        print(f"  Final fitness: {final_fitness:.6e}")
        print(f"  Total improvement: {improvement:.6e}")


    def calculate_improvement_rate(self): # скорость улучшения на разных этапах
        if len(self.best_fitness_history) < 2:
            return None

        rates = []
        # 5 этапов для анализа
        n_stages = 5
        stage_size = len(self.best_fitness_history) // n_stages

        for stage in range(n_stages):
            start_idx = stage * stage_size
            end_idx = min((stage + 1) * stage_size, len(self.best_fitness_history) - 1)

            if end_idx > start_idx:
                start_fitness = self.best_fitness_history[start_idx]
                end_fitness = self.best_fitness_history[end_idx]
                stage_improvement = start_fitness - end_fitness
                rate = stage_improvement / stage_size
                rates.append({
                    'stage': stage + 1,
                    'iterations': f"{start_idx}-{end_idx}",
                    'improvement': stage_improvement,
                    'rate_per_iter': rate
                })

        return rates

    def plot_ascii_convergence(self, width=50):
        # график сходимости
        if len(self.best_fitness_history) < 2:
            return "Недостаточно данных для графика"

        max_fitness = max(self.best_fitness_history)
        min_fitness = min(self.best_fitness_history)
        range_fitness = max_fitness - min_fitness

        if range_fitness == 0:
            return "Все значения одинаковы"

        step = max(1, len(self.best_fitness_history) // 20)
        indices = range(0, len(self.best_fitness_history), step)

        graph = ["\n📊 ASCII-график сходимости (fitness по итерациям):"]
        graph.append("   Fitness")
        graph.append("    ↑")

        for i, idx in enumerate(indices):
            fitness = self.best_fitness_history[idx]
            norm_pos = 1 - (fitness - min_fitness) / range_fitness
            bar_length = int(norm_pos * width)

            if i % 4 == 0:
                graph.append(f"{idx:4d} | {'█' * bar_length} {fitness:.2e}")
            else:
                graph.append(f"     | {'█' * bar_length}")

        graph.append(f"     +{'─' * width}→ Итерации")
        graph.append(f"     min:{min_fitness:.2e}  max:{max_fitness:.2e}")

        return '\n'.join(graph)

    def print_improvement_analysis(self):
        # вывод анализв скорости улучшения
        print("\n" + "=" * 60)
        print("📈 АНАЛИЗ СКОРОСТИ УЛУЧШЕНИЯ")
        print("=" * 60)

        # общая статистика
        initial = self.best_fitness_history[0]
        final = self.best_fitness_history[-1]
        total_improvement = initial - final
        total_iterations = len(self.best_fitness_history)

        print(f"\n📊 Общая статистика:")
        print(f"   Начальный fitness: {initial:.6e}")
        print(f"   Конечный fitness:  {final:.6e}")
        print(f"   Общее улучшение:   {total_improvement:.6e}")
        print(f"   Всего итераций:    {total_iterations}")
        print(f"   Средняя скорость:   {total_improvement / total_iterations:.6e} за итерацию")

        # скорость по этапам
        rates = self.calculate_improvement_rate()
        if rates:
            print(f"\n📈 Скорость улучшения по этапам:")
            print(f"   {'Этап':<6} {'Итерации':<12} {'Улучшение':<15} {'Скорость/итер':<15} {'Тренд':<10}")
            print(f"   {'─' * 60}")

            prev_rate = None
            for r in rates:
                if prev_rate is not None:
                    if r['rate_per_iter'] > prev_rate * 1.2:
                        trend = " ускоряется"
                    elif r['rate_per_iter'] < prev_rate * 0.8:
                        trend = " замедляется"
                    else:
                        trend = " стабильно"
                else:
                    trend = "начальный"

                print(
                    f"   {r['stage']:<6} {r['iterations']:<12} {r['improvement']:<15.2e} {r['rate_per_iter']:<15.2e} {trend}")
                prev_rate = r['rate_per_iter']

        # моменты наибольшего улучшения
        improvements = []
        for i in range(1, len(self.best_fitness_history)):
            imp = self.best_fitness_history[i - 1] - self.best_fitness_history[i]
            if imp > 0:
                improvements.append((i, imp))

        improvements.sort(key=lambda x: x[1], reverse=True)

        print(f"\n⚡ Топ-5 самых эффективных итераций:")
        for i, (iter_num, imp) in enumerate(improvements[:5]):
            print(f"   {i + 1}. Итерация {iter_num}: улучшение {imp:.6e}")

        # ASCII-график
        print(self.plot_ascii_convergence())

        # прогноз достижения оптимума
        if final > 0:
            recent_rate = rates[-1]['rate_per_iter'] if rates else total_improvement / total_iterations
            if recent_rate > 0:
                iterations_to_opt = final / recent_rate
                print(f"\n Прогноз:")
                print(f"   При текущей скорости ({recent_rate:.2e}/итер)")
                print(f"   до оптимума (0) осталось ≈ {iterations_to_opt:.0f} итераций")
                print(f"   или ≈ {iterations_to_opt * self.config.population_size:.0f} FEs")
        else:
            print(f"\n✨ Оптимум достигнут!")

        # эффективность операторов
        print(f"\n🎯 Эффективность операторов на разных этапах:")
        if len(self.improvements_history) >= total_iterations:
            # Связываем операторов с улучшениями
            # (это потребует дополнительного сбора данных в create_new_generation)
            pass

    def print_enhanced_statistics(self, start_time, end_time):
        self.print_final_statistics(start_time, end_time)
        self.print_improvement_analysis()

    def export_convergence_data(self, filename: str = None):
        data = {
            'iterations': list(range(len(self.best_fitness_history))),
            'fitness': [float(f) for f in self.best_fitness_history],
            'fdc': [float(f) for f in self.fdc_history],
            'pd': [float(p) for p in self.pd_history],
            'improvements': [int(i) for i in self.improvements_history],
            'operator_usage': dict(self.llh_usage_count),
            'pool_usage': dict(self.pool_usage_count),
            'config': asdict(self.config)
        }

        data['improvement_rates'] = []
        for i in range(1, len(self.best_fitness_history)):
            rate = self.best_fitness_history[i - 1] - self.best_fitness_history[i]
            data['improvement_rates'].append(float(rate))

        if hasattr(self, 'operator_success_history') and self.operator_success_history:
            data['operator_success'] = self.operator_success_history

        if filename is None:
            filename = f"convergence_data_{self.config.test_function_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"

        os.makedirs("convergence_data", exist_ok=True)
        filepath = os.path.join("convergence_data", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n сходимости сохранены в: {filepath}")
        return data
    def optimize(self):
        start_time = time.time()

        population, fitness, best_solution, best_fitness = self.initialize_population()
        current_FEs = self.config.population_size
        iteration = 0

        if self.config.global_optimum is not None:
            self.calculate_distance_to_optimum(best_solution)
            self.compare_with_optimum(best_solution, best_fitness)

        if self.config.verbose:
            print(f"\n{'=' * 60}")
            print(f"ES-HHA OPTIMIZATION STARTED")
            print(f"{'=' * 60}")
            print(f"Test function: {self.config.test_function_name}")
            print(f"Dimensions: {self.config.dimensions}")
            print(f"Population size: {self.config.population_size}")
            print(f"Max FEs: {self.config.max_FEs}")
            print(f"Initial best fitness: {best_fitness:.6e}")
            if self.config.global_optimum is not None:
                distance = np.linalg.norm(best_solution - self.config.global_optimum)
                print(f"Initial distance to optimum: {distance:.6f}")

        while current_FEs < self.config.max_FEs:
            (population, fitness, best_solution, best_fitness,
             improvements, llh_info, distance, comparison) = self.create_new_generation(
                population, fitness, best_solution, best_fitness,
                improvements=self.improvements_history[-1] if self.improvements_history else 0,
                iteration=iteration
            )

            current_FEs += self.config.population_size

            FDC = self.fdc_history[-1] if self.fdc_history else 0
            PD = self.pd_history[-1] if self.pd_history else 0
            self.print_iteration_info(iteration, best_fitness, improvements,
                                      FDC, PD, llh_info, distance, comparison)

            iteration += 1

        end_time = time.time()

        if self.config.verbose:
            # self.print_final_statistics(start_time, end_time)
            self.print_enhanced_statistics(start_time, end_time)

        results = {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'total_FEs': current_FEs,
            'iterations': iteration,
            'llh_usage': dict(self.llh_usage_count),
            'pool_usage': dict(self.pool_usage_count),
            'fdc_history': self.fdc_history,
            'pd_history': self.pd_history,
            'fitness_history': self.best_fitness_history,
            'distance_to_optimum_history': self.distance_to_optimum_history,
            'optimum_comparison_history': self.optimum_comparison_history,
            'execution_time': end_time - start_time,
            'config': asdict(self.config)
        }

        return results




def run_experiment(config: ES_HHA_Config) -> dict:


    test_func, optimum, opt_value, bounds = get_test_function(config.test_function_name)

    config.global_optimum = optimum
    config.lb_init = bounds[0]
    config.ub_init = bounds[1]
    config.lb_opt = bounds[0]
    config.ub_opt = bounds[1]

    optimizer = ES_HHA(test_func, config)
    results = optimizer.optimize()

    return results


def run_multiple_experiments(config: ES_HHA_Config, n_runs: int = 30, save_results: bool = True):

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
        results_dir = "experiment_results"
        os.makedirs(results_dir, exist_ok=True)

        config_filename = f"{results_dir}/config_{config.test_function_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
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

        results_summary = {
            'config': asdict(config),
            'n_runs': n_runs,
            'statistics': stats,
            'all_best_fitnesses': [float(f) for f in best_fitnesses],
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

        results_filename = f"{results_dir}/results_{config.test_function_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)

        csv_filename = f"{results_dir}/results_{config.test_function_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_filename, 'w', encoding='utf-8') as f:
            f.write("run,best_fitness,distance_to_optimum,iterations,execution_time\n")
            for i, res in enumerate(all_results):
                distance = res['distance_to_optimum_history'][-1] if res['distance_to_optimum_history'] else 0
                f.write(
                    f"{i + 1},{res['best_fitness']:.6e},{distance:.6f},{res['iterations']},{res['execution_time']:.2f}\n")

        print(f"\nResults saved to: {results_filename}")
        print(f"Config saved to: {config_filename}")
        print(f"CSV results saved to: {csv_filename}")

    return stats, all_results


def run_with_parameter_evolution():

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
                   base_results['best_fitness'] * 100)
    print(f"Amélioration: {improvement:.2f}%")

    print(f"\nMeilleurs paramètres trouvés:")
    for key, value in best_chromosome.to_dict().items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    return results, best_chromosome


if __name__ == "__main__":
    print("=" * 60)
    print("ES-HHA: Evolutionary Status Guided Hyper-Heuristic Algorithm")
    print("=" * 60)

    mode = "normal"

    if mode == "evolution":
        results, best_chromosome = run_with_parameter_evolution()
    else:

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

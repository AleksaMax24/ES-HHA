

import numpy as np
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Tuple, List

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
    R_adaptation_rate: float = 0.5

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


@dataclass
class ES_HHA_Config:

    # Основные параметры
    population_size: int = 100
    dimensions: int = 30
    max_FEs: int = 10000

    # Параметры параллельных вычислений
    use_parallel: bool = False  
    n_workers: int = -1  

    # Параметры высокоуровневого компонента
    w1: float = 0.5
    fdc_threshold: float = 0.6
    pd_threshold: float = 0.4

    # Параметры низкоуровневых эвристик
    F_exploitation: float = 0.8
    F_exploration: float = 0.8
    R_exploitation: float = 1.0
    R_exploration: float = 1.0
    Cr_binomial: float = 0.1
    Cr_exponential: float = 0.1
    R_adaptation_rate: float = 0.5
    p_best: float = 0.2

    # Параметры улучшений (встряска)
    shake_intensity: float = 0.1
    shake_threshold_improvements: int = 5
    shake_threshold_fdc: float = 0.1
    shake_threshold_pd: float = 0.01
    late_stage_threshold: float = 0.7
    late_stage_diversity: float = 0.1

    # Границы поиска
    lb_init: float = -100.0
    ub_init: float = 100.0
    lb_opt: float = -100.0
    ub_opt: float = 100.0

    # Метод обработки ограничений
    constraint_method: str = 'tanh'
    constraint_steepness: float = 1.0

    # Веса для операторов 
    exploitation_Fs: Optional[Dict[str, float]] = None
    exploration_Fs: Optional[Dict[str, float]] = None
    crossover_config: Optional[Dict[str, Dict]] = None
    exploration_weights: Optional[Dict[str, float]] = None

    # Логирование
    verbose: bool = True
    detailed_log: bool = False
    log_interval: int = 10
    save_history: bool = True

    # Информация о тестовой функции
    global_optimum: Optional[np.ndarray] = None
    test_function_name: str = "unknown"

    def __post_init__(self):

        if self.n_workers <= 0:
            import multiprocessing as mp
            self.n_workers = mp.cpu_count()

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
                'uniform_current': 0.25 / 3, 
                'normal_current': 0.25 / 3,
                'levy_current': 0.25 / 3,
                'DE_rand_1': 0.75 / 4, 
                'DE_cur_1': 0.75 / 4,
                'DE_cur_to_best_1': 0.75 / 4,
                'DE_cur_to_pbest_1': 0.75 / 4
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
        if config_dict.get('global_optimum') is not None:
            config_dict['global_optimum'] = np.array(config_dict['global_optimum'])

        return cls(**config_dict)

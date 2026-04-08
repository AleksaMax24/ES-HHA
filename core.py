import numpy as np
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import asdict
import multiprocessing as mp
from functools import partial

from operators import LLHOperator, LLHPoolManager
from config import ES_HHA_Config


def _evaluate_individual(x: np.ndarray,
                         lb: float,
                         ub: float,
                         objective_func: Callable) -> float:
    constrained = np.clip(x, lb, ub)
    return objective_func(constrained)


class ES_HHA:
    def __init__(self, objective_function: Callable, config: ES_HHA_Config):
        self.objective_function = objective_function
        self.config = config

        self.use_parallel = config.use_parallel
        self.n_workers = config.n_workers if config.n_workers > 0 else mp.cpu_count()
        self.pool = None
        self.lb = config.lb_opt
        self.ub = config.ub_opt

        self.llh_usage_count = defaultdict(int)
        self.pool_usage_count = defaultdict(int)

        self.fdc_history = []
        self.pd_history = []
        self.best_fitness_history = []
        self.distance_to_optimum_history = []
        self.optimum_comparison_history = []
        self.improvements_history = []
        self.operator_success_history = []

        self.best_solutions_history = []
        self.stagnation_counter = 0
        self.adaptive_cr = config.Cr_binomial
        self.adaptive_F = config.F_exploration

        self.llh_manager = LLHPoolManager(config)
        self.exploitation_pool = self.llh_manager.exploitation_pool
        self.exploration_pool = self.llh_manager.exploration_pool

        self.initial_R_exploitation = config.R_exploitation
        self.initial_R_exploration = config.R_exploration
        self.initial_best_fitness = None
        self.last_best_fitness = None
        self.no_improvement_count = 0

        self.fitness_window = []
        self.window_size = 20

    def __del__(self):
        if self.pool is not None:
            try:
                self.pool.close()
                self.pool.join()
            except:
                pass

    def _apply_constraints(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)

    def _evaluate_population_parallel(self, population: np.ndarray) -> np.ndarray:
        if self.pool is None:
            self.pool = mp.Pool(processes=self.n_workers)
        eval_func = partial(_evaluate_individual,
                            lb=self.lb,
                            ub=self.ub,
                            objective_func=self.objective_function)
        try:
            fitness = np.array(self.pool.map(eval_func, population))
        except Exception as e:
            print(f"Parallel evaluation failed: {e}, falling back to sequential")
            fitness = self._evaluate_population_sequential(population)
        return fitness

    def _evaluate_population_sequential(self, population: np.ndarray) -> np.ndarray:
        n = len(population)
        fitness = np.zeros(n)
        for i in range(n):
            constrained = self._apply_constraints(population[i])
            fitness[i] = self.objective_function(constrained)
        return fitness

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        if self.use_parallel and len(population) > 10:
            return self._evaluate_population_parallel(population)
        else:
            return self._evaluate_population_sequential(population)

    def calculate_FDC(self, population: np.ndarray, fitness: np.ndarray,
                      best_solution: np.ndarray) -> float:
        optimum = self.config.global_optimum if self.config.global_optimum is not None else best_solution
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

    def calculate_PD(self, population: np.ndarray) -> float:
        centroid = np.mean(population, axis=0)
        ranges = self.config.ub_opt - self.config.lb_opt
        ranges = np.maximum(ranges, 1e-10)
        normalized_diff = np.abs(population - centroid) / ranges
        pd_value = np.mean(normalized_diff)
        self.pd_history.append(pd_value)
        return pd_value

    def calculate_distance_to_optimum(self, solution: np.ndarray) -> Optional[float]:
        if self.config.global_optimum is not None:
            distance = np.linalg.norm(solution - self.config.global_optimum)
            self.distance_to_optimum_history.append(distance)
            return distance
        return None

    def compare_with_optimum(self, solution: np.ndarray, fitness: float) -> Optional[Dict]:
        if self.config.global_optimum is not None:
            opt = self.config.global_optimum
            if np.isscalar(opt):
                opt = np.full(self.config.dimensions, opt)
            optimum_fitness = self.objective_function(opt)
            comparison = {
                'current_fitness': fitness,
                'optimum_fitness': optimum_fitness,
                'fitness_difference': fitness - optimum_fitness,
                'distance_to_optimum': np.linalg.norm(solution - opt),
                'is_optimal': np.allclose(solution, opt, atol=1e-6)
            }
            self.optimum_comparison_history.append(comparison)
            return comparison
        return None

    def select_LLH(self, FDC: float, PD: float) -> Tuple[LLHOperator, Dict]:
        FDC_norm = (FDC + 1) / 2
        PD_norm = np.clip(PD, 0, 1)
        FDC_high = FDC_norm > self.config.fdc_threshold
        PD_high = PD_norm > self.config.pd_threshold
        selection_info = {'FDC': FDC, 'PD': PD, 'FDC_norm': FDC_norm, 'PD_norm': PD_norm,
                          'FDC_high': FDC_high, 'PD_high': PD_high, 'case': None,
                          'P_exploit': None, 'pool_type': None}
        P_exploit = self.config.w1 * FDC_norm + (1 - self.config.w1) * PD_norm
        if FDC_high and PD_high:
            pool = self.exploitation_pool
            pool_type = "exploitation"
            selection_info['case'] = 'a'
        elif FDC_high and not PD_high:
            selection_info['P_exploit'] = P_exploit
            selection_info['case'] = 'b'
            if np.random.random() < P_exploit:
                pool = self.exploitation_pool
                pool_type = "balanced_exploitation"
            else:
                pool = self.exploration_pool
                pool_type = "balanced_exploration"
        elif not FDC_high and PD_high:
            selection_info['P_exploit'] = P_exploit
            selection_info['case'] = 'c'
            if np.random.random() < P_exploit:
                pool = self.exploitation_pool
                pool_type = "balanced_exploitation"
            else:
                pool = self.exploration_pool
                pool_type = "balanced_exploration"
        else:
            pool = self.exploration_pool
            pool_type = "exploration"
            selection_info['case'] = 'd'
        selection_info['pool_type'] = pool_type
        selected_operator = self.llh_manager.select_weighted_operator(pool)
        self.llh_usage_count[selected_operator.name] += 1
        self.pool_usage_count[pool_type] += 1
        return selected_operator, selection_info

    def apply_LLH(self, operator: LLHOperator, population: np.ndarray,
                  best_solution: np.ndarray, current_index: int,
                  fitness: np.ndarray = None, pool_type: str = None) -> np.ndarray:
        kwargs = {'p_best': self.config.p_best}
        if pool_type and ('exploitation' in pool_type):
            kwargs['R'] = self.config.R_exploitation
        else:
            kwargs['R'] = self.config.R_exploration
        if fitness is not None:
            kwargs['fitness'] = fitness
        if hasattr(operator, 'crossover') and operator.crossover is not None:
            operator.crossover.Cr = self.adaptive_cr
        if hasattr(operator, 'F'):
            operator.F = self.adaptive_F
        return operator.apply(population, best_solution, current_index, **kwargs)

    def initialize_population(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        population = np.random.uniform(self.config.lb_init, self.config.ub_init,
                                       (self.config.population_size, self.config.dimensions))
        fitness = self._evaluate_population(population)
        best_idx = np.argmin(fitness)
        return population, fitness, population[best_idx], fitness[best_idx]

    def _check_stagnation(self, best_fitness: float, iteration: int) -> Tuple[bool, bool]:
        self.fitness_window.append(best_fitness)
        if len(self.fitness_window) > self.window_size:
            self.fitness_window.pop(0)
        if self.last_best_fitness is not None:
            if best_fitness < self.last_best_fitness - 1e-8:
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
        self.last_best_fitness = best_fitness
        hard_stagnation = self.no_improvement_count > 30
        if len(self.fitness_window) >= self.window_size:
            improvement = self.fitness_window[0] - self.fitness_window[-1]
            soft_stagnation = improvement < 1e-4
        else:
            soft_stagnation = False
        need_restart = hard_stagnation or (soft_stagnation and self.no_improvement_count > 15)
        need_more_exploration = soft_stagnation or (self.no_improvement_count > 10)
        return need_restart, need_more_exploration

    def _restart_population(self, population: np.ndarray, fitness: np.ndarray,
                            best_solution: np.ndarray, best_fitness: float,
                            iteration: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if self.config.verbose:
            print(f"🚀 RESTARTING POPULATION at iteration {iteration}")
        n_elite = max(5, self.config.population_size // 20)
        elite_indices = np.argsort(fitness)[:n_elite]
        elite_population = population[elite_indices].copy()
        elite_fitness = fitness[elite_indices].copy()
        new_population = []
        new_fitness = []
        for i in range(n_elite):
            new_population.append(elite_population[i])
            new_fitness.append(elite_fitness[i])
        for i in range(n_elite):
            noise_scale = 5.0
            mutated = elite_population[i] + noise_scale * np.random.randn(self.config.dimensions)
            mutated = self._apply_constraints(mutated)
            new_population.append(mutated)
            new_fitness.append(self.objective_function(mutated))
        n_random = self.config.population_size - len(new_population)
        random_population = np.random.uniform(self.config.lb_opt, self.config.ub_opt,
                                              (n_random, self.config.dimensions))
        random_fitness = self._evaluate_population(random_population)
        new_population.extend(random_population)
        new_fitness.extend(random_fitness)
        self.adaptive_cr = min(0.9, self.adaptive_cr + 0.2)
        self.adaptive_F = min(1.5, self.adaptive_F + 0.2)
        self.no_improvement_count = 0
        self.fitness_window = []
        new_population = np.array(new_population)
        new_fitness = np.array(new_fitness)
        best_idx = np.argmin(new_fitness)
        new_best_solution = new_population[best_idx].copy()
        new_best_fitness = new_fitness[best_idx]
        return new_population, new_fitness, new_best_solution, new_best_fitness

    def create_new_generation(self, population: np.ndarray, fitness: np.ndarray,
                              best_solution: np.ndarray, best_fitness: float,
                              improvements: int = 0, iteration: int = 0) -> Tuple:
        FDC = self.calculate_FDC(population, fitness, best_solution)
        PD = self.calculate_PD(population)
        need_restart, need_more_exploration = self._check_stagnation(best_fitness, iteration)
        if need_restart:
            population, fitness, best_solution, best_fitness = self._restart_population(
                population, fitness, best_solution, best_fitness, iteration)
            distance = self.calculate_distance_to_optimum(best_solution)
            comparison = self.compare_with_optimum(best_solution, best_fitness)
            self.best_fitness_history.append(best_fitness)
            return (population, fitness, best_solution, best_fitness, 0,
                    [], distance, comparison)

        self.best_solutions_history.append(best_solution.copy())
        if len(self.best_solutions_history) > 10:
            self.best_solutions_history.pop(0)

        if self.initial_best_fitness is None:
            self.initial_best_fitness = best_fitness

        if self.initial_best_fitness > 0:
            progress = 1 - (best_fitness / self.initial_best_fitness)
            self.config.R_exploitation = max(0.05, self.initial_R_exploitation *
                                             (1 - progress * self.config.R_adaptation_rate))
            self.config.R_exploration = max(0.05, self.initial_R_exploration *
                                            (1 - progress * self.config.R_adaptation_rate))

        if len(self.best_fitness_history) > 1:
            improvement = self.best_fitness_history[-2] - best_fitness
            if improvement > 0:
                self.adaptive_cr = min(0.9, self.adaptive_cr + 0.03)
                self.adaptive_F = min(1.3, self.adaptive_F + 0.02)
            else:
                self.adaptive_cr = max(0.1, self.adaptive_cr - 0.02)
                self.adaptive_F = max(0.4, self.adaptive_F - 0.01)

        if need_more_exploration:
            self.config.w1 = min(0.4, self.config.w1 - 0.01)
        else:
            self.config.w1 = min(0.7, self.config.w1 + 0.005)

        new_population = np.empty_like(population)
        iteration_llh_info = []

        for j in range(self.config.population_size):
            if need_more_exploration and np.random.random() < 0.15:
                if np.random.random() < 0.6:
                    new_individual = best_solution + 3.0 * np.random.randn(self.config.dimensions)
                else:
                    new_individual = np.random.uniform(self.config.lb_opt, self.config.ub_opt,
                                                       self.config.dimensions)
                iteration_llh_info.append({'operator': 'exploration_boost', 'pool_type': 'boost'})
            else:
                operator, selection_info = self.select_LLH(FDC, PD)
                new_individual = self.apply_LLH(operator, population, best_solution, j, fitness,
                                                selection_info['pool_type'])
                iteration_llh_info.append({'operator': operator.name, 'pool_type': selection_info['pool_type']})
            new_individual = self._apply_constraints(new_individual)
            new_population[j] = new_individual

        new_fitness = self._evaluate_population(new_population)
        improvement_mask = new_fitness < fitness
        improvements_count = np.sum(improvement_mask)
        self.improvements_history.append(improvements_count)

        final_population = np.where(improvement_mask[:, None], new_population, population)
        final_fitness = np.where(improvement_mask, new_fitness, fitness)

        best_new_idx = np.argmin(new_fitness)
        if new_fitness[best_new_idx] < best_fitness:
            new_best_solution = new_population[best_new_idx].copy()
            new_best_fitness = new_fitness[best_new_idx]
            self.no_improvement_count = 0
        else:
            new_best_solution = best_solution.copy()
            new_best_fitness = best_fitness

        distance = self.calculate_distance_to_optimum(new_best_solution)
        comparison = self.compare_with_optimum(new_best_solution, new_best_fitness)
        self.best_fitness_history.append(new_best_fitness)

        return (final_population, final_fitness,
                new_best_solution, new_best_fitness, improvements_count,
                iteration_llh_info, distance, comparison)

    def optimize(self) -> Dict:
        start_time = time.time()
        population, fitness, best_solution, best_fitness = self.initialize_population()
        current_FEs = self.config.population_size
        iteration = 0

        if self.config.global_optimum is not None:
            self.calculate_distance_to_optimum(best_solution)
            self.compare_with_optimum(best_solution, best_fitness)

        if self.config.verbose:
            print(f"\n{'=' * 60}\nES-HHA OPTIMIZATION STARTED\n{'=' * 60}")
            print(f"Test function: {self.config.test_function_name}")
            print(f"Dimensions: {self.config.dimensions}")
            print(f"Population size: {self.config.population_size}")
            print(f"Max FEs: {self.config.max_FEs}")
            print(f"Parallel mode: {self.use_parallel}")
            if self.use_parallel:
                print(f"Workers: {self.n_workers}")
            print(f"Initial best fitness: {best_fitness:.6e}")

        while current_FEs < self.config.max_FEs:
            (population, fitness, best_solution, best_fitness,
             improvements, llh_info, distance, comparison) = self.create_new_generation(
                population, fitness, best_solution, best_fitness,
                improvements=self.improvements_history[-1] if self.improvements_history else 0,
                iteration=iteration)
            current_FEs += self.config.population_size
            if self.config.verbose and (iteration % self.config.log_interval == 0 or self.config.detailed_log):
                FDC = self.fdc_history[-1] if self.fdc_history else 0
                PD = self.pd_history[-1] if self.pd_history else 0
                print(f"\n--- Iteration {iteration} ---")
                print(f"Best fitness: {best_fitness:.6e}")
                print(f"Improvements: {improvements}/{self.config.population_size}")
                print(f"FDC: {FDC:.3f}, PD: {PD:.3f}")
                if distance is not None:
                    print(f"Distance to optimum: {distance:.6f}")
                if comparison is not None:
                    print(f"Fitness diff: {comparison['fitness_difference']:.6e}")
            iteration += 1

        end_time = time.time()
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

        if self.config.verbose:
            print("\n" + "=" * 60 + "\nFINAL STATISTICS\n" + "=" * 60)
            print(f"\n⏱️ Execution: {end_time - start_time:.2f}s")
            print(f"🎯 Final fitness: {self.best_fitness_history[-1]:.6e}")
            print(f"📊 Total FEs: {current_FEs}")
            if self.config.global_optimum is not None and self.optimum_comparison_history:
                final = self.optimum_comparison_history[-1]
                print(f"\n🏆 Gap to optimum: {final['fitness_difference']:.6e}")

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
            'distance_to_optimum_history': self.distance_to_optimum_history,
            'optimum_comparison_history': self.optimum_comparison_history,
            'execution_time': end_time - start_time,
            'config': asdict(self.config)
        }

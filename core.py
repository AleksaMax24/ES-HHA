

import numpy as np
import time
import json
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import asdict

from operators import LLHOperator, LLHPoolManager
from config import ES_HHA_Config


class ES_HHA:


    def __init__(self, objective_function: Callable, config: ES_HHA_Config):
        self.objective_function = objective_function
        self.config = config

        # Статистика использования операторов
        self.llh_usage_count = defaultdict(int)
        self.pool_usage_count = defaultdict(int)

        # История метрик и результатов
        self.fdc_history = []
        self.pd_history = []
        self.best_fitness_history = []
        self.distance_to_optimum_history = []
        self.optimum_comparison_history = []
        self.improvements_history = []
        self.operator_success_history = []

        # Управление пулами операторов
        self.llh_manager = LLHPoolManager(config)
        self.exploitation_pool = self.llh_manager.exploitation_pool
        self.exploration_pool = self.llh_manager.exploration_pool

        # Для адаптации радиуса поиска
        self.initial_R_exploitation = config.R_exploitation
        self.initial_R_exploration = config.R_exploration
        self.initial_best_fitness = None


    def initialize_population(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:

        population = np.random.uniform(
            self.config.lb_init,
            self.config.ub_init,
            (self.config.population_size, self.config.dimensions)
        )
        fitness = self._evaluate_population(population)
        best_idx = np.argmin(fitness)

        return population, fitness, population[best_idx], fitness[best_idx]

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:

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


    def calculate_FDC(self, population: np.ndarray, fitness: np.ndarray,
                      best_solution: np.ndarray) -> float:

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

    def calculate_PD(self, population: np.ndarray) -> float:

        centroid = np.mean(population, axis=0)
        ranges = self.config.ub_opt - self.config.lb_opt
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



    def select_LLH(self, FDC: float, PD: float) -> Tuple[LLHOperator, Dict]:

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

        # Случай (a): FDC высокий, PD высокий - только эксплуатация
        if FDC_high and PD_high:
            pool = self.exploitation_pool
            pool_type = "exploitation"
            selection_info['case'] = 'a'

        # Случай (b): FDC высокий, PD низкий - сбалансированный подход
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

        # Случай (c): FDC низкий, PD высокий - сбалансированный подход
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

        # Случай (d): FDC низкий, PD низкий - только исследование
        else:
            pool = self.exploration_pool
            pool_type = "exploration"
            selection_info['case'] = 'd'

        selection_info['pool_type'] = pool_type
        selected_operator = self.llh_manager.select_weighted_operator(pool)

        # Статистика
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

        return operator.apply(population, best_solution, current_index, **kwargs)



    def create_new_generation(self, population: np.ndarray, fitness: np.ndarray,
                              best_solution: np.ndarray, best_fitness: float,
                              improvements: int = 0, iteration: int = 0) -> Tuple:


        # Расчет метрик
        FDC = self.calculate_FDC(population, fitness, best_solution)
        PD = self.calculate_PD(population)

        # Адаптация радиуса поиска (аналог уменьшения шага)
        if self.initial_best_fitness is None:
            self.initial_best_fitness = best_fitness

        if self.initial_best_fitness > 0:
            progress = 1 - (best_fitness / self.initial_best_fitness)
            self.config.R_exploitation = max(0.1, self.initial_R_exploitation *
                                             (1 - progress * self.config.R_adaptation_rate))
            self.config.R_exploration = max(0.1, self.initial_R_exploration *
                                            (1 - progress * self.config.R_adaptation_rate))

        # Механизм "встряски" при застревании (локальный оптимум)
        if (abs(FDC) < self.config.shake_threshold_fdc and
                PD < self.config.shake_threshold_pd and
                improvements < self.config.shake_threshold_improvements):

            if not hasattr(self, '_original_w1'):
                self._original_w1 = self.config.w1
            self.config.w1 = 0.3
            if self.config.verbose and iteration % 10 == 0:
                print(f"⚠️ Застревание! w1: {self._original_w1:.2f} → 0.3")
        else:
            if hasattr(self, '_original_w1'):
                self.config.w1 = self._original_w1
                delattr(self, '_original_w1')

        # Создание нового поколения
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

        # Увеличение разнообразия на поздних этапах оптимизации
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
                print(f"🔄 Добавлено {n_random} случайных особей")

        # Оценка нового поколения
        new_fitness = self._evaluate_population(new_population)

        # Жадная селекция (уравнение 17 из статьи)
        improvement_mask = new_fitness < fitness
        improvements_count = np.sum(improvement_mask)
        self.improvements_history.append(improvements_count)

        # Сохранение информации об успешных операторах
        operator_success = defaultdict(int)
        for j, (op_info, improved) in enumerate(zip(iteration_llh_info, improvement_mask)):
            if improved:
                operator_success[op_info['operator']] += 1

        self.operator_success_history.append(dict(operator_success))

        # Формирование финальной популяции
        final_population = np.where(improvement_mask[:, None], new_population, population)
        final_fitness = np.where(improvement_mask, new_fitness, fitness)

        # Обновление лучшего решения
        best_new_idx = np.argmin(new_fitness)
        if new_fitness[best_new_idx] < best_fitness:
            new_best_solution = new_population[best_new_idx].copy()
            new_best_fitness = new_fitness[best_new_idx]
        else:
            new_best_solution = best_solution.copy()
            new_best_fitness = best_fitness

        # История
        distance = self.calculate_distance_to_optimum(new_best_solution)
        comparison = self.compare_with_optimum(new_best_solution, new_best_fitness)
        self.best_fitness_history.append(new_best_fitness)

        return (final_population, final_fitness,
                new_best_solution, new_best_fitness, improvements_count,
                iteration_llh_info, distance, comparison)



    def optimize(self) -> Dict:

        start_time = time.time()

        # Инициализация
        population, fitness, best_solution, best_fitness = self.initialize_population()
        current_FEs = self.config.population_size
        iteration = 0

        if self.config.global_optimum is not None:
            self.calculate_distance_to_optimum(best_solution)
            self.compare_with_optimum(best_solution, best_fitness)

        # Логирование начала
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

        # Основной цикл
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

            self._print_iteration_info(iteration, best_fitness, improvements,
                                       FDC, PD, llh_info, distance, comparison)

            iteration += 1

        end_time = time.time()

        if self.config.verbose:
            self._print_final_statistics(start_time, end_time)

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



    def _print_iteration_info(self, iteration, best_fitness, improvements,
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
                    print("✨ REACHED GLOBAL OPTIMUM ✨")

            if self.config.population_size > 1000:
                top_ops = dict(sorted(op_count.items(), key=lambda x: x[1], reverse=True)[:3])
                top_pools = dict(sorted(pool_count.items(), key=lambda x: x[1], reverse=True)[:2])
                print(f"Top operators: {top_ops}")
                print(f"Top pools: {top_pools}")
            else:
                print(f"Operators used: {dict(op_count)}")
                print(f"Pool distribution: {dict(pool_count)}")

    def _print_final_statistics(self, start_time, end_time):

        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)

        total_llh_uses = sum(self.llh_usage_count.values())
        total_pool_uses = sum(self.pool_usage_count.values())
        total_iterations = len(self.best_fitness_history)
        total_fes = self.config.population_size * total_iterations

        print(f"\n⏱️ Execution time: {end_time - start_time:.2f} seconds")
        print(f"🔄 Total iterations: {total_iterations}")
        print(f"🎯 Final best fitness: {self.best_fitness_history[-1]:.6e}")
        print(f"📊 Total FEs: {total_fes}")
        print(f"👥 Population size: {self.config.population_size}")
        print(f"🧪 Test function: {self.config.test_function_name}")

        if self.config.global_optimum is not None and len(self.optimum_comparison_history) > 0:
            final_comparison = self.optimum_comparison_history[-1]
            print(f"\n🏆 Optimum Comparison:")
            print(f"  Global optimum fitness: {final_comparison['optimum_fitness']:.6e}")
            print(f"  Final fitness difference: {final_comparison['fitness_difference']:.6e}")
            print(f"  Final distance to optimum: {final_comparison['distance_to_optimum']:.6f}")
            print(f"  Reached global optimum: {final_comparison['is_optimal']}")

        print(f"\n📈 LLH Usage Statistics (top 5):")
        for llh, count in sorted(self.llh_usage_count.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / total_llh_uses) * 100 if total_llh_uses > 0 else 0
            print(f"  {llh}: {count} uses ({percentage:.1f}%)")

        print(f"\n📊 Pool Usage Statistics:")
        for pool, count in sorted(self.pool_usage_count.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_pool_uses) * 100 if total_pool_uses > 0 else 0
            print(f"  {pool}: {count} uses ({percentage:.1f}%)")

        print(f"\n📈 Performance Metrics:")
        print(f"  FEs per second: {total_fes / (end_time - start_time):.0f}")
        print(f"  Iterations per second: {total_iterations / (end_time - start_time):.2f}")

        print(f"\n📉 Convergence Analysis:")
        initial_fitness = self.best_fitness_history[0]
        final_fitness = self.best_fitness_history[-1]
        improvement = initial_fitness - final_fitness
        print(f"  Initial fitness: {initial_fitness:.6e}")
        print(f"  Final fitness: {final_fitness:.6e}")
        print(f"  Total improvement: {improvement:.6e}")

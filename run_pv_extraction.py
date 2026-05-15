"""
run_pv_extraction.py
Запуск ES-HHA для извлечения параметров фотоэлектрических моделей
по методологии статьи Yan et al. 2021 (EJADE-D).
Используется алгоритм ES-HHA вместо EJADE.
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ES_HHA_Config
from core import ES_HHA
from pv_models import (
    create_rtc_single_diode_model,
    create_rtc_double_diode_model,
    create_stm6_module_model,
    create_stp6_module_model,
    DecomposedSDM,
    DecomposedDDM,
    DecomposedTDM,
    DecomposedModule,
    create_rtc_triple_diode_model,
    RTC_TEMPERATURE_K,
    RTC_VOLTAGE,
    RTC_CURRENT
)



# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ТАБЛИЦ И ГРАФИКОВ


def print_parameters_table(title, params_dict):
    """Печатает красивую таблицу параметров"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for key, value in params_dict.items():
        if 'Isd' in key or 'Is' in key:
            print(f"{key:8} = {value:.8e} A")
        elif 'Rp' in key or 'Rsh' in key:
            print(f"{key:8} = {value:.4f} Ohm")
        elif 'Rs' in key:
            print(f"{key:8} = {value:.8f} Ohm")
        elif 'n' in key:
            print(f"{key:8} = {value:.8f}")
        else:
            print(f"{key:8} = {value:.8f} A")
    print("=" * 60)


def plot_operator_usage(llh_usage, pool_usage, title="Operator Usage"):
    """Строит столбчатую диаграмму использования операторов LLH и физических пулов"""
    # Левый график: LLH операторы
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    names = list(llh_usage.keys())
    counts = list(llh_usage.values())
    ax1.bar(names, counts, color='skyblue')
    ax1.set_xlabel('Operator')
    ax1.set_ylabel('Number of uses')
    ax1.set_title('Low-Level Heuristics Usage')
    ax1.tick_params(axis='x', rotation=45)

    # Правый график: физические пулы (объединяем)
    # pool_usage содержит ключи: 'exploitation', 'balanced_exploitation', 'exploration', 'balanced_exploration'
    explo_total = pool_usage.get('exploitation', 0) + pool_usage.get('balanced_exploitation', 0)
    explor_total = pool_usage.get('exploration', 0) + pool_usage.get('balanced_exploration', 0)
    pool_names = ['Exploitation', 'Exploration']
    pool_counts = [explo_total, explor_total]

    ax2.bar(pool_names, pool_counts, color=['#1f77b4', '#ff7f0e'])
    ax2.set_xlabel('Pool type')
    ax2.set_ylabel('Number of uses')
    ax2.set_title('Physical Pool Usage')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_fitness_convergence(fitness_history, best_fitness, best_iteration):
    """Строит график сходимости и отмечает точку лучшего фитнеса"""
    plt.figure(figsize=(10, 6))
    iterations = range(len(fitness_history))
    plt.plot(iterations, fitness_history, 'b-', linewidth=1.5, label='Best fitness')
    plt.scatter(best_iteration, best_fitness, color='red', s=100, zorder=5,
                label=f'Best: {best_fitness:.2e} at iter {best_iteration}')
    plt.axhline(y=best_fitness, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE (log scale)')
    plt.yscale('log')
    plt.title('Convergence Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_pool_usage_dynamics(pool_usage_history, title="Pool Usage Dynamics"):
    """Строит график изменения доли использования физических пулов (Exploitation vs Exploration) по итерациям
       с учётом всех типов вызовов, включая 'boost'(теперь нету белых пятен, ура)."""
    if not pool_usage_history:
        print("No pool usage history to plot.")
        return

    iterations = range(len(pool_usage_history))
    exploitation_frac = []
    exploration_frac = []

    for it in pool_usage_history:
        total = sum(it.values())                     # общее число вызовов в итерации
        if total == 0:
            exploitation_frac.append(0)
            exploration_frac.append(0)
            continue
        explo = it.get('exploitation', 0) + it.get('balanced_exploitation', 0)
        # К исследованию относим исследовательские пулы + 'boost' (exploration_boost)
        explr = (it.get('exploration', 0) + it.get('balanced_exploration', 0) +
                 it.get('boost', 0))
        exploitation_frac.append(explo / total)
        exploration_frac.append(explr / total)

    plt.figure(figsize=(10, 6))
    plt.stackplot(iterations, exploitation_frac, exploration_frac,
                  labels=['Exploitation', 'Exploration'],
                  colors=['#1f77b4', '#ff7f0e'], alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Fraction of pool usage')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ФУНКЦИИ ЗАПУСКА ДЛЯ РАЗНЫХ МОДЕЛЕЙ


def run_single_diode_decomposed(verbose: bool = True, plot: bool = False, metric: str = 'rmse', use_soft_constraints: bool = False):
    """
    Извлечение параметров SDM с декомпозицией.
    Ожидаемый RMSE = 9.8602e-04
    """
    print("\n" + "=" * 60)
    print("SINGLE DIODE MODEL (decomposed) – Yan et al. 2021")
    print("=" * 60)

    model, data = create_rtc_single_diode_model()
    decomp_model = DecomposedSDM(data)

    def print_intermediate(iteration, best_sol, best_fit):
        if iteration % 10 == 0 and iteration > 0:
            n, Rs = best_sol
            Ipv, Isd, Rp = decomp_model.compute_linear_params(n, Rs)
            print(f"\n--- Iteration {iteration} (best fitness {best_fit:.6e}) ---")
            print(f"  n     = {n:.8f}")
            print(f"  Rs    = {Rs:.8f} Ohm")
            print(f"  Ipv   = {Ipv:.8f} A")
            print(f"  Isd   = {Isd:.8e} A")
            print(f"  Rp    = {Rp:.4f} Ohm")

    lb_nonlinear = np.array([1.0, 0.0])  # n, Rs
    ub_nonlinear = np.array([2.0, 0.5])

    config = ES_HHA_Config(
        population_size=50,
        dimensions=2,
        max_FEs=2000,
        w1=0.5,
        fdc_threshold=0.5,
        pd_threshold=0.3,
        F_exploitation=0.8,
        F_exploration=0.8,
        R_exploitation=0.1,
        R_exploration=0.3,
        Cr_binomial=0.5,
        Cr_exponential=0.5,
        p_best=0.2,
        R_adaptation_rate=0.5,
        verbose=verbose,
        test_function_name="SingleDiodeModel_Decomposed",
        lb_init=lb_nonlinear,
        ub_init=ub_nonlinear,
        lb_opt=lb_nonlinear,
        use_soft_constraints=use_soft_constraints,
        ub_opt=ub_nonlinear
    )

    if metric == 'rmse':
        objective = decomp_model.objective_decomposed
    elif metric == 'mape':
        objective = decomp_model.objective_decomposed_mape
    else:
        raise ValueError(f"Unknown metric {metric}")

    optimizer = ES_HHA(objective, config, iteration_callback=print_intermediate)
    results = optimizer.optimize()


    best_solution = results['best_solution']
    best_fitness = results['best_fitness']
    n, Rs = best_solution
    Ipv, Isd, Rp = decomp_model.compute_linear_params(n, Rs)

    fitness_hist = results.get('fitness_history', [])
    if fitness_hist:
        best_iter = np.argmin(fitness_hist)
        best_val = fitness_hist[best_iter]
    else:
        best_iter = 0
        best_val = best_fitness

    print_parameters_table("Extracted Parameters (SDM)", {
        'Ipv': Ipv, 'Isd': Isd, 'Rs': Rs, 'Rp': Rp, 'n': n
    })
    print(f"Best fitness achieved at iteration: {best_iter} (value: {best_val:.6e})")
    print(f"Total iterations: {len(fitness_hist) - 1 if fitness_hist else 0}")

    if plot:
        plot_operator_usage(results['llh_usage'], results['pool_usage'], title="SDM Operator Usage")
        plot_fitness_convergence(fitness_hist, best_val, best_iter)
        if 'pool_usage_history' in results:
            plot_pool_usage_dynamics(results['pool_usage_history'], title="SDM Pool Usage Dynamics")
        plot_fdc_pd_evolution(results['fdc_history'], results['pd_history'],
                              title="FDC and PD Evolution (SDM)")
        plot_fitness_statistics(results['fitness_history'],
                                results.get('mean_fitness_history', []),
                                results.get('max_fitness_history', []),
                                title="Fitness Statistics (SDM)")

    return {
        'Ipv': Ipv, 'Isd': Isd, 'Rs': Rs, 'Rp': Rp, 'n': n,
        'best_fitness': best_fitness, 'RMSE': best_fitness,
        'best_iteration': best_iter, 'history': results
    }


def run_double_diode_full(verbose: bool = True, plot: bool = False, metric: str = 'rmse'):
    """
    Извлечение 7 параметров DDM (полный поиск).
    Ожидаемый RMSE = 9.8248e-04
    """
    print("\n" + "=" * 60)
    print("DOUBLE DIODE MODEL (full search) – Yan et al. 2021")
    print("=" * 60)

    model, data = create_rtc_double_diode_model()

    def print_intermediate(iteration, best_sol, best_fit):
        if iteration % 10 == 0 and iteration > 0:
            Ipv, Isd1, Isd2, Rs, Rp, n1, n2 = best_sol
            print(f"\n--- Iteration {iteration} (best fitness {best_fit:.6e}) ---")
            print(f"  Ipv  = {Ipv:.8f} A")
            print(f"  Isd1 = {Isd1:.8e} A")
            print(f"  Isd2 = {Isd2:.8e} A")
            print(f"  Rs   = {Rs:.8f} Ohm")
            print(f"  Rp   = {Rp:.4f} Ohm")
            print(f"  n1   = {n1:.8f}")
            print(f"  n2   = {n2:.8f}")

    lb = np.array([0.5, 0.0, 0.0, 0.0, 10.0, 1.0, 1.0])
    ub = np.array([0.9, 5e-5, 5e-5, 0.5, 100.0, 2.0, 2.0])

    config = ES_HHA_Config(
        population_size=50,
        dimensions=7,
        max_FEs=4000,
        w1=0.5,
        fdc_threshold=0.5,
        pd_threshold=0.3,
        F_exploitation=0.8,
        F_exploration=0.8,
        R_exploitation=0.1,
        R_exploration=0.3,
        Cr_binomial=0.5,
        Cr_exponential=0.5,
        p_best=0.2,
        R_adaptation_rate=0.5,
        verbose=verbose,
        test_function_name="DoubleDiodeModel_Full",
        lb_init=lb,
        ub_init=ub,
        lb_opt=lb,
        ub_opt=ub,
        use_soft_constraints=use_soft_constraints
    )

    if metric == 'rmse':
        objective = model.objective
    elif metric == 'mape':
        if not hasattr(model, 'objective_mape'):
            raise AttributeError("DoubleDiodeModel has no 'objective_mape' method. Please add it to pv_models.py")
        objective = model.objective_mape
    else:
        raise ValueError(f"Unknown metric {metric}")

    optimizer = ES_HHA(objective, config, iteration_callback=print_intermediate)
    results = optimizer.optimize()

    best_solution = results['best_solution']
    best_fitness = results['best_fitness']
    Ipv, Isd1, Isd2, Rs, Rp, n1, n2 = best_solution

    fitness_hist = results.get('fitness_history', [])
    if fitness_hist:
        best_iter = np.argmin(fitness_hist)
        best_val = fitness_hist[best_iter]
    else:
        best_iter = 0
        best_val = best_fitness

    print_parameters_table("Extracted Parameters (DDM)", {
        'Ipv': Ipv, 'Isd1': Isd1, 'Isd2': Isd2, 'Rs': Rs, 'Rp': Rp, 'n1': n1, 'n2': n2
    })
    print(f"Best fitness achieved at iteration: {best_iter} (value: {best_val:.6e})")
    print(f"Total iterations: {len(fitness_hist) - 1 if fitness_hist else 0}")

    if plot:
        plot_operator_usage(results['llh_usage'], results['pool_usage'], title="DDM Full Operator Usage")
        plot_fitness_convergence(fitness_hist, best_val, best_iter)
        if 'pool_usage_history' in results:
            plot_pool_usage_dynamics(results['pool_usage_history'], title="SDM Pool Usage Dynamics")
        plot_fdc_pd_evolution(results['fdc_history'], results['pd_history'],
                              title="FDC and PD Evolution (SDM)")
        plot_fitness_statistics(results['fitness_history'],
                                results.get('mean_fitness_history', []),
                                results.get('max_fitness_history', []),
                                title="Fitness Statistics (SDM)")

    return {
        'Ipv': Ipv, 'Isd1': Isd1, 'Isd2': Isd2, 'Rs': Rs, 'Rp': Rp, 'n1': n1, 'n2': n2,
        'best_fitness': best_fitness, 'RMSE': best_fitness,
        'best_iteration': best_iter, 'history': results
    }


def run_triple_diode_decomposed(verbose: bool = True, plot: bool = False, metric: str = 'rmse', use_soft_constraints: bool = False):
    """
    Извлечение параметров TDM с декомпозицией (n1,n2,n3,Rs).
    Ожидаемый RMSE ~9.82e-04
    """
    print("\n" + "=" * 60)
    print("TRIPLE DIODE MODEL (decomposed) – 9 параметров")
    print("=" * 60)

    _, data = create_rtc_triple_diode_model()
    decomp = DecomposedTDM(data)

    def print_intermediate(iteration, best_sol, best_fit):
        if iteration % 10 == 0 and iteration > 0:
            n1, n2, n3, Rs = best_sol
            Ipv, Isd1, Isd2, Isd3, Rp = decomp.compute_linear_params(n1, n2, n3, Rs)
            print(f"\n--- Iteration {iteration} (best fitness {best_fit:.6e}) ---")
            print(f"  n1   = {n1:.8f}, n2 = {n2:.8f}, n3 = {n3:.8f}")
            print(f"  Rs   = {Rs:.8f} Ohm")
            print(f"  Ipv  = {Ipv:.8f} A")
            print(f"  Isd1 = {Isd1:.8e} A, Isd2 = {Isd2:.8e} A, Isd3 = {Isd3:.8e} A")
            print(f"  Rp   = {Rp:.4f} Ohm")

    lb = np.array([1.0, 1.0, 1.0, 0.0])  # n1, n2, n3, Rs
    ub = np.array([2.0, 2.0, 2.0, 0.5])

    config = ES_HHA_Config(
        population_size=100,
        dimensions=4,
        max_FEs=10000,
        w1=0.5,
        fdc_threshold=0.5,
        pd_threshold=0.3,
        F_exploitation=0.8,
        F_exploration=0.8,
        R_exploitation=0.1,
        R_exploration=0.3,
        Cr_binomial=0.5,
        Cr_exponential=0.5,
        p_best=0.2,
        R_adaptation_rate=0.5,
        verbose=verbose,
        test_function_name="TripleDiodeModel_Decomposed",
        lb_init=lb,
        ub_init=ub,
        lb_opt=lb,
        ub_opt=ub,
        use_soft_constraints=use_soft_constraints
    )

    if metric == 'rmse':
        objective = decomp.objective_decomposed
    elif metric == 'mape':
        if not hasattr(decomp, 'objective_decomposed_mape'):
            raise AttributeError(
                "DecomposedTDM has no 'objective_decomposed_mape' method. Please add it to pv_models.py")
        objective = decomp.objective_decomposed_mape
    else:
        raise ValueError(f"Unknown metric {metric}")

    optimizer = ES_HHA(objective, config, iteration_callback=print_intermediate)
    results = optimizer.optimize()

    best = results['best_solution']
    best_fitness = results['best_fitness']
    n1, n2, n3, Rs = best
    Ipv, Isd1, Isd2, Isd3, Rp = decomp.compute_linear_params(n1, n2, n3, Rs)

    fitness_hist = results.get('fitness_history', [])
    if fitness_hist:
        best_iter = np.argmin(fitness_hist)
        best_val = fitness_hist[best_iter]
    else:
        best_iter = 0
        best_val = best_fitness

    print_parameters_table("Extracted Parameters (TDM with decomposition)", {
        'Ipv': Ipv, 'Isd1': Isd1, 'Isd2': Isd2, 'Isd3': Isd3,
        'Rs': Rs, 'Rp': Rp, 'n1': n1, 'n2': n2, 'n3': n3
    })
    print(f"Best fitness achieved at iteration: {best_iter} (value: {best_val:.6e})")
    print(f"Total iterations: {len(fitness_hist) - 1 if fitness_hist else 0}")

    if plot:
        plot_operator_usage(results['llh_usage'], results['pool_usage'], title="TDM Operator Usage")
        plot_fitness_convergence(fitness_hist, best_val, best_iter)
        if 'pool_usage_history' in results:
            plot_pool_usage_dynamics(results['pool_usage_history'], title="SDM Pool Usage Dynamics")
        plot_fdc_pd_evolution(results['fdc_history'], results['pd_history'],
                              title="FDC and PD Evolution (SDM)")
        plot_fitness_statistics(results['fitness_history'],
                                results.get('mean_fitness_history', []),
                                results.get('max_fitness_history', []),
                                title="Fitness Statistics (SDM)")

    return {
        'Ipv': Ipv, 'Isd1': Isd1, 'Isd2': Isd2, 'Isd3': Isd3,
        'Rs': Rs, 'Rp': Rp, 'n1': n1, 'n2': n2, 'n3': n3,
        'best_fitness': best_fitness, 'RMSE': best_fitness,
        'best_iteration': best_iter, 'history': results
    }


def run_stm6_module_decomposed(verbose: bool = True, plot: bool = False, metric: str = 'rmse', use_soft_constraints: bool = False):
    """STM6-40/36 с декомпозицией (n, Rs) – ожидаемый RMSE 1.7298e-03"""
    print("\n" + "=" * 60)
    print("PV MODULE STM6-40/36 (decomposed) – Yan et al. 2021")
    print("=" * 60)

    _, data = create_stm6_module_model()
    decomp = DecomposedModule(data)

    def print_intermediate(iteration, best_sol, best_fit):
        if iteration % 10 == 0 and iteration > 0:
            n, Rs = best_sol
            Ipv, Isd, Rp, _ = decomp.get_parameters(n, Rs)
            print(f"\n--- Iteration {iteration} (best fitness {best_fit:.6e}) ---")
            print(f"  n     = {n:.8f}")
            print(f"  Rs    = {Rs:.8f} Ohm")
            print(f"  Ipv   = {Ipv:.8f} A")
            print(f"  Isd   = {Isd:.8e} A")
            print(f"  Rp    = {Rp:.4f} Ohm")

    lb = np.array([1.0, 0.0])
    ub = np.array([2.0, 0.36])

    config = ES_HHA_Config(
        population_size=50,
        dimensions=2,
        max_FEs=3000,
        w1=0.5,
        fdc_threshold=0.5,
        pd_threshold=0.3,
        F_exploitation=0.8,
        F_exploration=0.8,
        R_exploitation=0.1,
        R_exploration=0.3,
        Cr_binomial=0.5,
        Cr_exponential=0.5,
        p_best=0.2,
        R_adaptation_rate=0.5,
        verbose=verbose,
        test_function_name="STM6_Decomposed",
        lb_init=lb,
        ub_init=ub,
        lb_opt=lb,
        ub_opt=ub,
        use_soft_constraints=use_soft_constraints
    )
    if metric == 'rmse':
        objective = decomp.objective_decomposed
    elif metric == 'mape':
        if not hasattr(decomp, 'objective_decomposed_mape'):
            raise AttributeError(
                "DecomposedModule has no 'objective_decomposed_mape' method. Please add it to pv_models.py")
        objective = decomp.objective_decomposed_mape
    else:
        raise ValueError(f"Unknown metric {metric}")

    optimizer = ES_HHA(objective, config, iteration_callback=print_intermediate)
    results = optimizer.optimize()


    best_solution = results['best_solution']
    best_fitness = results['best_fitness']
    n, Rs = best_solution
    Ipv, Isd, Rp, _ = decomp.get_parameters(n, Rs)

    fitness_hist = results.get('fitness_history', [])
    if fitness_hist:
        best_iter = np.argmin(fitness_hist)
        best_val = fitness_hist[best_iter]
    else:
        best_iter = 0
        best_val = best_fitness

    print_parameters_table("Extracted Parameters (STM6-40/36)", {
        'Ipv': Ipv, 'Isd': Isd, 'Rs': Rs, 'Rp': Rp, 'n': n
    })
    print(f"Best fitness achieved at iteration: {best_iter} (value: {best_val:.6e})")
    print(f"Total iterations: {len(fitness_hist) - 1 if fitness_hist else 0}")

    if plot:
        plot_operator_usage(results['llh_usage'], results['pool_usage'], title="STM6 Operator Usage")
        plot_fitness_convergence(fitness_hist, best_val, best_iter)
        if 'pool_usage_history' in results:
            plot_pool_usage_dynamics(results['pool_usage_history'], title="SDM Pool Usage Dynamics")
        plot_fdc_pd_evolution(results['fdc_history'], results['pd_history'],
                              title="FDC and PD Evolution (SDM)")
        plot_fitness_statistics(results['fitness_history'],
                                results.get('mean_fitness_history', []),
                                results.get('max_fitness_history', []),
                                title="Fitness Statistics (SDM)")

    return {
        'Ipv': Ipv, 'Isd': Isd, 'Rs': Rs, 'Rp': Rp, 'n': n,
        'best_fitness': best_fitness, 'RMSE': best_fitness,
        'best_iteration': best_iter, 'history': results
    }


def run_stp6_module_decomposed(verbose: bool = True, plot: bool = False, metric: str = 'rmse', use_soft_constraints: bool = False):
    """STP6-120/36 с декомпозицией (n, Rs) – ожидаемый RMSE 1.6601e-02"""
    print("\n" + "=" * 60)
    print("PV MODULE STP6-120/36 (decomposed) – Yan et al. 2021")
    print("=" * 60)

    _, data = create_stp6_module_model()
    decomp = DecomposedModule(data)

    def print_intermediate(iteration, best_sol, best_fit):
        if iteration % 10 == 0 and iteration > 0:
            n, Rs = best_sol
            Ipv, Isd, Rp, _ = decomp.get_parameters(n, Rs)
            print(f"\n--- Iteration {iteration} (best fitness {best_fit:.6e}) ---")
            print(f"  n     = {n:.8f}")
            print(f"  Rs    = {Rs:.8f} Ohm")
            print(f"  Ipv   = {Ipv:.8f} A")
            print(f"  Isd   = {Isd:.8e} A")
            print(f"  Rp    = {Rp:.4f} Ohm")

    lb = np.array([1.0, 0.0])
    ub = np.array([2.0, 0.36])

    config = ES_HHA_Config(
        population_size=50,
        dimensions=2,
        max_FEs=7000,
        w1=0.5,
        fdc_threshold=0.5,
        pd_threshold=0.3,
        F_exploitation=0.8,
        F_exploration=0.8,
        R_exploitation=0.1,
        R_exploration=0.3,
        Cr_binomial=0.5,
        Cr_exponential=0.5,
        p_best=0.2,
        R_adaptation_rate=0.5,
        verbose=verbose,
        test_function_name="STP6_Decomposed",
        lb_init=lb,
        ub_init=ub,
        lb_opt=lb,
        ub_opt=ub,
        use_soft_constraints=use_soft_constraints
    )
    if metric == 'rmse':
        objective = decomp.objective_decomposed
    elif metric == 'mape':
        if not hasattr(decomp, 'objective_decomposed_mape'):
            raise AttributeError(
                "DecomposedModule has no 'objective_decomposed_mape' method. Please add it to pv_models.py")
        objective = decomp.objective_decomposed_mape
    else:
        raise ValueError(f"Unknown metric {metric}")

    optimizer = ES_HHA(objective, config, iteration_callback=print_intermediate)
    results = optimizer.optimize()


    best_solution = results['best_solution']
    best_fitness = results['best_fitness']
    n, Rs = best_solution
    Ipv, Isd, Rp, _ = decomp.get_parameters(n, Rs)

    fitness_hist = results.get('fitness_history', [])
    if fitness_hist:
        best_iter = np.argmin(fitness_hist)
        best_val = fitness_hist[best_iter]
    else:
        best_iter = 0
        best_val = best_fitness

    print_parameters_table("Extracted Parameters (STP6-120/36)", {
        'Ipv': Ipv, 'Isd': Isd, 'Rs': Rs, 'Rp': Rp, 'n': n
    })
    print(f"Best fitness achieved at iteration: {best_iter} (value: {best_val:.6e})")
    print(f"Total iterations: {len(fitness_hist) - 1 if fitness_hist else 0}")

    if plot:
        plot_operator_usage(results['llh_usage'], results['pool_usage'], title="STP6 Operator Usage")
        plot_fitness_convergence(fitness_hist, best_val, best_iter)
        if 'pool_usage_history' in results:
            plot_pool_usage_dynamics(results['pool_usage_history'], title="SDM Pool Usage Dynamics")
        plot_fdc_pd_evolution(results['fdc_history'], results['pd_history'],
                              title="FDC and PD Evolution (SDM)")
        plot_fitness_statistics(results['fitness_history'],
                                results.get('mean_fitness_history', []),
                                results.get('max_fitness_history', []),
                                title="Fitness Statistics (SDM)")

    return {
        'Ipv': Ipv, 'Isd': Isd, 'Rs': Rs, 'Rp': Rp, 'n': n,
        'best_fitness': best_fitness, 'RMSE': best_fitness,
        'best_iteration': best_iter, 'history': results
    }

def plot_fdc_pd_evolution(fdc_history, pd_history, title="FDC and PD Evolution"):
    # график изменения FDC и PD по итерациям
    if not fdc_history or not pd_history:
        print("No FDC/PD history to plot.")
        return
    iterations = range(len(fdc_history))
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, fdc_history, 'b-', label='FDC', linewidth=1.5)
    plt.plot(iterations, pd_history, 'r-', label='PD', linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_fitness_statistics(fitness_history, mean_fitness_history, max_fitness_history,
                            title="Fitness Statistics"):
    #график лучшего, среднего и худшего фитнеса по итерациям
    if not fitness_history:
        print("No fitness history to plot.")
        return
    min_len = min(len(fitness_history), len(mean_fitness_history), len(max_fitness_history))
    if min_len == 0:
        print("Not enough data to plot fitness statistics.")
        return

    fitness_history = fitness_history[:min_len]
    mean_fitness_history = mean_fitness_history[:min_len]
    max_fitness_history = max_fitness_history[:min_len]
    iterations = range(len(fitness_history))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, fitness_history, 'g-', label='Best (min)', linewidth=2)
    if mean_fitness_history:
        plt.plot(iterations, mean_fitness_history, 'b--', label='Mean', linewidth=1.5)
    if max_fitness_history:
        plt.plot(iterations, max_fitness_history, 'r:', label='Max (worst)', linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (log scale)')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_all_yan_experiments(plot: bool = False, metric: str = 'rmse', use_soft_constraints: bool = False):
    print("\n" + "=" * 70)
    print("RUNNING ALL EXPERIMENTS FROM YAN et al. 2021 (EJADE-D)")
    print("Using ES-HHA with decomposition for ALL models")
    print("=" * 70)

    results = {}
    results['SDM'] = run_single_diode_decomposed(verbose=True, plot=plot, metric=metric, use_soft_constraints=use_soft_constraints)
    results['DDM'] = run_double_diode_full(verbose=True, plot=plot, metric=metric, use_soft_constraints=use_soft_constraints)
    results['TDM'] = run_triple_diode_decomposed(verbose=True, plot=plot, metric=metric, use_soft_constraints=use_soft_constraints)
    results['STM6'] = run_stm6_module_decomposed(verbose=True, plot=plot, metric=metric, use_soft_constraints=use_soft_constraints)
    results['STP6'] = run_stp6_module_decomposed(verbose=True, plot=plot, metric=metric, use_soft_constraints=use_soft_constraints)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, res in results.items():
        fitness = res.get('best_fitness', res.get('RMSE'))
        print(f"{name:6} RMSE = {fitness:.6e}")
    return results



# ТОЧКА ВХОДА

if __name__ == "__main__":
    # режимы:
    # 'sdm_decomposed', 'ddm_full', 'tdm_decomposed', 'stm6_decomposed', 'stp6_decomposed', 'all'
    mode = "ddm_full"  # пример
    metric = "rmse" # rmse/mape
    use_soft_constraints = True # перевключение clip(False) и тангенс(True)

    if mode == "sdm_decomposed":
        run_single_diode_decomposed(verbose=True, plot=True, metric=metric)
    elif mode == "ddm_full":
        run_double_diode_full(verbose=True, plot=True, metric=metric)
    elif mode == "tdm_decomposed":
        run_triple_diode_decomposed(verbose=True, plot=True, metric=metric)
    elif mode == "stm6_decomposed":
        run_stm6_module_decomposed(verbose=True, plot=True, metric=metric)
    elif mode == "stp6_decomposed":
        run_stp6_module_decomposed(verbose=True, plot=True, metric=metric)
    elif mode == "all":
        run_all_yan_experiments(plot=True, metric=metric)
    else:
        print(f"Unknown mode: {mode}")
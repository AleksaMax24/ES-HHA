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


def print_parameters_table(title, params_dict):

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    names = list(llh_usage.keys())
    counts = list(llh_usage.values())
    ax1.bar(names, counts, color='skyblue')
    ax1.set_xlabel('Operator')
    ax1.set_ylabel('Number of uses')
    ax1.set_title('Low-Level Heuristics Usage')
    ax1.tick_params(axis='x', rotation=45)

    pool_names = list(pool_usage.keys())
    pool_counts = list(pool_usage.values())
    ax2.bar(pool_names, pool_counts, color='lightgreen')
    ax2.set_xlabel('Pool type')
    ax2.set_ylabel('Number of uses')
    ax2.set_title('Pool Usage')
    ax2.tick_params(axis='x', rotation=45)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_fitness_convergence(fitness_history, best_fitness, best_iteration):

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



def run_single_diode_decomposed(verbose: bool = True, plot: bool = False):

    print("\n" + "=" * 60)
    print("SINGLE DIODE MODEL (decomposed) – Yan et al. 2021")
    print("=" * 60)

    model, data = create_rtc_single_diode_model()
    decomp_model = DecomposedSDM(data)

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
        ub_opt=lb_nonlinear
    )

    optimizer = ES_HHA(decomp_model.objective_decomposed, config)
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

    return {
        'Ipv': Ipv, 'Isd': Isd, 'Rs': Rs, 'Rp': Rp, 'n': n,
        'best_fitness': best_fitness, 'RMSE': best_fitness,
        'best_iteration': best_iter, 'history': results
    }


def run_double_diode_full(verbose: bool = True, plot: bool = False):
    """
    Извлечение 7 параметров DDM (полный поиск).
    Ожидаемый RMSE = 9.8248e-04
    """
    print("\n" + "=" * 60)
    print("DOUBLE DIODE MODEL (full search) – Yan et al. 2021")
    print("=" * 60)

    model, data = create_rtc_double_diode_model()

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
        ub_opt=ub
    )

    optimizer = ES_HHA(model.objective, config)
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

    return {
        'Ipv': Ipv, 'Isd1': Isd1, 'Isd2': Isd2, 'Rs': Rs, 'Rp': Rp, 'n1': n1, 'n2': n2,
        'best_fitness': best_fitness, 'RMSE': best_fitness,
        'best_iteration': best_iter, 'history': results
    }


def run_triple_diode_decomposed(verbose: bool = True, plot: bool = False):
    """
    Извлечение параметров TDM с декомпозицией (n1,n2,n3,Rs).
    Ожидаемый RMSE ~9.82e-04
    """
    print("\n" + "=" * 60)
    print("TRIPLE DIODE MODEL (decomposed) – 9 параметров")
    print("=" * 60)

    _, data = create_rtc_triple_diode_model()
    decomp = DecomposedTDM(data)

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
        ub_opt=ub
    )

    optimizer = ES_HHA(decomp.objective_decomposed, config)
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

    return {
        'Ipv': Ipv, 'Isd1': Isd1, 'Isd2': Isd2, 'Isd3': Isd3,
        'Rs': Rs, 'Rp': Rp, 'n1': n1, 'n2': n2, 'n3': n3,
        'best_fitness': best_fitness, 'RMSE': best_fitness,
        'best_iteration': best_iter, 'history': results
    }


def run_stm6_module_decomposed(verbose: bool = True, plot: bool = False):
    """STM6-40/36 с декомпозицией (n, Rs) – ожидаемый RMSE 1.7298e-03"""
    print("\n" + "=" * 60)
    print("PV MODULE STM6-40/36 (decomposed) – Yan et al. 2021")
    print("=" * 60)

    _, data = create_stm6_module_model()
    decomp = DecomposedModule(data)

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
        ub_opt=ub
    )
    optimizer = ES_HHA(decomp.objective_decomposed, config)
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

    return {
        'Ipv': Ipv, 'Isd': Isd, 'Rs': Rs, 'Rp': Rp, 'n': n,
        'best_fitness': best_fitness, 'RMSE': best_fitness,
        'best_iteration': best_iter, 'history': results
    }


def run_stp6_module_decomposed(verbose: bool = True, plot: bool = False):
    """STP6-120/36 с декомпозицией (n, Rs) – ожидаемый RMSE 1.6601e-02"""
    print("\n" + "=" * 60)
    print("PV MODULE STP6-120/36 (decomposed) – Yan et al. 2021")
    print("=" * 60)

    _, data = create_stp6_module_model()
    decomp = DecomposedModule(data)

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
        ub_opt=ub
    )
    optimizer = ES_HHA(decomp.objective_decomposed, config)
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

    return {
        'Ipv': Ipv, 'Isd': Isd, 'Rs': Rs, 'Rp': Rp, 'n': n,
        'best_fitness': best_fitness, 'RMSE': best_fitness,
        'best_iteration': best_iter, 'history': results
    }


def run_all_yan_experiments(plot: bool = False):
    """Последовательно запускает все эксперименты (SDM, DDM, TDM, STM6, STP6)"""
    print("\n" + "=" * 70)
    print("RUNNING ALL EXPERIMENTS FROM YAN et al. 2021 (EJADE-D)")
    print("Using ES-HHA with decomposition for ALL models")
    print("=" * 70)

    results = {}
    results['SDM'] = run_single_diode_decomposed(verbose=True, plot=plot)
    results['DDM'] = run_double_diode_full(verbose=True, plot=plot)  # или run_double_diode_decomposed
    results['TDM'] = run_triple_diode_decomposed(verbose=True, plot=plot)
    results['STM6'] = run_stm6_module_decomposed(verbose=True, plot=plot)
    results['STP6'] = run_stp6_module_decomposed(verbose=True, plot=plot)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, res in results.items():
        fitness = res.get('best_fitness', res.get('RMSE'))
        print(f"{name:6} RMSE = {fitness:.6e}")
    return results


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================
if __name__ == "__main__":
    # Выберите режим:
    # 'sdm_decomposed', 'ddm_full', 'tdm_decomposed', 'stm6_decomposed', 'stp6_decomposed', 'all'
    mode = "all"  # пример

    if mode == "sdm_decomposed":
        run_single_diode_decomposed(verbose=True, plot=True)
    elif mode == "ddm_full":
        run_double_diode_full(verbose=True, plot=True)
    elif mode == "tdm_decomposed":
        run_triple_diode_decomposed(verbose=True, plot=True)
    elif mode == "stm6_decomposed":
        run_stm6_module_decomposed(verbose=True, plot=True)
    elif mode == "stp6_decomposed":
        run_stp6_module_decomposed(verbose=True, plot=True)
    elif mode == "all":
        run_all_yan_experiments(plot=True)
    else:
        print(f"Unknown mode: {mode}")

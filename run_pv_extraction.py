"""
run_pv_extraction.py
Запуск ES-HHA для извлечения параметров фотоэлектрических моделей
"""

import numpy as np
import sys
import os

# Добавляем пути к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ES_HHA_Config
from core import ES_HHA
from pv_models import (
    create_rtc_single_diode_model,
    create_rtc_double_diode_model,
    create_stm6_module_model,
    create_stp6_module_model,
    DecomposedSDM,
    RTC_TEMPERATURE_K,
    RTC_VOLTAGE,
    RTC_CURRENT
)


def run_single_diode_extraction(use_decomposition: bool = False, verbose: bool = True):
    """
    Запуск ES-HHA для извлечения параметров Single Diode Model

    Параметры для извлечения (SDM):
    - Ipv: фототок [0, 2] A
    - Isd: ток насыщения [0, 50e-6] A
    - Rs: последовательное сопротивление [0, 0.5] Ohm
    - Rp: шунтирующее сопротивление [0, 100] Ohm
    - n: фактор идеальности [1, 2]

    Статья сообщает оптимальное значение RMSE = 9.8602e-04
    """
    print("\n" + "=" * 60)
    print("SINGLE DIODE MODEL PARAMETER EXTRACTION")
    print("=" * 60)

    # Создаем модель с данными RTC France
    model, data = create_rtc_single_diode_model()

    # Определяем границы параметров
    lb = np.array([0.5, 0.0, 0.0, 10.0, 1.0])
    ub = np.array([0.9, 2e-5, 0.5, 100.0, 2.0])

    # Если используем декомпозицию, ищем только нелинейные параметры
    if use_decomposition:
        print("\nUsing search space decomposition (nonlinear params only: [n, Rs])")

        # Декомпозированная модель
        decomp_model = DecomposedSDM(data)

        # Границы для нелинейных параметров
        lb_nonlinear = np.array([1.0, 0.0])
        ub_nonlinear = np.array([2.0, 0.5])

        # Настройка конфигурации
        config = ES_HHA_Config(
            population_size=50,
            dimensions=2,
            max_FEs=5000,  # Увеличим FEs
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
            lb_init=np.array([1.0, 0.0]),  # n >= 1
            ub_init=np.array([2.0, 0.5]),
            lb_opt=np.array([1.0, 0.0]),
            ub_opt=np.array([2.0, 0.5])
        )

        # Запуск оптимизации
        optimizer = ES_HHA(decomp_model.objective_decomposed, config)
        results = optimizer.optimize()
        best_solution = results['best_solution']
        best_fitness = results['best_fitness']
        history = results  # или results['fitness_history']

        n, Rs = best_solution
        Ipv, Isd, Rp = decomp_model.compute_linear_params(n, Rs)

        print("\n" + "=" * 60)
        print("EXTRACTION RESULTS (with decomposition)")
        print("=" * 60)
        print(f"Extracted parameters:")
        print(f"  Ipv = {Ipv:.8f} A")
        print(f"  Isd = {Isd:.8e} A")
        print(f"  Rs  = {Rs:.8f} Ohm")
        print(f"  Rp  = {Rp:.4f} Ohm")
        print(f"  n   = {n:.8f}")
        print(f"  RMSE = {best_fitness:.6e}")

        # Сравнение с эталоном из статьи
        expected_rmse = 9.8602e-04
        if best_fitness <= expected_rmse * 1.01:
            print(f"\n✓ Result matches expected RMSE ({expected_rmse:.2e})")
        else:
            print(f"\n⚠ Result differs from expected ({expected_rmse:.2e})")

        return {
            'Ipv': Ipv, 'Isd': Isd, 'Rs': Rs, 'Rp': Rp, 'n': n,
            'RMSE': best_fitness,
            'history': history
        }

    else:
        print("\nUsing full parameter search (5 parameters)")

        # Настройка конфигурации для полного поиска
        config = ES_HHA_Config(
            population_size=50,
            dimensions=2,
            max_FEs=2000,
            w1=0.5,
            fdc_threshold=0.5,
            pd_threshold=0.3,
            F_exploitation=0.5,  # Уменьшен для стабильности
            F_exploration=0.5,
            R_exploitation=0.05,  # Маленький радиус для точной настройки
            R_exploration=0.2,
            Cr_binomial=0.5,
            Cr_exponential=0.5,
            p_best=0.2,
            R_adaptation_rate=0.3,
            verbose=verbose,
            test_function_name="SingleDiodeModel_Decomposed",
            lb_init=np.array([1.0, 0.0]),  # [n, Rs]
            ub_init=np.array([2.0, 0.5]),
            lb_opt=np.array([1.0, 0.0]),
            ub_opt=np.array([2.0, 0.5])
        )

        optimizer = ES_HHA(model.objective, config)
        best_solution, best_fitness, history = optimizer.optimize()

        Ipv, Isd, Rs, Rp, n = best_solution

        print("\n" + "=" * 60)
        print("EXTRACTION RESULTS (full search)")
        print("=" * 60)
        print(f"Extracted parameters:")
        print(f"  Ipv = {Ipv:.8f} A")
        print(f"  Isd = {Isd:.8e} A")
        print(f"  Rs  = {Rs:.8f} Ohm")
        print(f"  Rp  = {Rp:.4f} Ohm")
        print(f"  n   = {n:.8f}")
        print(f"  RMSE = {best_fitness:.6e}")

        return {
            'Ipv': Ipv, 'Isd': Isd, 'Rs': Rs, 'Rp': Rp, 'n': n,
            'RMSE': best_fitness,
            'history': history
        }


def run_double_diode_extraction(verbose: bool = True):
    """
    Запуск ES-HHA для извлечения параметров Double Diode Model

    Параметры для извлечения (DDM):
    - Ipv: фототок [0, 2] A
    - Isd1, Isd2: токи насыщения [0, 50e-6] A
    - Rs: последовательное сопротивление [0, 0.5] Ohm
    - Rp: шунтирующее сопротивление [0, 100] Ohm
    - n1, n2: факторы идеальности [1, 2]

    Статья сообщает оптимальное значение RMSE = 9.8248e-04
    """
    print("\n" + "=" * 60)
    print("DOUBLE DIODE MODEL PARAMETER EXTRACTION")
    print("=" * 60)

    model, data = create_rtc_double_diode_model()

    # Определяем границы параметров
    lb = np.array([0.5, 0.0, 0.0, 0.0, 10.0, 1.0, 1.0])
    ub = np.array([0.9, 2e-5, 2e-5, 0.5, 100.0, 2.0, 2.0])

    config = ES_HHA_Config(
        population_size=100,
        dimensions=7,
        max_FEs=4000,
        w1=0.5,
        fdc_threshold=0.5,
        pd_threshold=0.3,
        F_exploitation=0.8,
        F_exploration=0.8,
        R_exploitation=0.1,
        R_exploration=0.5,
        Cr_binomial=0.5,
        Cr_exponential=0.5,
        p_best=0.2,
        R_adaptation_rate=0.5,
        verbose=verbose,
        test_function_name="DoubleDiodeModel",
        lb_init=lb[0],
        ub_init=ub[6],
        lb_opt=lb,
        ub_opt=ub
    )

    optimizer = ES_HHA(model.objective, config)
    best_solution, best_fitness, history = optimizer.optimize()

    Ipv, Isd1, Isd2, Rs, Rp, n1, n2 = best_solution

    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)
    print(f"Extracted parameters:")
    print(f"  Ipv  = {Ipv:.8f} A")
    print(f"  Isd1 = {Isd1:.8e} A")
    print(f"  Isd2 = {Isd2:.8e} A")
    print(f"  Rs   = {Rs:.8f} Ohm")
    print(f"  Rp   = {Rp:.4f} Ohm")
    print(f"  n1   = {n1:.8f}")
    print(f"  n2   = {n2:.8f}")
    print(f"  RMSE = {best_fitness:.6e}")

    # Сравнение с эталоном из статьи
    expected_rmse = 9.8248e-04
    if best_fitness <= expected_rmse * 1.01:
        print(f"\n✓ Result matches expected RMSE ({expected_rmse:.2e})")
    else:
        print(f"\n⚠ Result differs from expected ({expected_rmse:.2e})")

    return {
        'Ipv': Ipv, 'Isd1': Isd1, 'Isd2': Isd2,
        'Rs': Rs, 'Rp': Rp, 'n1': n1, 'n2': n2,
        'RMSE': best_fitness,
        'history': history
    }


def run_module_extraction(module_type: str = "STM6", verbose: bool = True):
    """
    Запуск ES-HHA для извлечения параметров PV модуля

    module_type: "STM6" или "STP6"
    """
    print("\n" + "=" * 60)
    print(f"{module_type} PV MODULE PARAMETER EXTRACTION")
    print("=" * 60)

    if module_type == "STM6":
        model, data = create_stm6_module_model()
        max_fes = 3000
        expected_rmse = 1.7298e-03
    else:
        model, data = create_stp6_module_model()
        max_fes = 7000
        expected_rmse = 1.6601e-02

    # Границы параметров
    lb = np.array([0, 0, 0, 0, 1.0])
    ub = np.array([8, 5e-5, 0.36, 1500, 2.0])

    config = ES_HHA_Config(
        population_size=100,
        dimensions=5,
        max_FEs=max_fes,
        w1=0.5,
        fdc_threshold=0.5,
        pd_threshold=0.3,
        F_exploitation=0.8,
        F_exploration=0.8,
        R_exploitation=0.1,
        R_exploration=0.5,
        Cr_binomial=0.5,
        Cr_exponential=0.5,
        p_best=0.2,
        R_adaptation_rate=0.5,
        verbose=verbose,
        test_function_name=f"{module_type}_Module",
        lb_init=lb[0],
        ub_init=ub[4],
        lb_opt=lb,
        ub_opt=ub
    )

    optimizer = ES_HHA(model.objective, config)
    best_solution, best_fitness, history = optimizer.optimize()

    Ipv, Isd, Rs, Rp, n = best_solution

    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)
    print(f"Extracted parameters:")
    print(f"  Ipv = {Ipv:.8f} A")
    print(f"  Isd = {Isd:.8e} A")
    print(f"  Rs  = {Rs:.8f} Ohm")
    print(f"  Rp  = {Rp:.4f} Ohm")
    print(f"  n   = {n:.8f}")
    print(f"  RMSE = {best_fitness:.6e}")

    if best_fitness <= expected_rmse * 1.01:
        print(f"\n✓ Result matches expected RMSE ({expected_rmse:.2e})")
    else:
        print(f"\n⚠ Result differs from expected ({expected_rmse:.2e})")

    return {
        'Ipv': Ipv, 'Isd': Isd, 'Rs': Rs, 'Rp': Rp, 'n': n,
        'RMSE': best_fitness,
        'history': history
    }


def run_comparison_with_decomposition(verbose: bool = True):
    """
    Сравнение полного поиска и поиска с декомпозицией
    Как в статье Yan et al. 2021
    """
    print("\n" + "=" * 60)
    print("COMPARISON: FULL SEARCH vs DECOMPOSITION")
    print("=" * 60)

    # Полный поиск
    print("\n--- Full Search (5 parameters) ---")
    result_full = run_single_diode_extraction(use_decomposition=False, verbose=False)

    # Поиск с декомпозицией
    print("\n--- Decomposed Search (2 nonlinear parameters) ---")
    result_decomp = run_single_diode_extraction(use_decomposition=True, verbose=False)

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Full Search RMSE:     {result_full['RMSE']:.6e} (FEs=5000)")
    print(f"Decomposed RMSE:      {result_decomp['RMSE']:.6e} (FEs=2000)")
    print(f"Expected RMSE (paper): 9.8602e-04")

    improvement = (result_full['RMSE'] - result_decomp['RMSE']) / result_full['RMSE'] * 100
    print(f"\nImprovement: {improvement:.2f}%")

    return result_full, result_decomp


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ES-HHA FOR PHOTOVOLTAIC PARAMETER EXTRACTION")
    print("Based on: Yan et al. (2021) - Adaptive differential evolution with decomposition")
    print("=" * 60)

    # Выбор режима
    mode = "sdm_decomposed"  # sdm_full, sdm_decomposed, ddm, stm6, stp6, comparison

    if mode == "sdm_full":
        run_single_diode_extraction(use_decomposition=False, verbose=True)

    elif mode == "sdm_decomposed":
        run_single_diode_extraction(use_decomposition=True, verbose=True)

    elif mode == "ddm":
        run_double_diode_extraction(verbose=True)

    elif mode == "stm6":
        run_module_extraction("STM6", verbose=True)

    elif mode == "stp6":
        run_module_extraction("STP6", verbose=True)

    elif mode == "comparison":
        run_comparison_with_decomposition(verbose=True)

    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: sdm_full, sdm_decomposed, ddm, stm6, stp6, comparison")
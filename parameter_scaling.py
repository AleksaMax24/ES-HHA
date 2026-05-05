import numpy as np
import sys
import os


from pv_models import (
    SingleDiodeModuleModel, PVData,
    STP6_VOLTAGE, STP6_CURRENT, STP6_Ns, STP6_Np, STP6_TEMPERATURE_K,
    STM6_Ns, STM6_Np, STM6_TEMPERATURE_K,
    RTC_TEMPERATURE_K
)


# эталонные параметры (из литературы Yan et al. 2021)

params_sdm_cell = {
    'Ipv': 0.76077553,
    'Isd': 0.32302079e-6,
    'Rs':  0.03637709,
    'Rp':  53.71852020,
    'n':   1.48118359
}

# STM6 модуль (данные из Table 10)
params_stm6_module = {
    'Ipv': 1.66390478,        # A
    'Isd': 1.73865688e-6,     # A
    'Rs':  0.00427377,        # Ohm
    'Rp':  15.92829402,       # Ohm
    'n':   1.52030292
}

# STP6 модуль (эталон из Table 11)
params_stp6_module_ref = {
    'Ipv': 7.47252992,        # A
    'Isd': 2.33499500e-6,     # A
    'Rs':  0.00459463,        # Ohm
    'Rp':  22.21990200,       # Ohm
    'n':   1.26010348
}


# пересчёт параметров STM6 к ячейке

Ns_stm6 = STM6_Ns
Np_stm6 = STM6_Np
params_stm6_cell = {
    'Ipv': params_stm6_module['Ipv'] / Np_stm6,
    'Isd': params_stm6_module['Isd'] / Np_stm6,
    'Rs':  params_stm6_module['Rs'] * Np_stm6 / Ns_stm6,
    'Rp':  params_stm6_module['Rp'] * Np_stm6 / Ns_stm6,
    'n':   params_stm6_module['n']
}

params_sdm_cell = params_sdm_cell.copy()
Ns_sdm = 1
Np_sdm = 1


# линейная экстраполяция параметров ячейки по Ns

def extrapolate_linear(Ns, Ns1, val1, Ns2, val2):
    slope = (val2 - val1) / (Ns2 - Ns1)
    return val1 + slope * (Ns - Ns1)

Ns_stp6 = STP6_Ns   # 120
Np_stp6 = STP6_Np   # 36

params_stp6_cell = {}
for key in params_sdm_cell.keys():
    params_stp6_cell[key] = extrapolate_linear(
        Ns_stp6, Ns_sdm, params_sdm_cell[key], Ns_stm6, params_stm6_cell[key]
    )


# масштабирование параметров ячейки обратно к модулю STP6

params_stp6_module_pred = {
    'Ipv': params_stp6_cell['Ipv'] * Np_stp6,
    'Isd': params_stp6_cell['Isd'] * Np_stp6,
    'Rs':  params_stp6_cell['Rs'] * Ns_stp6 / Np_stp6,
    'Rp':  params_stp6_cell['Rp'] * Ns_stp6 / Np_stp6,
    'n':   params_stp6_cell['n']
}


# сравнение предсказанных параметров с эталонными

print("="*60)
print("Параметры ячейки (cell-level)")
print("="*60)
print(f"SDM (Ns=1):      Ipv={params_sdm_cell['Ipv']:.6f}, Isd={params_sdm_cell['Isd']:.2e}, Rs={params_sdm_cell['Rs']:.6f}, Rp={params_sdm_cell['Rp']:.2f}, n={params_sdm_cell['n']:.6f}")
print(f"STM6 cell:       Ipv={params_stm6_cell['Ipv']:.6f}, Isd={params_stm6_cell['Isd']:.2e}, Rs={params_stm6_cell['Rs']:.6f}, Rp={params_stm6_cell['Rp']:.2f}, n={params_stm6_cell['n']:.6f}")
print(f"STP6 cell (pred): Ipv={params_stp6_cell['Ipv']:.6f}, Isd={params_stp6_cell['Isd']:.2e}, Rs={params_stp6_cell['Rs']:.6f}, Rp={params_stp6_cell['Rp']:.2f}, n={params_stp6_cell['n']:.6f}")

print("\n"+"="*60)
print("Параметры модуля STP6 (Ns=120, Np=36)")
print("="*60)
print("Предсказанные:")
for k, v in params_stp6_module_pred.items():
    print(f"  {k:3} = {v:.8f}" + (" A" if k in ('Ipv','Isd') else " Ohm" if k in ('Rs','Rp') else ""))
print("\nЭталонные (из статьи):")
for k, v in params_stp6_module_ref.items():
    print(f"  {k:3} = {v:.8f}" + (" A" if k in ('Ipv','Isd') else " Ohm" if k in ('Rs','Rp') else ""))

print("\nОтносительная ошибка предсказания (модуль):")
for k in params_stp6_module_pred:
    err = abs(params_stp6_module_pred[k] - params_stp6_module_ref[k]) / abs(params_stp6_module_ref[k]) * 100
    print(f"  {k}: {err:.2f}%")


# генерация I-V кривой STP6 по предсказанным параметрам и сравнение с реальной

data_stp6 = PVData(
    voltage=STP6_VOLTAGE,
    current=STP6_CURRENT,
    temperature_K=STP6_TEMPERATURE_K,
    Ns=STP6_Ns,
    Np=STP6_Np,
    name="STP6_real"
)

model_pred = SingleDiodeModuleModel(data_stp6)
I_pred = model_pred.compute_current(
    np.array([params_stp6_module_pred['Ipv'], params_stp6_module_pred['Isd'],
              params_stp6_module_pred['Rs'], params_stp6_module_pred['Rp'],
              params_stp6_module_pred['n']])
)

I_real = data_stp6.current
rmse = np.sqrt(np.mean((I_real - I_pred)**2))
print(f"\nRMSE между предсказанным и реальным током для STP6: {rmse:.6e}")

# эталонные параметры для сравнения
I_ref = model_pred.compute_current(
    np.array([params_stp6_module_ref['Ipv'], params_stp6_module_ref['Isd'],
              params_stp6_module_ref['Rs'], params_stp6_module_ref['Rp'],
              params_stp6_module_ref['n']])
)
rmse_ref = np.sqrt(np.mean((I_real - I_ref)**2))
print(f"RMSE для эталонных параметров STP6: {rmse_ref:.6e}")


# вывод таблицы ошибок по точкам (первые 5)

print("\nПоточное сравнение (первые 5 точек):")
print("V (V)\tI_real (A)\tI_pred (A)\tError (A)")
for i in range(min(5, len(STP6_VOLTAGE))):
    err_abs = I_real[i] - I_pred[i]
    print(f"{STP6_VOLTAGE[i]:.1f}\t{I_real[i]:.4f}\t\t{I_pred[i]:.4f}\t\t{err_abs:.4f}")
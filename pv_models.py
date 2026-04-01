import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass



class PhysicalConstants:
    # физические константы для PV моделей
    q = 1.60217646e-19  # элементарный заряд (C)
    k = 1.3806503e-23  # постоянная Больцмана (J/K)

# Температура для RTC France: 33°C = 306.15 K
RTC_TEMPERATURE_K = 306.15

# I-V данные для RTC France (26 точек)
RTC_VOLTAGE = np.array([
    -0.2057, -0.1291, -0.0588, 0.0057, 0.0646, 0.1185, 0.1678, 0.2132, 0.2545, 0.2924,
    0.3269, 0.3585, 0.3873, 0.4137, 0.4373, 0.4590, 0.4784, 0.4960, 0.5119, 0.5265,
    0.5398, 0.5521, 0.5633, 0.5736, 0.5833, 0.5900
])

RTC_CURRENT = np.array([
    0.7640, 0.7620, 0.7605, 0.7605, 0.7600, 0.7590, 0.7570, 0.7570, 0.7555, 0.7540,
    0.7505, 0.7465, 0.7385, 0.7280, 0.7065, 0.6755, 0.6320, 0.5730, 0.4990, 0.4130,
    0.3165, 0.2120, 0.1035, -0.0100, -0.1230, -0.2100
])

# STM6-40/36 (mono-crystalline)
STM6_VOLTAGE = np.array([
    0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 11, 12.1, 13.2, 14.3, 15.4, 16.5, 17.6, 18.7, 19.8, 20.9
])
STM6_CURRENT = np.array([
    1.663, 1.663, 1.660, 1.658, 1.655, 1.650, 1.642, 1.632, 1.618, 1.600, 1.578, 1.552, 1.520, 1.480, 1.430, 1.370,
    1.290, 1.190, 1.060, 0.900
])
STM6_Ns = 36
STM6_Np = 1
STM6_TEMPERATURE_K = 324.15  # 51°C

# STP6-120/36 (poly-crystalline)
STP6_VOLTAGE = np.array([
    0, 2.2, 4.4, 6.6, 8.8, 11, 13.2, 15.4, 17.6, 19.8, 22, 24.2, 26.4, 28.6, 30.8, 33, 35.2, 37.4, 39.6, 41.8
])
STP6_CURRENT = np.array([
    7.48, 7.48, 7.47, 7.45, 7.42, 7.38, 7.32, 7.24, 7.14, 7.00, 6.82, 6.58, 6.24, 5.78, 5.16, 4.32, 3.24, 1.92, 0.36,
    -1.08
])
STP6_Ns = 120
STP6_Np = 36
STP6_TEMPERATURE_K = 324.15  # 51°C


@dataclass
class PVData:
    # контейнер для данных PV модели
    voltage: np.ndarray
    current: np.ndarray
    temperature_K: float
    Ns: int = 1  
    Np: int = 1  
    name: str = "unknown"

    @property
    def n_points(self) -> int:
        return len(self.voltage)


class SingleDiodeModel:
    """
    Single Diode Model (SDM) - 5 параметров
    Параметры: Ipv, Isd, Rs, Rp, n

    Уравнение: I = Ipv - Isd * (exp(q*(V+I*Rs)/(n*k*T)) - 1) - (V+I*Rs)/Rp
    """

    def __init__(self, data: PVData):
        self.data = data
        self.V = data.voltage
        self.T = data.temperature_K
        self.Ns = data.Ns
        self.Np = data.Np
        self.n_points = data.n_points

        self.q = PhysicalConstants.q
        self.k = PhysicalConstants.k

        self.VT = self.k * self.T / self.q  # Тепловое напряжение

    def compute_current(self, params: np.ndarray, V: np.ndarray = None) -> np.ndarray:
        # вычисление тока для заданных параметров с использованием итеративного метода

        Ipv, Isd, Rs, Rp, n = params

        if V is None:
            V = self.V

        # Начальное приближение (I = Ipv - V/Rp)
        I = np.maximum(Ipv - V / Rp, -0.1)

        # Итерационное решение
        for _ in range(10):
            exp_arg = (V + I * Rs) / (n * self.VT)
            exp_arg = np.clip(exp_arg, -100, 100)
            I_new = Ipv - Isd * (np.exp(exp_arg) - 1) - (V + I * Rs) / Rp
            I_new = np.clip(I_new, -0.5, Ipv + 0.1)

            if np.max(np.abs(I_new - I)) < 1e-10:
                break
            I = I_new

        return I

    def compute_current_fast(self, params: np.ndarray, V: np.ndarray = None) -> np.ndarray:
        # вычисление тока через Lambert W (более быстрый метод)

        Ipv, Isd, Rs, Rp, n = params

        if V is None:
            V = self.V

        V_T = self.VT

        a = n * V_T / Rs
        b = Isd * Rs / a
        c = (Ipv + Isd) * Rs + V
        d = a * Rp / (Rs + Rp)

        W_arg = b * np.exp(c / d)
        W_arg = np.clip(W_arg, 1e-10, 1e100)

        # Lambert W аппроксимация
        W = np.real(self._lambertw(W_arg))

        I = (Ipv + Isd - V / Rp) / (1 + Rs / Rp) - (a / Rs) * W

        return np.clip(I, -0.5, Ipv + 0.1)

    def _lambertw(self, z: np.ndarray) -> np.ndarray:
        # приближенное вычисление Lambert W (главной ветви)
        from scipy.special import lambertw as scipy_lambertw
        try:
            return scipy_lambertw(z, k=0)
        except ImportError:
            # приближение для случая без scipy
            result = np.zeros_like(z)
            for i, zi in enumerate(z):
                if zi == 0:
                    result[i] = 0
                else:
                    w = np.log(zi)
                    for _ in range(5):
                        w = w - (w * np.exp(w) - zi) / (
                                    np.exp(w) * (w + 1) - (w + 2) * (w * np.exp(w) - zi) / (2 * w + 2))
                    result[i] = w
            return result

    def objective(self, params: np.ndarray) -> float:
        # целевая функция RMSE (Root Mean Square Error)

        I_calc = self.compute_current(params)

        # вычисляем RMSE
        rmse = np.sqrt(np.mean((self.data.current - I_calc) ** 2))
        return rmse


class DoubleDiodeModel:
    """
    Double Diode Model (DDM) - 7 параметров
    Параметры: Ipv, Isd1, Isd2, Rs, Rp, n1, n2

    Уравнение: I = Ipv - Isd1*(exp(q*(V+I*Rs)/(n1*k*T))-1)
                    - Isd2*(exp(q*(V+I*Rs)/(n2*k*T))-1) - (V+I*Rs)/Rp
    """

    def __init__(self, data: PVData):
        self.data = data
        self.V = data.voltage
        self.T = data.temperature_K
        self.n_points = data.n_points

        self.q = PhysicalConstants.q
        self.k = PhysicalConstants.k

        self.VT = self.k * self.T / self.q

    def compute_current(self, params: np.ndarray) -> np.ndarray:
        Ipv, Isd1, Isd2, Rs, Rp, n1, n2 = params

        I = np.maximum(Ipv - self.V / Rp, -0.1)

        for _ in range(20):
            exp1 = np.exp((self.V + I * Rs) / (n1 * self.VT))
            exp2 = np.exp((self.V + I * Rs) / (n2 * self.VT))

            I_new = Ipv - Isd1 * (exp1 - 1) - Isd2 * (exp2 - 1) - (self.V + I * Rs) / Rp
            I_new = np.clip(I_new, -0.5, Ipv + 0.1)

            if np.max(np.abs(I_new - I)) < 1e-10:
                break
            I = I_new

        return I

    def objective(self, params: np.ndarray) -> float:
        I_calc = self.compute_current(params)
        rmse = np.sqrt(np.mean((self.data.current - I_calc) ** 2))
        return rmse


class SingleDiodeModuleModel(SingleDiodeModel):
    """
    Single Diode PV Module Model - 5 параметров
    Учитывает последовательное и параллельное соединение ячеек
    """

    def compute_current(self, params: np.ndarray, V: np.ndarray = None) -> np.ndarray:
        Ipv, Isd, Rs, Rp, n = params

        if V is None:
            V = self.V

        V_T = self.VT

        I = np.maximum(Ipv * self.Np - V / (Rp * self.Ns / self.Np), -0.1)

        for _ in range(20):
            exp_arg = (V * self.Np + I * Rs * self.Ns) / (n * self.Ns * self.Np * V_T)
            exp_arg = np.clip(exp_arg, -100, 100)
            I_new = (Ipv * self.Np - Isd * self.Np * (np.exp(exp_arg) - 1) -
                     (V * self.Np + I * Rs * self.Ns) / (Rp * self.Ns))
            I_new = np.clip(I_new, -0.5, Ipv * self.Np + 0.1)

            if np.max(np.abs(I_new - I)) < 1e-10:
                break
            I = I_new

        return I



class DecomposedSDM:
    """
    Single Diode Model с декомпозицией поискового пространства

    параметры разделены на:
    - Нелинейные: [n, Rs]
    - Линейные: [Ipv, Isd, Rp]

    линейные параметры аналитически 
    """

    def __init__(self, data: PVData):
        self.data = data
        self.V = data.voltage
        self.I_meas = data.current
        self.T = data.temperature_K
        self.n_points = data.n_points

        self.q = PhysicalConstants.q
        self.k = PhysicalConstants.k
        self.VT = self.k * self.T / self.q

    def compute_linear_params(self, n: float, Rs: float) -> Tuple[float, float, float]:

        n = np.clip(n, 1.0, 2.0)
        Rs = np.clip(Rs, 0.0, 0.5)

        Rs = min(Rs, 0.5)

        M = np.zeros(self.n_points)
        Q = np.zeros(self.n_points)

        for i in range(self.n_points):
            exp_arg = (self.V[i] + self.I_meas[i] * Rs) / (n * self.VT)

            exp_arg = np.clip(exp_arg, -50, 50)
            M[i] = -(np.exp(exp_arg) - 1)
            Q[i] = -(self.V[i] + self.I_meas[i] * Rs)


        E = np.ones(self.n_points)

        A = np.array([
            [np.dot(E, E), np.dot(E, M), np.dot(E, Q)],
            [np.dot(M, E), np.dot(M, M), np.dot(M, Q)],
            [np.dot(Q, E), np.dot(Q, M), np.dot(Q, Q)]
        ])

        B = np.array([
            np.dot(E, self.I_meas),
            np.dot(M, self.I_meas),
            np.dot(Q, self.I_meas)
        ])


        A = A + np.eye(3) * 1e-8

        try:
            x = np.linalg.solve(A, B)
            Ipv = x[0]
            Isd = x[1]
            inv_Rp = x[2]


            Ipv = np.clip(Ipv, 0.5, 0.9)
            Isd = np.clip(Isd, 1e-9, 2e-5)
            Rp = 1.0 / inv_Rp if inv_Rp > 1e-10 else 100.0
            Rp = np.clip(Rp, 10.0, 2000.0)

            return Ipv, Isd, Rp

        except np.linalg.LinAlgError:

            return 0.76, 3e-7, 53.0

    def objective_decomposed(self, params: np.ndarray) -> float:

        # целевая функция с декомпозицией
 
        n, Rs = params


        if n < 1.0 or n > 2.0:
            return 1e6 + abs(n - 1.0) * 1e4
        if Rs < 0.0 or Rs > 0.5:
            return 1e6 + abs(Rs) * 1e4

        try:
            Ipv, Isd, Rp = self.compute_linear_params(n, Rs)

       
            if Ipv < 0.5 or Ipv > 0.9:
                return 1e5 + abs(Ipv - 0.76) * 1e4
            if Isd < 1e-9 or Isd > 1e-5:
                return 1e5 + abs(Isd) * 1e4
            if Rp < 10 or Rp > 2000:
                return 1e5 + abs(Rp - 53) * 1e2

           
            I_calc = np.zeros(self.n_points)
            for i in range(self.n_points):
                try:
                    exp_arg = (self.V[i] + self.I_meas[i] * Rs) / (n * self.VT)
                    exp_arg = np.clip(exp_arg, -50, 50)
                    I_calc[i] = Ipv - Isd * (np.exp(exp_arg) - 1) - (self.V[i] + self.I_meas[i] * Rs) / Rp
                except:
                    return 1e6

           
            I_calc = np.clip(I_calc, -0.5, 1.0)

           
            rmse = np.sqrt(np.mean((self.I_meas - I_calc) ** 2))

           
            if rmse > 0.1:
                return rmse * 1e6

            return rmse

        except (np.linalg.LinAlgError, OverflowError, ValueError) as e:
            return 1e6




def create_rtc_single_diode_model() -> Tuple[SingleDiodeModel, PVData]:

    data = PVData(
        voltage=RTC_VOLTAGE,
        current=RTC_CURRENT,
        temperature_K=RTC_TEMPERATURE_K,
        Ns=1,
        Np=1,
        name="RTC_France_SDM"
    )
    return SingleDiodeModel(data), data


def create_rtc_double_diode_model() -> Tuple[DoubleDiodeModel, PVData]:

    data = PVData(
        voltage=RTC_VOLTAGE,
        current=RTC_CURRENT,
        temperature_K=RTC_TEMPERATURE_K,
        Ns=1,
        Np=1,
        name="RTC_France_DDM"
    )
    return DoubleDiodeModel(data), data


def create_stm6_module_model() -> Tuple[SingleDiodeModuleModel, PVData]:

    data = PVData(
        voltage=STM6_VOLTAGE,
        current=STM6_CURRENT,
        temperature_K=STM6_TEMPERATURE_K,
        Ns=STM6_Ns,
        Np=STM6_Np,
        name="STM6-40/36"
    )
    return SingleDiodeModuleModel(data), data


def create_stp6_module_model() -> Tuple[SingleDiodeModuleModel, PVData]:

    data = PVData(
        voltage=STP6_VOLTAGE,
        current=STP6_CURRENT,
        temperature_K=STP6_TEMPERATURE_K,
        Ns=STP6_Ns,
        Np=STP6_Np,
        name="STP6-120/36"
    )
    return SingleDiodeModuleModel(data), data




if __name__ == "__main__":
    print("=" * 60)
    print("PV MODELS TEST")
    print("=" * 60)


    model, data = create_rtc_single_diode_model()

    #параметры (из статьи)
    # Ipv=0.76077553, Isd=0.32302079e-6, Rs=0.03637709, Rp=53.71852020, n=1.48118359
    params = np.array([0.76077553, 0.32302079e-6, 0.03637709, 53.71852020, 1.48118359])

    rmse = model.objective(params)

    print(f"\nRTC France Single Diode Model:")
    print(
        f"  Parameters: Ipv={params[0]:.6f}, Isd={params[1]:.2e}, Rs={params[2]:.6f}, Rp={params[3]:.2f}, n={params[4]:.4f}")
    print(f"  RMSE: {rmse:.6e}")
    print(f"  Expected RMSE from paper: 9.8602e-04")


    I_calc = model.compute_current(params)

    errors = data.current - I_calc
    print(f"  Max absolute error: {np.max(np.abs(errors)):.6e}")
    print(f"  Mean absolute error: {np.mean(np.abs(errors)):.6e}")


    print("\n" + "=" * 60)
    print("DECOMPOSED MODEL TEST")
    print("=" * 60)

    decomp_model = DecomposedSDM(data)
    nonlinear_params = np.array([params[4], params[2]]) 


    Ipv_dec, Isd_dec, Rp_dec = decomp_model.compute_linear_params(nonlinear_params[0], nonlinear_params[1])

    print(f"  Original Ipv: {params[0]:.6f} -> Decomposed: {Ipv_dec:.6f}")
    print(f"  Original Isd: {params[1]:.2e} -> Decomposed: {Isd_dec:.2e}")
    print(f"  Original Rp: {params[3]:.2f} -> Decomposed: {Rp_dec:.2f}")

    rmse_dec = decomp_model.objective_decomposed(nonlinear_params)
    print(f"  Decomposed RMSE: {rmse_dec:.6e}")

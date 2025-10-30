# -*- coding: utf-8 -*-
# ПЗ-8. Тактическое планирование эксперимента
# Объект: расстояние R после M=8 шагов случайного блуждания на треугольной решетке (вариант 5 ПЗ-7)

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, kstest, shapiro

# ---------- НАСТРОЙКИ ----------
M = 8                    # число шагов блуждания
beta_conf = 0.95         # доверие β
alpha = 1 - beta_conf

# Требуемые точности:
eps_mean_abs = 0.10      # требуемая абсолютная точность для среднего (|x̄ - μ| ≤ ε)
rel_var_eps  = 0.15      # требуемая относительная точность для дисперсии (±15%)

# Пробный объём и максимум для бутстрапа
n_pilot = 5000
B_boot  = 2000           # число бутстрап-репликаций

rng = np.random.default_rng(42)

# ---------- МОДЕЛЬ ПАРАМЕТРА (вариант 5 ПЗ-7) ----------
angles = np.deg2rad([0, 60, 120, 180, 240, 300])
dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (6,2)

def simulate_R(n):
    """Смещение после M шагов на треугольной решетке; возвращает массив расстояний R."""
    idx = rng.integers(0, 6, size=(n, M))
    steps = dirs[idx]            # (n, M, 2)
    pos = steps.sum(axis=1)      # (n, 2)
    R = np.linalg.norm(pos, axis=1)
    return R

# ---------- ПРОБНЫЙ ЭКСПЕРИМЕНТ ----------
R_pilot = simulate_R(n_pilot)
xbar_p = float(np.mean(R_pilot))
s2_p   = float(np.var(R_pilot, ddof=1))
s_p    = math.sqrt(s2_p)

# Нормальность распределения параметра R (не среднего!)
# Шаг 1: стандартизуем под N(μ,σ²) с оценками из пилота
z_p = (R_pilot - xbar_p) / s_p
ks_stat, ks_p = kstest(z_p, 'norm')      # К–С к стандартной нормали
sh_stat, sh_p = shapiro(R_pilot)         # Шапиро–Уилк (на случай небольших n)

normal_ok = (ks_p > alpha) and (sh_p > alpha)

# ---------- ПЛАНИРОВАНИЕ ОБЪЕМА ДЛЯ СРЕДНЕГО ----------
z = norm.ppf(1 - alpha/2)

if normal_ok:
    # Нормальный подход: n >= (z * s / eps)^2 (используем пилотную s как приближение σ)
    n_mean = math.ceil((z * s_p / eps_mean_abs) ** 2)
    mean_method = "нормальная аппроксимация (z)"
else:
    # Не нормаль: Чебышев для среднего — P(|x̄ - μ| ≥ ε) ≤ σ² / (n ε²)
    n_mean = math.ceil(s2_p / (alpha * eps_mean_abs**2))
    mean_method = "неравенство Чебышева"

# ---------- ПЛАНИРОВАНИЕ ОБЪЕМА ДЛЯ ДИСПЕРСИИ ----------
if normal_ok:
    # При нормальности: (n-1)s^2/χ² дают CI для σ².
    # Требуем относит. погр. ρ: хотим, чтобы обе стороны CI были в пределах ±ρ от σ².
    # Возьмем σ² ≈ s2_p, решим итерационно.
    rho = rel_var_eps
    n_var = 10
    while True:
        n_var += 1
        q1 = chi2.ppf(alpha/2, df=n_var-1)
        q2 = chi2.ppf(1 - alpha/2, df=n_var-1)
        lower = (n_var-1) * s2_p / q2
        upper = (n_var-1) * s2_p / q1
        # относительные отклонения от s2_p
        rel_low = 1 - lower/s2_p
        rel_up  = upper/s2_p - 1
        if rel_low <= rho and rel_up <= rho:
            break
    var_method = "χ²-интервалы (нормальность принята)"
else:
    # Универсально: бутстрап для s^2 и подбор n так, чтобы 95% ДИ имели относит. ширину ≤ 2ρ
    # Прикинем через асимптотику Var(s^2) ≈ (μ4 - σ^4)/n; возьмём μ4 ≈ 3σ^4 (норм. оценка) → Var(s^2)≈(2σ^4)/n
    # Даёт грубое n0, затем уточним бутстрапом.
    rho = rel_var_eps
    n0 = math.ceil(2 / ( (rho**2) * alpha ))   # грубая стартовая оценка
    n_var = max(200, n0)
    # уточнение бутстрапом (несколько шагов увеличиваем n, пока половина относит. ДИ ≤ρ)
    while True:
        # сгенерируем одно большое выборочное распределение и бутстрапим s^2
        sample = simulate_R(n_var)
        s2_hat = np.var(sample, ddof=1)
        boot = []
        for _ in range(B_boot):
            idx = rng.integers(0, n_var, size=n_var)
            b = sample[idx]
            boot.append(np.var(b, ddof=1))
        boot = np.sort(boot)
        lo = boot[int(alpha/2 * B_boot)]
        hi = boot[int((1 - alpha/2) * B_boot)]
        rel_low = 1 - lo/s2_hat
        rel_up  = hi/s2_hat - 1
        if rel_low <= rho and rel_up <= rho:
            break
        n_var = int(n_var * 1.25)  # увеличиваем на 25%
    var_method = "бутстрап (распределение не нормальное)"

n_required = max(n_mean, n_var)

# ---------- ОСНОВНОЙ ЭКСПЕРИМЕНТ ----------
R = simulate_R(n_required)
xbar = float(np.mean(R))
s2   = float(np.var(R, ddof=1))
s    = math.sqrt(s2)

# Интервалы для среднего
if normal_ok:
    half = z * s / math.sqrt(n_required)
    ci_mean = (xbar - half, xbar + half)
else:
    half = math.sqrt(s2 / (alpha * n_required))   # Чебышев
    ci_mean = (xbar - half, xbar + half)

# Интервалы для дисперсии
if normal_ok:
    q1 = chi2.ppf(alpha/2, df=n_required-1)
    q2 = chi2.ppf(1 - alpha/2, df=n_required-1)
    ci_var = ((n_required-1) * s2 / q2, (n_required-1) * s2 / q1)
else:
    # бутстрап-ДИ
    boot = []
    for _ in range(B_boot):
        idx = rng.integers(0, n_required, size=n_required)
        boot.append(np.var(R[idx], ddof=1))
    boot = np.sort(boot)
    ci_var = (boot[int(alpha/2 * B_boot)], boot[int((1 - alpha/2) * B_boot)])

# ---------- ОТЧЁТ ----------
print("ПРОБНЫЙ ЭКСПЕРИМЕНТ")
print("-------------------")
print(f"n_pilot = {n_pilot},  x̄_p = {xbar_p:.5f},  s_p^2 = {s2_p:.5f}")
print(f"Нормальность параметра R:  KS p={ks_p:.3f}, Shapiro p={sh_p:.3f}  → {'OK' if normal_ok else 'НЕ нормаль'}\n")

print("ПЛАНИРОВАНИЕ ОБЪЁМА")
print("-------------------")
print(f"Точность среднего: ε = {eps_mean_abs}, доверие β = {beta_conf}")
print(f"Метод для среднего: {mean_method}, требуемо n_mean = {n_mean}")
print(f"Точность дисперсии: относительная ±{rel_var_eps*100:.1f}%, доверие β = {beta_conf}")
print(f"Метод для дисперсии: {var_method}, требуемо n_var = {n_var}")
print(f"Итого берём n = max(n_mean, n_var) = {n_required}\n")

print("ОСНОВНОЙ ЭКСПЕРИМЕНТ")
print("--------------------")
print(f"Оценка среднего: x̄ = {xbar:.5f},  {beta_conf:.2f}-ДИ = [{ci_mean[0]:.5f}, {ci_mean[1]:.5f}]")
print(f"Оценка дисперсии: s^2 = {s2:.5f},  {beta_conf:.2f}-ДИ = [{ci_var[0]:.5f}, {ci_var[1]:.5f}]\n")

# ---------- ГИСТОГРАММА И ЭМПИРИЧЕСКАЯ СФР ----------
K = 25
counts, edges = np.histogram(R, bins=K, range=(0, R.max()))
rel = counts / n_required
cum = np.cumsum(rel)

print("ГИСТОГРАММА ПАРАМЕТРА R")
print("Интервал                Кол-во   Норм.частота   Меньше или равно")
print("-----------------------------------------------------------------")
for i in range(K):
    a, b = edges[i], edges[i+1]
    print(f"[{a:6.3f} - {b:6.3f})    {counts[i]:6d}       {rel[i]:0.4f}           {cum[i]:0.4f}")

# Графики
plt.hist(R, bins=K, density=True, alpha=0.6, edgecolor="black", label="Гистограмма R")
plt.title(f"ПЗ-8: распределение параметра R (n={n_required})")
plt.xlabel("R"); plt.ylabel("Плотность"); plt.legend(); plt.show()

R_sorted = np.sort(R)
Fn = np.arange(1, n_required+1) / n_required
plt.step(R_sorted, Fn, where='post', label="Эмпирическая F*(R)")
plt.title("Эмпирическая функция распределения F*(R)")
plt.xlabel("R"); plt.ylabel("F"); plt.legend(); plt.show()

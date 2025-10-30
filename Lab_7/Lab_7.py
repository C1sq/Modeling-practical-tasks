# -*- coding: utf-8 -*-
# ПЗ№7, вариант 5: случайное блуждание на треугольной решётке (6 направлений)
# Требуется: расстояние от начала после M=8 шагов; гистограмма и эмпирическая CDF;
# оценка пригодности аппроксимации Рэлея (KS-тест).

import math

import matplotlib.pyplot as plt
import numpy as np

# ---------------- Параметры эксперимента ----------------
M = 8  # число шагов блуждания (по условию)
N = 20000  # число траекторий (экспериментов) >= 1000
K = 25  # число интервалов для гистограммы (15 или 25)

rng = np.random.default_rng(42)

# ---------------- Треугольная решётка: 6 направлений ----------------
# Единичный шаг в направлениях с углами 0°, 60°, 120°, 180°, 240°, 300°
angles = np.deg2rad([0, 60, 120, 180, 240, 300])
dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # shape=(6,2)

# ---------------- Генерация N траекторий по M шагов ----------------
# Выбираем для каждой траектории последовательность направлений
idx = rng.integers(0, 6, size=(N, M))  # индексы направлений
steps = dirs[idx]  # (N, M, 2)
pos = steps.sum(axis=1)  # конечные координаты (x,y), shape=(N,2)
r = np.linalg.norm(pos, axis=1)  # расстояние от начала

# ---------------- Статистики выборки ----------------
mean_r = float(np.mean(r))
var_r = float(np.var(r, ddof=1))

print("РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОГО АНАЛИЗА")
print("==================================")
print(f"Число экспериментов: {N}")
print(f"Число шагов: {M}")
print(f"Математическое ожидание r: {mean_r:.6f}")
print(f"Дисперсия r:             {var_r:.6f}\n")

# ---------------- Гистограмма и эмпирическая CDF ----------------
counts, edges = np.histogram(r, bins=K, range=(0.0, r.max()))
rel = counts / N
cum = np.cumsum(rel)

print("ГИСТОГРАММА РАСПРЕДЕЛЕНИЯ R")
print("Интервал                Кол-во   Норм.частота   Меньше или равно")
print("-----------------------------------------------------------------")
for i in range(K):
    a, b = edges[i], edges[i + 1]
    print(f"[{a:6.3f} - {b:6.3f})    {counts[i]:6d}       {rel[i]:0.4f}           {cum[i]:0.4f}")

# ---------- Проверка аппроксимации распределением Рэлея ----------
# Теория: при изотропном блуждании (и для нашей шестиугольной симметрии тоже)
# X и Y ~ примерно N(0, σ^2) с σ^2 ≈ M/2, поэтому R = sqrt(X^2+Y^2) ~ Rayleigh(σ).
# Оценим σ по ММП: σ_hat = sqrt(E[R^2]/2)
sigma_hat = math.sqrt(np.mean(r ** 2) / 2.0)


def rayleigh_cdf(x, sigma):
    return 1.0 - np.exp(-x * x / (2.0 * sigma * sigma))


# Колмогоров против F_Rayleigh(r; σ_hat)
r_sorted = np.sort(r)
Fn = np.arange(1, N + 1) / N
F_theor = rayleigh_cdf(r_sorted, sigma_hat)
D_plus = float(np.max(Fn - F_theor))
D_minus = float(np.max(F_theor - (np.arange(0, N) / N)))
Dn = max(D_plus, D_minus)
Dcrit = 1.36 / math.sqrt(N)  # α = 0.05 (асимптотика)

print("\nПроверка аппроксимации Рэлея:")
print(f"Оценка σ (ММП): {sigma_hat:.6f}")
print(f"Колмогоров: D = {Dn:.5f} (D+={D_plus:.5f}, D-={D_minus:.5f}),  D_crit(0.05) = {Dcrit:.5f}")
print(
    f"Вывод: {'аппроксимация Рэлея приемлема (H0 не отвергаем)' if Dn < Dcrit else 'аппроксимация не подтверждается на 5% уровне'}")

# ---------------- Графики ----------------
# 1) Гистограмма + теор. плотность Рэлея с σ_hat
xx = np.linspace(0, r.max(), 400)
rayleigh_pdf = (xx / (sigma_hat ** 2)) * np.exp(-xx * xx / (2 * sigma_hat * sigma_hat))

plt.hist(r, bins=K, density=True, alpha=0.6, edgecolor='black', label="Гистограмма R")
plt.plot(xx, rayleigh_pdf, linewidth=2, label=f"Рэлея, σ̂={sigma_hat:.3f}")
plt.title("Расстояние после M=8 шагов (треугольная решётка)")
plt.xlabel("r");
plt.ylabel("Плотность");
plt.legend();
plt.show()

# 2) Эмпирическая и теоретическая CDF
plt.step(r_sorted, Fn, where='post', label="Эмпирическая F*(r)")
plt.plot(xx, rayleigh_cdf(xx, sigma_hat), label="Рэлея F(r; σ̂)")
plt.title("Эмпирическая и теоретическая функция распределения R")
plt.xlabel("r");
plt.ylabel("F(r)");
plt.legend();
plt.show()

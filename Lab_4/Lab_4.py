# -*- coding: utf-8 -*-
# ПЗ-4. Генерация нормального распределения N(4.3, 0.5)
# Метод ЦПТ и метод Бокса–Маллера
# Проверка распределения по критериям Пирсона и Колмогорова

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm  # используем функции pdf и cdf из SciPy

# ---------------- Параметры ----------------
N = 1000          # объём выборки (>=1000)
K = 25            # число интервалов
mu = 4.3
sigma2 = 0.5
sigma = math.sqrt(sigma2)
rng = np.random.default_rng(42)

# ---------------- Генерация: ЦПТ ----------------
def generate_normal_clt(n, mu, s, rng):
    # сумма 12 U(0,1) - 6 ~ N(0,1)
    u = rng.random((n, 12))
    z = np.sum(u, axis=1) - 6.0
    return mu + s * z

# ---------------- Генерация: Бокс–Маллер ----------------
def generate_normal_box_muller(n, mu, s, rng):
    m = (n + 1) // 2
    u1 = rng.random(m)
    u2 = rng.random(m)
    r = np.sqrt(-2.0 * np.log(u1 + 1e-16))
    theta = 2.0 * np.pi * u2
    z1 = r * np.cos(theta)
    z2 = r * np.sin(theta)
    z = np.concatenate([z1, z2])[:n]
    return mu + s * z

# ---------------- Подпрограмма анализа ----------------
def report(name, sample, mu, s, K):
    n = len(sample)
    mean = float(np.mean(sample))
    var  = float(np.var(sample, ddof=1))

    print(f"\n{name}")
    print("=" * len(name))
    print(f"Объем выборки: {n}")
    print(f"Число участков разбиения: {K}\n")
    print(f"Математическое ожидание (выборочное): {mean:.6f}")
    print(f"Дисперсия (выборочная):              {var:.6f}\n")

    # --- Гистограмма ---
    left, right = mu - 4*s, mu + 4*s
    edges = np.linspace(left, right, K+1)
    counts, _ = np.histogram(sample, bins=edges)
    rel = counts / n
    cum = np.cumsum(rel)

    print("ГИСТОГРАММА РАСПРЕДЕЛЕНИЯ")
    print("Интервал                 Количество   Норм.частота   Меньше или равно")
    print("---------------------------------------------------------------------")
    for i in range(K):
        a, b = edges[i], edges[i+1]
        print(f"[{a:6.3f} - {b:6.3f})       {counts[i]:5d}        {rel[i]:0.4f}            {cum[i]:0.4f}")

    # --- Критерий Пирсона ---
    expected = np.array([n * (norm.cdf(edges[i+1], mu, s) - norm.cdf(edges[i], mu, s))
                         for i in range(K)])
    mask = expected > 0
    chi2 = float(np.sum((counts[mask] - expected[mask])**2 / expected[mask]))
    df = mask.sum() - 1
    print("\nСТАТИСТИЧЕСКИЕ КРИТЕРИИ")
    print("========================")
    print("1. Критерий Пирсона (хи-квадрат):")
    print(f"   Статистика хи-квадрат: {chi2:0.4f}")
    print(f"   Степени свободы: ≈ {df}")

    # --- Критерий Колмогорова ---
    xs = np.sort(sample)
    Fn = np.arange(1, n+1) / n
    Ftheor = norm.cdf(xs, mu, s)
    Dplus  = float(np.max(Fn - Ftheor))
    Dminus = float(np.max(Ftheor - (np.arange(0, n) / n)))
    Dn = max(Dplus, Dminus)
    Dcrit = 1.36 / math.sqrt(n)  # α=0.05
    print("\n2. Критерий Колмогорова:")
    print(f"   Статистика Колмогорова: {Dn:0.5f} (D+={Dplus:0.5f}, D-={Dminus:0.5f})")
    print(f"   Критическое значение (alpha=0.05): {Dcrit:0.5f}")
    print(f"   Результат: {'Распределение нормальное (не отвергаем H0)' if Dn < Dcrit else 'Отвергаем H0'}")

    # --- Графики ---
    xs_grid = np.linspace(left, right, 1000)
    plt.hist(sample, bins=edges, density=True, alpha=0.6, edgecolor="black", label="Гистограмма")
    plt.plot(xs_grid, norm.pdf(xs_grid, mu, s), linewidth=2, label="Теоретическая плотность")
    plt.title(f"{name}: гистограмма и теоретическая плотность N({mu}, {sigma2})")
    plt.xlabel("x"); plt.ylabel("Плотность"); plt.legend(); plt.show()

    plt.step(xs, Fn, where="post", label="Эмпирическая F*(x)")
    plt.plot(xs_grid, norm.cdf(xs_grid, mu, s), label="Теоретическая F(x)")
    plt.title(f"{name}: функция распределения")
    plt.xlabel("x"); plt.ylabel("F(x)"); plt.legend(); plt.show()

# ---------------- Генерация и анализ ----------------
X_clt = generate_normal_clt(N, mu, sigma, rng)
report("Метод ЦПТ", X_clt, mu, sigma, K)

X_bm = generate_normal_box_muller(N, mu, sigma, rng)
report("Метод Бокса–Маллера", X_bm, mu, sigma, K)

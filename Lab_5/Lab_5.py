# -*- coding: utf-8 -*-
# ПЗ: Экспоненциальное и Бета-распределение
# Требуется: N>=1000; k=15 или 25; построить гистограмму и СФР;
# оценить матожидание и дисперсию; проверить соответствие теории
# по Критерию Пирсона ИЛИ Колмогорова (выбор TEST ниже).

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, beta  # для теоретических pdf/cdf и квантилей

# ---------------- ПАРАМЕТРЫ ----------------
N = 1000        # объём выборки (>=1000)
K = 25          # число интервалов (15 или 25)
TEST = 'kolmogorov'   # 'kolmogorov' ИЛИ 'pearson'

# ПОДСТАВЬТЕ из своего варианта табл.5:
lambda_ = 0.8   # Экспоненциальное (интенсивность >0): E=1/lambda, Var=1/lambda^2
alpha   = 2.0   # Бета-распределение: alpha>0
beta_b  = 5.0   # Бета-распределение: beta>0

rng = np.random.default_rng(42)

# ---------------- ПОДПРОГРАММЫ ГЕНЕРАЦИИ ----------------
def gen_exponential(n, lam, rng):
    """Инверсионный метод: X = -ln(1-U)/λ, U~U(0,1)."""
    U = rng.random(n)
    return -np.log1p(-U) / lam

def gen_beta_via_gamma(n, a, b, rng):
    """Бета(α,β) через гаммы: X=G1/(G1+G2), где G1~Γ(α,1), G2~Γ(β,1)."""
    g1 = rng.gamma(shape=a, scale=1.0, size=n)
    g2 = rng.gamma(shape=b, scale=1.0, size=n)
    return g1 / (g1 + g2)

# ---------------- ВСПОМОГАТЕЛЬНЫЕ ----------------
def print_hist_table(counts, edges, n):
    rel = counts / n
    cum = np.cumsum(rel)
    print("ГИСТОГРАММА РАСПРЕДЕЛЕНИЯ")
    print("Интервал                      Кол-во   Норм.частота   Меньше или равно")
    print("---------------------------------------------------------------------")
    for i in range(len(counts)):
        a, b = edges[i], edges[i+1]
        print(f"[{a:8.4f} ; {b:8.4f})      {counts[i]:5d}       {rel[i]:0.4f}          {cum[i]:0.4f}")

def kolmogorov_test(sample, cdf_func):
    n = len(sample)
    xs = np.sort(sample)
    Fn = np.arange(1, n+1) / n
    F  = cdf_func(xs)
    Dplus  = float(np.max(Fn - F))
    Dminus = float(np.max(F - (np.arange(0, n) / n)))
    Dn = max(Dplus, Dminus)
    Dcrit = 1.36 / math.sqrt(n)  # α=0.05 (асимптотика)
    return Dn, Dplus, Dminus, Dcrit, (Dn < Dcrit)

def pearson_test(counts, edges, n, cdf_func):
    expected = np.array([n * (cdf_func(edges[i+1]) - cdf_func(edges[i])) for i in range(len(counts))])
    mask = expected > 0
    chi2 = float(np.sum((counts[mask] - expected[mask])**2 / expected[mask]))
    df   = int(mask.sum() - 1)   # в учебных задачах обычно без поправок
    return chi2, df

def analyze(name, sample, edges, pdf_func, cdf_func):
    n = len(sample)
    mean = float(np.mean(sample))
    var  = float(np.var(sample, ddof=1))

    print(f"\n{name}")
    print("=" * len(name))
    print(f"Объем выборки: {n}")
    print(f"Число участков разбиения: {len(edges)-1}\n")
    print(f"Математическое ожидание (выборочное): {mean:.6f}")
    print(f"Дисперсия (выборочная):              {var:.6f}\n")

    counts, _ = np.histogram(sample, bins=edges)
    print_hist_table(counts, edges, n)

    print("\nСТАТИСТИЧЕСКИЕ КРИТЕРИИ")
    print("========================")
    if TEST.lower() == 'kolmogorov':
        Dn, Dp, Dm, Dcrit, ok = kolmogorov_test(sample, cdf_func)
        print("Критерий Колмогорова:")
        print(f"   D = {Dn:.5f}  (D+={Dp:.5f}, D-={Dm:.5f})")
        print(f"   Критическое значение (α=0.05): {Dcrit:.5f}")
        print(f"   Результат: {'не отвергаем H0' if ok else 'отвергаем H0'}")
    else:
        chi2, df = pearson_test(counts, edges, n, cdf_func)
        print("Критерий Пирсона (хи-квадрат):")
        print(f"   χ² = {chi2:.4f}, df ≈ {df}  (сравни с табличным χ²_crit при выбранном α)")
        print(f"   Результат: {'не отвергаем H0' if chi2 < 36.415 and df==24 else 'см. сравнение с χ²_crit'}")

    # Графики
    xs_grid = np.linspace(edges[0], edges[-1], 1000)
    plt.hist(sample, bins=edges, density=True, alpha=0.6, edgecolor="black", label="Гистограмма")
    plt.plot(xs_grid, pdf_func(xs_grid), linewidth=2, label="Теоретическая плотность")
    plt.title(f"{name}: гистограмма и теоретическая плотность")
    plt.xlabel("x"); plt.ylabel("Плотность"); plt.legend(); plt.show()

    xs = np.sort(sample)
    Fn = np.arange(1, n+1) / n
    plt.step(xs, Fn, where="post", label="Эмпирическая F*(x)")
    plt.plot(xs_grid, cdf_func(xs_grid), label="Теоретическая F(x)")
    plt.title(f"{name}: функция распределения")
    plt.xlabel("x"); plt.ylabel("F(x)"); plt.legend(); plt.show()

# ---------------- ЭКСПОНЕНЦИАЛЬНОЕ ----------------
X_exp = gen_exponential(N, lambda_, rng)

# границы: от 0 до 99.5%-квантили — почти вся масса
right_exp = float(expon.ppf(0.995, scale=1.0/lambda_))
edges_exp = np.linspace(0.0, right_exp, K+1)

pdf_exp = lambda x: expon.pdf(x, scale=1.0/lambda_)
cdf_exp = lambda x: expon.cdf(x, scale=1.0/lambda_)

print("Теория (экспоненциальное): E[X]=1/λ, Var[X]=1/λ²")
print(f"E[X] теор. = {1.0/lambda_:.6f}, Var[X] теор. = {(1.0/lambda_)**2:.6f}")
analyze("Экспоненциальное распределение", X_exp, edges_exp, pdf_exp, cdf_exp)

# ---------------- БЕТА-РАСПРЕДЕЛЕНИЕ ----------------
X_beta = gen_beta_via_gamma(N, alpha, beta_b, rng)

edges_beta = np.linspace(0.0, 1.0, K+1)

pdf_beta = lambda x: beta.pdf(x, alpha, beta_b)
cdf_beta = lambda x: beta.cdf(x, alpha, beta_b)

print("\nТеория (бета): E[X]=α/(α+β), Var[X]=αβ/[(α+β)²(α+β+1)]")
print(f"E[X] теор. = {alpha/(alpha+beta_b):.6f}, Var[X] теор. = {(alpha*beta_b)/(((alpha+beta_b)**2)*(alpha+beta_b+1)):.6f}")
analyze("Бета-распределение", X_beta, edges_beta, pdf_beta, cdf_beta)

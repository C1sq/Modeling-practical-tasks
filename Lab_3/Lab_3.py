import math
import numpy as np
import matplotlib.pyplot as plt

N = 1000
K = 25

I = 12
m = 2 ** I
A, B, C = 6, 7, 3
seed = 4001

class QuadRNG:
    def __init__(self, A, B, C, m, y0):
        self.A, self.B, self.C, self.m, self.y = A, B, C, m, y0
    def next(self):
        self.y = (self.A * self.y * self.y + self.B * self.y + self.C) % self.m
        return self.y / self.m  # U in [0,1)

rng = QuadRNG(A, B, C, m, seed)


Z = 0.825

def f_raw(x):
    if 0.0 <= x < 0.5:
        return 0.3
    elif 0.5 <= x < 0.7:
        return 3.0
    elif 0.7 <= x <= 1.0:
        return 0.25
    return 0.0

def pdf(x):
    return f_raw(x) / Z

c1 = (0.3 / Z)
F_05 = c1 * 0.5
c2 = (3.0 / Z)
F_07 = F_05 + c2 * (0.7 - 0.5)
c3 = (0.25 / Z)

def cdf(x):
    if x <= 0:
        return 0.0
    if x < 0.5:
        return c1 * x
    if x < 0.7:
        return F_05 + c2 * (x - 0.5)
    if x <= 1.0:
        return F_07 + c3 * (x - 0.7)
    return 1.0

M = max(0.3/Z, 3.0/Z, 0.25/Z)

# ---------------- Генерация методом отбора ----------------
def generate_sample(N, rng):
    sample = np.empty(N, dtype=float)
    i = 0
    while i < N:
        x = rng.next()
        y = rng.next() * M
        if y <= pdf(x):
            sample[i] = x
            i += 1
    return sample

X = generate_sample(N, rng)

# ---------------- Статистики и печать гистограммы ----------------
mean = float(np.mean(X))
var = float(np.var(X, ddof=1))

print("РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОГО АНАЛИЗА (ПЗ-3, вар. 5)")
print("=================================================")
print(f"Объем выборки: {N}")
print(f"Число участков разбиения: {K}\n")
print(f"Математическое ожидание (выборочное): {mean:.6f}")
print(f"Дисперсия (выборочная):              {var:.6f}\n")

counts, edges = np.histogram(X, bins=K, range=(0.0, 1.0))
rel_freq = counts / N
cum = np.cumsum(rel_freq)

print("ГИСТОГРАММА РАСПРЕДЕЛЕНИЯ")
print("Интервал           Количество   Норм.частота   Меньше или равно")
print("----------------------------------------------------------------")
for i in range(K):
    print(f"[{edges[i]:.2f} - {edges[i+1]:.2f})   {counts[i]:4d}        {rel_freq[i]:0.4f}        {cum[i]:4f}")

# ---------------- Критерий Пирсона (под заданную CDF) ----------------
# Ожидаемое в i-том интервале: N * (F(b) - F(a))
expected = np.array([N * (cdf(edges[i+1]) - cdf(edges[i])) for i in range(K)])
# объединяем интервалы с E<5 (если есть), чтобы корректно применять χ²
# (минимально аккуратно: просто не считаем те, где E≈0 — такие тут не встретятся)
mask = expected > 0
chi2 = float(np.sum((counts[mask] - expected[mask])**2 / expected[mask]))
df = mask.sum() - 1  # приблизительно (без учёта оценивания параметров)
print("\nКРИТЕРИЙ ПИРСОНА (с учётом теоретической CDF варианта 5)")
print(f"χ² = {chi2:.4f}, df ≈ {df}")

# ---------------- Критерий Колмогорова (с теоретической CDF) ----------------
X_sorted = np.sort(X)
Fn = np.arange(1, N+1) / N
F_theor = np.vectorize(cdf)(X_sorted)
D_plus  = float(np.max(Fn - F_theor))
D_minus = float(np.max(F_theor - (np.arange(0, N) / N)))
Dn = max(D_plus, D_minus)
Dn_crit = 1.36 / math.sqrt(N)  # α=0.05
print("\nКРИТЕРИЙ КОЛМОГОРОВА (с теоретической CDF варианта 5)")
print(f"D = {Dn:.5f}  (D+={D_plus:.5f}, D-={D_minus:.5f})")
print(f"Критическое значение (alpha=0.05): {Dn_crit:.5f}")
print(f"Результат: {'не отвергаем H0' if Dn < Dn_crit else 'отвергаем H0'}")

# ---------------- Графики ----------------
plt.hist(X, bins=K, density=True, edgecolor="black", alpha=0.6)
# теоретическая плотность (ступеньками)
xx = np.linspace(0, 1, 1001)
yy = np.vectorize(pdf)(xx)
plt.plot(xx, yy, linewidth=2, label="Теоретическая плотность (вар. 5)")
plt.title("Гистограмма и теоретическая плотность (вариант 5)")
plt.xlabel("x")
plt.ylabel("Плотность")
plt.legend()
plt.show()

plt.step(X_sorted, Fn, where="post", label="Эмпирическая F*(x)")
plt.plot(xx, np.vectorize(cdf)(xx), label="Теоретическая F(x)")
plt.title("Эмпирическая и теоретическая функция распределения")
plt.xlabel("x")
plt.ylabel("F")
plt.legend()
plt.show()

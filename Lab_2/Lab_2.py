# ПЗ №2, вариант 5: квадратичный ГПСЧ + Пирсон, Колмогоров, Покер-тест (k=2)

import math
import numpy as np

N = 5000       # объём выборки (у тебя в ПЗ-1 было 5000)
K = 25         # число интервалов для Пирсона/гистограммы
I = 12
m = 2 ** I
A, B, C = 6, 7, 3
y0 = 4001

def quad_cong(N, A, B, C, m, y):
    u = np.empty(N, dtype=float)
    for i in range(N):
        y = (A * y * y + B * y + C) % m
        u[i] = y / m
    return u

u = quad_cong(N, A, B, C, m, y0)

mean = float(np.mean(u))
var  = float(np.var(u, ddof=0))
counts, edges = np.histogram(u, bins=K, range=(0.0, 1.0))
cum = np.cumsum(counts)

print("РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОГО АНАЛИЗА")
print("==================================")
print(f"Объем выборки: {N}")
print(f"Число участков разбиения: {K}\n")
print(f"Математическое ожидание: {mean:.6f}")
print(f"Дисперсия: {var:.6f}\n")

print("ГИСТОГРАММА РАСПРЕДЕЛЕНИЯ")
print("Интервал           Количество   Норм.частота   Меньше или равно")
print("----------------------------------------------------------------")
w = 1.0 / K
for i in range(K):
    left, right = i*w, (i+1)*w
    print(f"[{left:0.2f} - {right:0.2f})   {counts[i]:4d}        {counts[i]/N:0.4f}        {cum[i]:4d}")

expected = N / K
chi2_stat = float(np.sum((counts - expected) ** 2 / expected))
df_chi2 = K - 1
CHI2_CRIT_95 = 36.415  # df=24 (для K=25)
print("\nСТАТИСТИЧЕСКИЕ КРИТЕРИИ")
print("========================")
print("1. Критерий Пирсона (хи-квадрат):")
print(f"   Статистика хи-квадрат: {chi2_stat:0.4f}")
print(f"   Степени свободы: {df_chi2}")
print(f"   Критическое значение (alpha=0.05): {CHI2_CRIT_95}")
print(f"   Результат: {'Распределение равномерное (не отвергаем H0)' if chi2_stat < CHI2_CRIT_95 else 'Отклоняем H0'}\n")

u_sorted = np.sort(u)
i = np.arange(1, N+1)
Dn_plus  = np.max(i/N - u_sorted)
Dn_minus = np.max(u_sorted - (i-1)/N)
Dn = max(Dn_plus, Dn_minus)
KS_CRIT_05 = 1.36 / math.sqrt(N)
print("2. Критерий Колмогорова:")
print(f"   Статистика Колмогорова: {Dn:0.5f}  (D+={Dn_plus:0.5f}, D-={Dn_minus:0.5f})")
print(f"   Критическое значение (alpha=0.05): {KS_CRIT_05:0.5f}")
print(f"   Результат: {'Распределение равномерное (не отвергаем H0)' if Dn < KS_CRIT_05 else 'Отклоняем H0'}\n")

# ---------- 3) Покер-тест (k=2) ----------
# Берём первые две десятичные цифры числа U: d1,d2 ∈ {0..9}
two = np.floor(u * 100).astype(int)   # 00..99
d1, d2 = two // 10, two % 10
n_blocks = N

# Категории: «пара» (d1==d2) и «разные» (d1!=d2)
obs_pair = int(np.sum(d1 == d2))
obs_diff = n_blocks - obs_pair

# Для базы 10: P(pair)=10/100=0.1, P(diff)=0.9
exp_pair = 0.1 * n_blocks
exp_diff = 0.9 * n_blocks

poker_chi2 = (obs_pair - exp_pair) ** 2 / exp_pair + (obs_diff - exp_diff) ** 2 / exp_diff
POKER_CRIT_95 = 3.841  # df=1
POKER_CRIT_99 = 6.635  # df=1

print("3. Покер-тест (k = 2):")
print(f"   Наблюдения: пара={obs_pair}, разные={obs_diff} (из {n_blocks})")
print(f"   Ожидания:   пара={exp_pair:.1f}, разные={exp_diff:.1f}")
print(f"   χ² = {poker_chi2:.4f}, df=1")
print(f"   Критические значения: 0.05→{POKER_CRIT_95}, 0.01→{POKER_CRIT_99}")
print(f"   Результат: {'не отвергаем H0 (соответствует равномерности по k=2)' if poker_chi2 < POKER_CRIT_95 else 'отвергаем H0'}")

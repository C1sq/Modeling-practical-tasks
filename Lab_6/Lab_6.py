# -*- coding: utf-8 -*-
# ПЗ№6. Стрельба по мишеням: события A,B,C и их оценка Монте-Карло + аналитика.
import numpy as np
from math import sqrt

# ---------- Параметры (ЗАМЕНИ НА СВОИ) ----------
N = 3                      # число стрелков
k = 5                      # боезапас каждого
p = np.array([0.3, 0.5, 0.6], dtype=float)  # p_i: вероятность попадания за 1 выстрел
M = 200_000                # число прогонов Монте-Карло (можно увеличить)

# ---------- АНАЛИТИКА ----------
# Обозначения: q_i = 1 - p_i
q = 1.0 - p

# A: «в сумме останется хотя бы один патрон»
#   эквивалентно НЕ(все израсходовали весь боезапас) → доп. событие = ∏ q_i^k
P_A_exact = 1.0 - np.prod(q**k)

# B: «ни у кого не израсходован весь боезапас»
#   т.е. у каждого успех произошёл до k-го выстрела → 1 - q_i^(k-1)
P_B_exact = np.prod(1.0 - q**(k-1))

# C: «ровно один израсходует весь боезапас, остальные — нет»
#   сумма по j: (q_j^k) * ∏_{i≠j} (1 - q_i^(k-1))
tmp = 1.0 - q**(k-1)
P_C_exact = float(np.sum((q**k) * np.prod(tmp) / tmp))  # аккуратная форма
# или явно:
# P_C_exact = float(sum((q[j]**k) * np.prod(tmp[np.arange(N)!=j]) for j in range(N)))

# ---------- МОНТЕ-КАРЛО ----------
rng = np.random.default_rng(42)

# Матрица попаданий: shape=(M, N, k)
U = rng.random((M, N, k))
hits = U < p[None, :, None]          # True там, где попал

# индекс первого попадания (если не было — argmax вернёт 0, это поправим маской)
first_idx = np.argmax(hits, axis=2)  # 0..k-1
had_hit   = hits.any(axis=2)
bullets_used = np.where(had_hit, first_idx + 1, k)   # сколько выстрелов сделал i-й в прогоне
leftover_total = N*k - bullets_used.sum(axis=1)

# События
A_sim = (leftover_total >= 1)
B_sim = np.all(bullets_used < k, axis=1)
C_sim = (np.sum(bullets_used == k, axis=1) == 1)

pA_hat = A_sim.mean()
pB_hat = B_sim.mean()
pC_hat = C_sim.mean()

def ci95(phat, m):
    # нормальная аппроксимация для доли
    d = 1.96 * sqrt(phat*(1-phat)/m)
    return max(0.0, phat - d), min(1.0, phat + d)

ciA = ci95(pA_hat, M)
ciB = ci95(pB_hat, M)
ciC = ci95(pC_hat, M)

# ---------- Отчёт ----------
print("Параметры: N =", N, "k =", k, "p =", p.tolist(), " M =", M, "\n")

print("Аналитические вероятности:")
print(f"  P(A) = 1 - ∏(1-p_i)^k              = {P_A_exact:.6f}")
print(f"  P(B) = ∏[1 - (1-p_i)^(k-1)]        = {P_B_exact:.6f}")
print(f"  P(C) = Σ_j (1-p_j)^k ∏_{'{i≠j}'} [1 - (1-p_i)^(k-1)] = {P_C_exact:.6f}\n")

print("Монте-Карло (оценка ± 95% ДИ):")
print(f"  P̂(A) = {pA_hat:.6f}  CI95 = [{ciA[0]:.6f}, {ciA[1]:.6f}]")
print(f"  P̂(B) = {pB_hat:.6f}  CI95 = [{ciB[0]:.6f}, {ciB[1]:.6f}]")
print(f"  P̂(C) = {pC_hat:.6f}  CI95 = [{ciC[0]:.6f}, {ciC[1]:.6f}]")

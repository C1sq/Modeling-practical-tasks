import numpy as np

# Параметры варианта 5
N = 5000      # объём выборки
K = 25        # число интервалов
I = 12
m = 2 ** I    # модуль
A, B, C = 6, 7, 3
y = 4001      # начальное значение

x = []
for i in range(N):
    y = (A * y * y + B * y + C) % m
    x.append(y / m)

mean = sum(x) / N
m2 = sum(v**2 for v in x) / N
m3 = sum(v**3 for v in x) / N
var = m2 - mean**2
unbiased_var = sum((v - mean)**2 for v in x) / (N - 1)

theor_mean = 0.5
theor_var = 1/12
theor_m2 = 1/3
theor_m3 = 1/4

counts, edges = np.histogram(x, bins=K)
expected = N / K
chi2 = sum((counts[i] - expected)**2 / expected for i in range(K))

print(f"Распределение чисел по интервалам [{1/K:.6f}]:")
cum = 0
for i in range(K):
    cum += counts[i] / N
    norm_freq = counts[i] / expected
    print(f"{i+1:2d}-й интервал: {counts[i]:3d}   норм. частота: {norm_freq:.6f}   "
          f"Меньше или равно: {cum:.6f}")

print("\nВыборочная средняя:", round(mean, 6))
print("Математическое ожидание (теор.):", theor_mean)
print("Несмещённая оценка дисперсии:", round(unbiased_var, 6))
print("Требуемая дисперсия (теор.):", round(theor_var, 6))
print("Второй момент:", round(m2, 6), " (теор. 1/3 =", theor_m2, ")")
print("Третий момент:", round(m3, 6), " (теор. 1/4 =", theor_m3, ")")
print("Коэффициент ХИ-квадрат:", round(chi2, 6))


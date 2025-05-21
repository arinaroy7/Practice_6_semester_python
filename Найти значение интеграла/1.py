# Найти значение интеграла, значения а, в определяю вариантом 1
# Найти значение интеграла по квадратурной формуле Гаусса с термя узлами
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

#Функция
def f(x):
    return 1 / (1 + 10 * x)

#Параметры
a, b = 0, 1
epsilon = 1e-4

#По формуле левых прямоугольников
def left_rectangles(f, a, b, epsilon):
    n = 1
    integral_old = (b - a) * f(a)
    while True:
        h = (b - a) / n
        x = np.linspace(a, b - h, n)
        integral_new = h * np.sum(f(x))
        if abs(integral_new - integral_old) < epsilon:
            break
        integral_old = integral_new
        n *= 2
    return integral_new, n

integral_lr, n_lr = left_rectangles(f, a, b, epsilon)

print(f"Задание 1: интеграл методом левых прямоугольников: {integral_lr:.6f} (n = {n_lr})")

#График функции и прямоугольников
x_plot = np.linspace(a, b, 1000)
y_plot = f(x_plot)

plt.figure(figsize=(10, 5))
plt.plot(x_plot, y_plot, label="f(x) = 1 / (1 + 10x)", color="red")
h = (b - a) / n_lr
for i in range(n_lr):
    xi = a + i * h
    plt.bar(xi, f(xi), width=h, align='edge', alpha=0.3, color='white', edgecolor='black')
plt.title("Метод левых прямоугольников")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

#Формула Гаусса с 3 узлами на интервале [a, b] Вариант 1
gauss_nodes = [-np.sqrt(3/5), 0, np.sqrt(3/5)]
gauss_weights = [5/9, 8/9, 5/9]

def gauss_quadrature_3(f, a, b):
    mid = (a + b) / 2
    half_length = (b - a) / 2
    result = 0
    for i in range(3):
        xi = mid + half_length * gauss_nodes[i]
        result += gauss_weights[i] * f(xi)
    return half_length * result

integral_gauss = gauss_quadrature_3(f, a, b)
print(f"Задание 2: вычисление интеграла по формуле Гаусса с 3 узлами: {integral_gauss:.6f}")
exact_value, _ = quad(f, a, b)
print(f"Точное значение (scipy): {exact_value:.6f}")
print(f"Погрешность метода левых прямоугольников: {abs(exact_value - integral_lr):.6e}")
print(f"Погрешность формулы Гаусса: {abs(exact_value - integral_gauss):.6e}")
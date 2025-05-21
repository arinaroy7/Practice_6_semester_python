# Построить многочлен Ньютона по ФИО
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from numpy.linalg import lstsq

t = 8 # Фамилия Имя Отчество
k = 5
N = 21

x_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y_vals = np.array([
    0.2 * N, 0.3 * t, 0.5 * k, 0.6 * N, 0.7 * t,
    k, 0.8 * N, 1.2 * k, 1.3 * t, N
])

# Многочлен Ньютона
def divided_differences(x, y):
    n = len(y)
    coeffs = np.copy(y)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coeffs[i] = (coeffs[i] - coeffs[i - 1]) / (x[i] - x[i - j])
    return coeffs

def newton_poly(x, x_data, coeffs):
    n = len(coeffs)
    result = coeffs[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x - x_data[i]) + coeffs[i]
    return result

coeffs_newton = divided_differences(x_vals, y_vals)

#Параболический сплайн
spline = CubicSpline(x_vals, y_vals)

# Среднеквадратичное приближение
def build_basis_matrix(x):
    Phi = np.vstack([
        np.ones_like(x),
        x * (1 - x),
        x**2 * (1 - x),
        x**3 * (1 - x),
        x**4 * (1 - x)
    ]).T
    return Phi

Phi = build_basis_matrix(x_vals)
coeffs_ls, _, _, _ = lstsq(Phi, y_vals, rcond=None)

def least_squares_func(x):
    basis = build_basis_matrix(x)
    return basis @ coeffs_ls

x_plot = np.linspace(0.1, 1.0, 400)
plt.figure(figsize=(12, 7))
#plt.plot(x_vals, y_vals, 'ro', label='Табличные данные')
plt.plot(x_plot, newton_poly(x_plot, x_vals, coeffs_newton), label='Полином Ньютона')
plt.plot(x_plot, spline(x_plot), label='Параболический сплайн')
plt.plot(x_plot, least_squares_func(x_plot), label='Среднеквадратичное приближение')
plt.legend()
plt.title('Интерполяция и приближение функции y(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

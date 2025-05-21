import numpy as np
import matplotlib.pyplot as plt

#Система уравнений
def f1(x, y):
    return 2 * x**3 - y**2 - 1

def f2(x, y):
    return x * y**3 - y - 4

# Нахождение частных производных
def df1_dx(x, y):
    return 6 * x**2

def df1_dy(x, y):
    return -2 * y

def df2_dx(x, y):
    return y**3

def df2_dy(x, y):
    return 3 * x * y**2 - 1

#Якобиан и метод Ньютона
def newton_method(x0, y0, tol=1e-5, max_iter=10):
    x, y = x0, y0
    for _ in range(max_iter):
        F = np.array([f1(x, y), f2(x, y)])
        #Якобиан
        J = np.array([
            [df1_dx(x, y), df1_dy(x, y)],
            [df2_dx(x, y), df2_dy(x, y)]
        ])

        delta = np.linalg.solve(J, F)
        x_new = x - delta[0]
        y_new = y - delta[1]

        if np.linalg.norm([x_new - x, y_new - y], ord=np.inf) < tol:
            break
        x, y = x_new, y_new
    return x_new, y_new
x0, y0 = 1.0, 2.0 #Начальное приближение
x_sol, y_sol = newton_method(x0, y0)

residual = [f1(x_sol, y_sol), f2(x_sol, y_sol)] #Невязка
print(f"Решение: x = {x_sol:.4f}, y = {y_sol:.4f}")
print(f"Невязка: f1 = {residual[0]:.5e}, f2 = {residual[1]:.5e}")

x_vals = np.linspace(0.5, 2, 400) # График
y_vals = np.linspace(1, 2.5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
F1 = f1(X, Y)
F2 = f2(X, Y)

plt.figure(figsize=(8, 6))
plt.contour(X, Y, F1, levels=[0], colors='blue', linewidths=2)
plt.contour(X, Y, F2, levels=[0], colors='red', linestyles='dashed', linewidths=2)
plt.plot(x_sol, y_sol, 'ko', label='Решение')
plt.xlabel('x')
plt.ylabel('y')
plt.title('График уровня функций системы')
plt.legend(['Решение', 'f1=0 (синяя)', 'f2=0 (красная)'])
plt.grid(True)
plt.show()
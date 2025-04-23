import numpy as np
import matplotlib.pyplot as plt

def phi(x): #Функция метода простой итерации
    return np.cos(0.387 * x) ** 2

def f(x): # Фун. для подсчета невязки
    return np.sqrt(x) - np.cos(0.387 * x)

x0 = 0
tolerance = 1e-4
max_iterations = 100

iterations = [x0] #Список для хранения всех итераций

for _ in range(max_iterations):
    x1 = phi(iterations[-1])
    iterations.append(x1)
    if abs(iterations[-1] - iterations[-2]) < tolerance:
        break

x_final = iterations[-1]
residual = abs(f(x_final))

print(f"Решение: x = {x_final:.6f}")
print(f"Невязка: |f(x)| = {residual:.6e}")
print(f"Количество итераций: {len(iterations) - 1}")

x_values = np.linspace(0, 1.5, 400)
y_values = np.sqrt(x_values) - np.cos(0.387 * x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label=r'$f(x) = \sqrt{x} - \cos(0.387x)$')
plt.axhline(0, color='gray', linestyle='--')
plt.scatter(x_final, f(x_final), color='red', label=f'Приближённый корень: x ≈ {x_final:.4f}')
plt.title('График функции и приближённого корня')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
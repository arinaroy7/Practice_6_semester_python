import numpy as np
import matplotlib.pyplot as plt

epsilon = 1e-3

true_value = 2**0.5 # Значение для невязки
x0 = 1.5 # Начальное приближение
a = 2

def newton_sqrt(a, x0, epsilon): # Метод Ньютона для нахождения корня
    approximations = [x0]
    residuals = [abs(x0**2 - a)]  # Считаем невязку

    while True:
        x_next = 0.5 * (x0 + a / x0)
        approximations.append(x_next) #Добавляем новое приближение
        residual = abs(x_next**2 - a) #Вычисляем невязку
        residuals.append(residual)
        if abs(x_next - x0) < epsilon:
            break
        x0 = x_next

    return approximations, residuals

approximations, residuals = newton_sqrt(a, x0, epsilon)

#Выводим таблицу итераций и невязок
for i, (x, res) in enumerate(zip(approximations, residuals)):
    print(f"Итерация {i}: x = {x:.6f}, Невязка = {res:.6e}")

print(f"\nПриближенное значение √2: {approximations[-1]:.6f} ± {epsilon}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(len(residuals)), residuals, marker='o')
plt.yscale('log')
plt.xlabel('Номер итерации')
plt.ylabel('Невязка |x^2 - a|')
plt.title('Сходимость метода Ньютона')
plt.grid(True)

# График 2: Поведение функции f(x) = x^2 - a
plt.subplot(1, 2, 2)
f_values = [x**2 - a for x in approximations] #Вычисляем f(x)
plt.plot(approximations, f_values, marker='s', color='green')
plt.axhline(0, color='gray', linestyle='--') #Линия y=0
plt.xlabel('x (приближения)')
plt.ylabel('f(x) = x^2 - a')
plt.title('Поведение функции f(x) при приближениях')
plt.grid(True)

plt.tight_layout()
plt.show()
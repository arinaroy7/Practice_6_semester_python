# 1 пример: Метод простой итерации, метод нахождения скалярных произведений
# import matplotlib.pyplot as plt
import numpy as np

A = np.array([
    [8, 1, 1],
    [-1, 4, 1],
    [1, 1, 25]
])

#Начальный вектор
x = np.array([1.0, 1.0, 1.0])
eps = 0.4
max_iterations = 100
lambda_old = 0
lambdas = []

#Итерационный процесс (степенной метод)
for iteration in range(max_iterations):
    x_new = A @ x
    lambda_new = np.max(np.abs(x_new)) 
    x_new = x_new / lambda_new

    lambdas.append(lambda_new)

    if abs(lambda_new - lambda_old) < eps:
        break

    x = x_new
    lambda_old = lambda_new

y = x_new

#Невязка
residual = A @ y - lambda_new * y
residual_norm = np.linalg.norm(residual)

print(f"Максимальное по модулю собственное значение λ ≈ {lambda_new:.4f}")
print("Собственный вектор y:")
print(y)
print("\nНевязка r = A y - λ y:")
print(residual)
print(f"Норма невязки: {residual_norm:.4e}")

#Построение графика сходимости λ
plt.plot(lambdas, marker='o')
plt.axhline(lambda_new, color='r', linestyle='--', label=f'λ ≈ {lambda_new:.2f}')
plt.title("Сходимость оценки собственного значения")
plt.xlabel("Итерация")
plt.ylabel("λ")
plt.grid(True)
plt.legend()
plt.show()
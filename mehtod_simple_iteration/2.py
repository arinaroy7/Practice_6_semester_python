import numpy as np
# import matplotlib.pyplot as plt

A = np.array([
    [8, -1, 1],
    [1, 4, 1],
    [1, 1, 25]
])

x = np.array([1.0, 1.0, 1.0])
epsilon = 1e-1
lambdas = []
residuals = []

def normalize(v):
    return v / np.linalg.norm(v)

for i in range(100):
    x_new = A @ x
    y = A @ x_new
    lam = np.dot(x_new, y) / np.dot(x_new, x_new)
    lambdas.append(lam)
    
    #Вычисляем невязку: ||A*x - λ*x||
    residual = np.linalg.norm(A @ x_new - lam * x_new)
    residuals.append(residual)
    
    if i > 0 and abs(lambdas[-1] - lambdas[-2]) < epsilon:
        break
    
    x = x_new

eigenvector = normalize(x_new)
eigenvalue = lambdas[-1]

print(f"Максимальное собственное значение: {eigenvalue:.2f}")
print(f"Собственный вектор (нормализованный):\n{eigenvector}")
print(f"Невязка: {residuals[-1]:.2e}")

#Строим график
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(lambdas, marker='o')
plt.title("Сходимость собственного значения")
plt.xlabel("Итерация")
plt.ylabel("λ")

plt.subplot(1, 2, 2)
plt.semilogy(residuals, marker='o', color='r')
plt.title("Невязка на каждой итерации")
plt.xlabel("Итерация")
plt.ylabel("||Ax - λx||")

plt.tight_layout()
plt.grid(True)
plt.show()
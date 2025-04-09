# Метод Крылова пример номер 3
import numpy as np
# import matplotlib.pyplot as plt

A = np.array([[8, 1, 1],
              [-1, 4, 1],
              [1, 1, 25]], dtype=float)

x0 = np.array([1, 1, 1], dtype=float)

#векторы Крылова
x1 = A @ x0
x2 = A @ x1
x3 = A @ x2

C = np.column_stack((x1, x2, x3))
RHS = x3

p = np.linalg.solve(C, RHS)

p1, p2, p3 = p

print(f"\nХарактеристическое уравнение: λ³ - {p1:.2f}λ² + {p2:.2f}λ - {p3:.2f} = 0")

coeffs = [1, -p1, p2, -p3]
roots = np.roots(coeffs)
lambdas = np.sort(roots)[::-1]

print("\nСобственные значения:")
for i, l in enumerate(lambdas, 1):
    print(f"λ{i} = {l:.4f}")

#базис Крылова
x1 = A @ x0
x2 = A @ x1
x3 = A @ x2
X = np.column_stack((x1, x2, x3))

#Вычисляю собственные вектора
def compute_eigenvector(lmbda, p):
    beta1 = 1
    beta2 = lmbda - p[0]
    beta3 = beta2 * lmbda - p[1]
    y = beta1 * x1 + beta2 * x2 + beta3 * x3
    return y / y[-1] 

eigenvectors = []
residuals = []

print("\nСобственные векторы и невязки:")
for i, lmbda in enumerate(lambdas, 1):
    y = compute_eigenvector(lmbda, p)
    r = A @ y - lmbda * y
    res_norm = np.linalg.norm(r)
    eigenvectors.append(y)
    residuals.append(res_norm)
    print(f"y{i} = {np.round(y, 4)}")
    print(f"‖Ay - λy‖ = {res_norm:.2e}\n")

#строим график и невязку проверяем
plt.figure(figsize=(8, 5))
plt.semilogy(range(1, 4), residuals, marker='o', linestyle='--', color='blue')
plt.title("Невязка ‖Ay - λy‖ для каждого собственного значения")
plt.xlabel("Номер собственного значения")
plt.ylabel("Невязка (в лог. масштабе)")
plt.grid(True, which='both', linestyle=':')
plt.xticks([1, 2, 3])
plt.show()
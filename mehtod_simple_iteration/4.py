import numpy as np
# import matplotlib.pyplot as plt

A = np.array([
    [5, 1, 1],
    [-1, 10, 1],
    [1, 1, 12]
])

#Находим собственные значения собственных векторов
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Собственные значения:")
print(eigenvalues)

print("\nСобственные векторы (по столбцам):")
print(eigenvectors)

#Проверка
residuals = []
for i in range(len(eigenvalues)):
    λ = eigenvalues[i]
    x = eigenvectors[:, i]
    residual = A @ x - λ * x
    residuals.append(np.linalg.norm(residual))
    print(f"\nПроверка для λ = {λ:.4f}")
    print("Ax - λx =")
    print(residual)

#Визуализация: график собственных векторов
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

origin = np.zeros(3)
colors = ['r', 'g', 'b']

for i in range(3):
    vec = eigenvectors[:, i]
    ax.quiver(*origin, *vec, color=colors[i], label=f'λ = {eigenvalues[i]:.2f}', arrow_length_ratio=0.1)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_title('Собственные векторы матрицы A')
ax.legend()
plt.show()

#Построим график 
plt.figure()
plt.bar(range(1, 4), residuals, tick_label=[f"λ{i+1}" for i in range(3)], color='orange')
plt.title("Невязка |Ax - λx|")
plt.ylabel("Норма невязки")
plt.grid(True)
plt.show()

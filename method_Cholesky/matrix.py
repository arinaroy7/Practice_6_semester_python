# Решение системы линейных уравнений с помощью метода Холецкого
# Вычисление невязки и построение графика
# import numpy as np
# import matplotlib.pyplot as plt

def cholesky_decomposition(A): #Разложение матрицы A методом Холецкого
    n = len(A)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for k in range(i + 1):
            temp_sum = sum(L[i][j] * L[k][j] for j in range(k))
            if i == k:  # Элементы по диагонали
                L[i][k] = (A[i][i] - temp_sum) ** 0.5
            else:
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - temp_sum))

    return L

def cholesky_solve(L, b): #Решение системы уравнений с помощью разложения Холецкого
    n = len(L)
    L_transpose = np.transpose(L)
    y = [0.0] * n # Прямая подстановка
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]

    x = [0.0] * n # Обратная подстановка
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L_transpose[i][j] * x[j] for j in range(i + 1, n))) / L[i][i]

    return x

A = np.array([[3, 2, 1],
              [1, 2, 1],
              [4, 3, -2]]).astype(float)

b = np.array([10, 8, 4]).astype(float)

print("Матрица A:\n", A)
print("Вектор b:\n", b)

if np.all(np.linalg.eigvals(A) > 0): # Проверка положительной определённости матрицы

    L = cholesky_decomposition(A)  # Разложение Холецкого
    print("Матрица L:")
    print('\n'.join('\t'.join(map(str, row)) for row in L))

    L_transpose = np.transpose(L)
    print("Транспонированная матрица L^T:")
    print('\n'.join('\t'.join(map(str, row)) for row in L_transpose))

    x = cholesky_solve(L, b)
    print("Решение методом Холецкого:", x)

    y = np.linalg.solve(A, b)
    print("Решение через linalg.solve:", y)  # Проверка решения через встроенную функцию NumPy
else:
    print("Матрица не является положительно определённой, метод Холецкого не применим.")
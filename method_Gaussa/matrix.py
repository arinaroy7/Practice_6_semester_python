import numpy as np
import math

def gauss_elimination(A, b):
  n = len(A)
  A = np.hstack([A, b.reshape(-1, 1)])

    #прямой ход
  for i in range(n):
    max_el = abs(A[i][i])
    max_row = i
    for k in range(i+1, n):
      if abs(A[k][i]) > max_el:
        max_el =abs(A[k][i])
        max_row = k
    A[[i, max_row]] = A[[max_row, i]]

    #зануляем элементы
    for k in range(i+1, n):
        factor = A[k][i] / A[i][i] if A[i][i] !=0 else 0
        for j in range (i, n+1):
          if i == j:
            A[k][j] = 0
          else:
            A[k][j] -= factor * A[i][j]

  #обратный ход для нахождения решения
  x = np.zeros(n)
  for i in range(n-1, -1, -1):
    x[i] = A[i][n] / A[i][i] if A[i][i] !=0 else 0
    for k in range(i-1, -1, -1): #идем в обратном порядке
        if not np.isnan(x[i]):
            A[k][i] -= A[k][i] * x[i]
  return x 

# Определяем нашу матрицу
A = np.array([[3, 2, 1],
              [1, 2, 1],
              [4, 3, -2]]).astype(float)

b = np.array([10, 8, 4]).astype(float)

print("A =", A)
print("B =", b)

# Решаем систему методом Гаусса 

x = gauss_elimination(A, b)
print(x, "- решение методом Гаусса")

y = np.linalg.solve(A, b)
print(y, "- решение через linalg.solve")

# Вычисление невязки 

n_values = np.arange(0, 100)
norm_values = []

for n in n_values:
  A = np.random.rand(n, n).astype(float) #генерируем случайную квадратную матрицу n*n
  b = np.random.rand(n).astype(float)

  x = gauss_elimination(A, b)
  eps = np.linalg.norm(b - np.dot(A, x))
  norm_values.append(eps)

# корректируем масштабирование значений невязки 

l = len(norm_values)
for i in range(l):
  norm_values[i] = round(norm_values[i] * pow(10, 12), 2) 
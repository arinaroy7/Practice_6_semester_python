# Написать три программы, первая вычисления квадратного корня, вторая построения графика функции "На Берлин", третья - построение графика
# import numpy as np
# import matplotlib.pyplot as plt

print(f"Задание 1:")
def square_root(a, x0=5.0, tolerance=1e-10, max_iterations=100):
    if a <= 0:
        raise ValueError("a должно быть больше нуля.")

    x_n = x0
    for _ in range(max_iterations):
        x_next = 0.5 * (x_n + a / x_n)
        if abs(x_next - x_n) < tolerance:
            return x_next
        x_n = x_next

    return x_n

a = float(input("Введите число a (a > 0): "))
result = square_root(a)
print(f"Квадратный корень из {a} равен {result}")

print(f"Задание 2 - На Берлин :")
def x(t, k):
    return 1 - k * t

k_values = [1, 2, 3, 4, 5, 6, 7]
colors = ['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']

plt.figure(figsize=(10, 6))

for k, color in zip(k_values, colors):
    t_values = np.linspace(1/(k + 1), 1/k, 100)
    x_values = x(t_values, k)

    plt.plot(t_values, x_values, label=f'k = {k}', color=color)

plt.title('График функции x(t) = 1 - kt')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')
plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.legend()
plt.grid()
plt.show()

print(f"Задание 3:")

def x(t):
    if -1 < t < 1:
        return np.exp(1 / (t**2 - 1))
    else:
        return 0

t_values = np.linspace(-2, 2, 400)
x_values = [x(t) for t in t_values]

plt.figure(figsize=(10, 6))
plt.plot(t_values, x_values, label='x(t)', color='darkorange')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.title('График функции x(t)')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.xlim(-2, 2)
plt.ylim(-0.1, 2)
plt.grid()
plt.legend()
plt.show()

print(f"Задание №4:")

def find_root(x0, epsilon, max_iterations=1000):
    x = np.cos(x0)
    for _ in range(max_iterations):
        new_x = np.cos(x)
        if abs(new_x - x) < epsilon:
            return new_x
        x = new_x
    return x

epsilon = float(input("Введите значение точности : "))
x0 = 1
root = find_root(x0, epsilon)
print(f"Найденный корень: {root}")

x_values = np.linspace(-1, 1, 400)
y_values = np.cos(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='y = cos(x)', color='mediumseagreen')
plt.plot(x_values, x_values, label='y = x', color='crimson', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.title('График функции x = cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid()
plt.axvline(root, color='blue', linestyle=':', label=f'Корень: {root:.5f}')
plt.legend()
plt.show()
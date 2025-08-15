import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
a, b = 0, 10
n = 100000
h = (b - a) / n
x = np.linspace(a, b, n+1)

# f(x, y) da EDO
def f(x, y):
    return np.sin(np.pi * x) - y

# Solução analítica
def y_analitico(x):
    C = 1 - (-np.pi)/(1 + np.pi**2)
    return (np.sin(np.pi * x) - np.pi * np.cos(np.pi * x)) / (1 + np.pi**2) + C * np.exp(-x)

# Inicialização
y = np.zeros(n+1)
y[0] = 1

# Primeiro passo (Euler simples para começar)
y[1] = y[0] + h * f(x[0], y[0])

# Euler central
for i in range(1, n):
    y[i+1] = y[i-1] + 2*h*f(x[i], y[i])

# Comparação
y_exato = y_analitico(x)

plt.figure(figsize=(10,6))
plt.plot(x, y_exato, label='Solução Analítica', color='red')
plt.plot(x, y, '--', label='Euler Central', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparação: Solução Analítica vs Euler Central')
plt.grid(True)
plt.legend()
plt.show()
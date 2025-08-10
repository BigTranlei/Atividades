import numpy as np
import matplotlib.pyplot as plt

# parâmetros
h = 0.1
T = 5.0
N = int(T / h)
x = np.zeros((N+1, 2))

# condição inicial (x1 = y, x2 = y')
x[0, :] = [1.0, 3.0]

# função do sistema y'' + 4 y' + 4 y = 0
def f(x):
    x1, x2 = x
    return np.array([x2, -4.0*x1 - 4.0*x2])

# Euler explícito
for n in range(N):
    x[n+1] = x[n] + h * f(x[n])

# pontos de tempo para a solução exata
t = np.linspace(0, T, N+1)
y_exact = (1.0 + 5.0*t) * np.exp(-2.0*t)
yp_exact = (3.0 - 10.0*t) * np.exp(-2.0*t)

# grade para o campo de vetores
x1_vals = np.linspace(-3, 3, 20)   # y (vertical no seu plot)
x2_vals = np.linspace(-6, 6, 20)   # y' (horizontal no seu plot)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Campo de vetores (dX1 = x2, dX2 = -4 x1 -4 x2)
dX1 = X2
dX2 = -4*X1 - 4*X2
U = dX2  # variação de x2
V = dX1  # variação de x1

# normalizar para visualização
Nrm = np.sqrt(U**2 + V**2)
Nrm[Nrm == 0] = 1.0
U2 = U / Nrm
V2 = V / Nrm

plt.figure(figsize=(8, 6))
plt.quiver(X2, X1, U2, V2, angles='xy', alpha=0.6)

# trajetória aproximada por Euler (x[:,0]=y, x[:,1]=y')
plt.plot(x[:, 1], x[:, 0], '-o', markersize=3, label=f'Euler h={h}')

# trajetória exata (y' horizontal, y vertical)
plt.plot(yp_exact, y_exact, '-', lw=2, label='Solução exata')

plt.plot(x[0, 1], x[0, 0], 'ks', label='início (1,3)')
plt.xlabel("x2 = y'")
plt.ylabel("x1 = y")
plt.title("Plano de fase (y', y)")
plt.legend()
plt.grid(True)
plt.xlim(-6, 6)
plt.ylim(-3, 3)
plt.show()
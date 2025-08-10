import numpy as np
import matplotlib.pyplot as plt

# parâmetros de integração
h = 0.1
T = 10
t = np.arange(0, T + h, h)
N = len(t)

# condições iniciais
y0 = 1
dy0 = 0

# solução exata (para comparar)
y_ex = np.cos(2*t)
dy_ex = -2*np.sin(2*t)

# ============================================================
# Método de Euler progressivo
# ============================================================
y_f = np.zeros(N)
dy_f = np.zeros(N)
y_f[0] = y0
dy_f[0] = dy0

for n in range(N-1):
    y_f[n+1] = y_f[n] + h * dy_f[n]
    dy_f[n+1] = dy_f[n] - 4*h * y_f[n]

# ============================================================
# Método de Euler regressivo
# ============================================================
y_b = np.zeros(N)
dy_b = np.zeros(N)
y_b[0] = y0
dy_b[0] = dy0

# Matriz do sistema linear
M = np.array([[1, -h],
              [4*h, 1]])
M_inv = np.linalg.inv(M)

for n in range(N-1):
    Xn = np.array([y_b[n], dy_b[n]])
    y_b[n+1], dy_b[n+1] = M_inv @ Xn

# ============================================================
# Diferença central (2ª ordem)
# ============================================================
y_c = np.zeros(N)
dy_c = np.zeros(N)
y_c[0] = y0
dy_c[0] = dy0

# passo inicial usando Taylor
y_c[1] = y0 + h*dy0 + 0.5*h*h*(-4*y0)

for n in range(1, N-1):
    y_c[n+1] = 2*y_c[n] - y_c[n-1] - 4*h*h*y_c[n]

# aproximação de y'
for n in range(N-1):
    dy_c[n] = (y_c[n+1] - y_c[n]) / h
dy_c[-1] = dy_c[-2]

# ============================================================
# Campo vetorial do sistema
# ============================================================
x1_vals = np.linspace(-2.5, 2.5, 20)
x2_vals = np.linspace(-4, 4, 20)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Sistema: x1' = x2, x2' = -4 x1
U = X2
V = -4*X1

# ============================================================
# Gráfico: Plano de fase
# ============================================================
plt.figure(figsize=(9, 6))
plt.quiver(X2, X1, V, U, angles='xy', scale_units='xy',
           scale=15, alpha=0.4, color='gray')

plt.plot(dy_f, y_f, 'o-', ms=3, label='Euler prog.')
plt.plot(dy_b, y_b, 's-', ms=3, label='Euler reg.')
plt.plot(dy_c, y_c, '^-', ms=3, label='Dif. central')
plt.plot(dy_ex, y_ex, 'k-', lw=2, label='Solução exata')

plt.scatter([dy_f[0]], [y_f[0]], c='k', marker='s', label="início")

plt.xlabel("x2 = y'")
plt.ylabel("x1 = y")
plt.title(f"Plano de fase — T = {T}, h = {h}")
plt.legend(loc='upper right')
plt.xlim(-4, 4)
plt.ylim(-2.5, 2.5)
plt.gca().set_aspect('equal')
plt.grid(True)
plt.show()
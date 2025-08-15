import numpy as np
import matplotlib.pyplot as plt

# Coeficientes da EDO
a = 5.0
b = 4.0

# Condições iniciais
y0 = 1.0
ypp0 = 3.0
yp0 = -(ypp0 + b*y0) / a

# Parâmetros
h = 0.01
t_max = 5.0
t = np.arange(0, t_max + h, h)
N = len(t)

# Vetores
y = np.zeros(N)
v = np.zeros(N)
y[0] = y0
v[0] = yp0

# Funções do sistema


def f_y(v):
    return v


def f_v(y, v):
    return -a*v - b*y


# Método RK4
for n in range(N-1):
    k1y = f_y(v[n])
    k1v = f_v(y[n], v[n])

    k2y = f_y(v[n] + 0.5*h*k1v)
    k2v = f_v(y[n] + 0.5*h*k1y, v[n] + 0.5*h*k1v)

    k3y = f_y(v[n] + 0.5*h*k2v)
    k3v = f_v(y[n] + 0.5*h*k2y, v[n] + 0.5*h*k2v)

    k4y = f_y(v[n] + h*k3v)
    k4v = f_v(y[n] + h*k3y, v[n] + h*k3v)

    y[n+1] = y[n] + (h/6)*(k1y + 2*k2y + 2*k3y + k4y)
    v[n+1] = v[n] + (h/6)*(k1v + 2*k2v + 2*k3v + k4v)

# Solução analítica
r1, r2 = np.roots([1, a, b])
C2 = (yp0 - r1*y0)/(r2 - r1)
C1 = y0 - C2
y_a = C1*np.exp(r1*t) + C2*np.exp(r2*t)
v_a = r1*C1*np.exp(r1*t) + r2*C2*np.exp(r2*t)

# Gráficos
plt.figure(figsize=(10, 4))
plt.plot(t, y, label='RK4')
plt.plot(t, y_a, '--', label='Analítico')
plt.legend()
plt.grid()
plt.xlabel('t')
plt.ylabel('y')

plt.figure(figsize=(6, 6))
plt.plot(y, v, label='RK4')
plt.plot(y_a, v_a, '--', label='Analítico')
plt.xlabel('y')
plt.ylabel("y'")
plt.grid()
plt.axis('equal')
plt.legend()
plt.show()

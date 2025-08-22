import numpy as np
import matplotlib.pyplot as plt

# parâmetros do retângulo
a = 1.0   # comprimento em x
b = 1.0   # comprimento em y
N_terms = 50  # número de termos na série de Fourier

# função de contorno na borda x=a
def f(y):
    return np.sin(np.pi * y)  # pode trocar por outra função

# grade de pontos
nx, ny = 100, 100
x = np.linspace(0, a, nx)
y = np.linspace(0, b, ny)
X, Y = np.meshgrid(x, y)

# cálculo da solução pela série
U = np.zeros_like(X)

for n in range(1, N_terms+1):
    # coeficiente A_n
    # integral de f(y)*sin(n pi y/b) de 0 a b
    ys = np.linspace(0, b, 1000)
    integrand = f(ys) * np.sin(n*np.pi*ys/b)
    An = (2/b) * np.trapz(integrand, ys) / np.sinh(n*np.pi*a/b)
    
    # contribuição do termo n
    U += An * np.sinh(n*np.pi*X/b) * np.sin(n*np.pi*Y/b)

# gráfico de contorno preenchido
plt.figure(figsize=(6,5))
contour = plt.contourf(X, Y, U, levels=30, cmap="inferno")
plt.colorbar(contour, label="u(x,y)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solução numérica da equação de Laplace (Problema de Dirichlet)")
plt.show()
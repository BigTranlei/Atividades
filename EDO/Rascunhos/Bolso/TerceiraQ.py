import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Definição da EDO
# ==============================
def analytical_solution(x):
    """Solução analítica do PVI."""
    return np.cos(2 * x)

def numerical_method(h, method="n"):
    """
    Resolve a EDO y'' + 4y = 0 para diferentes aproximações:
    Euler = "n", "n+1" ou "n-1"
    """
    x = np.arange(0, 10, h)
    y = np.zeros(len(x))
    y[0] = 1
    y[1] = 1  # Euler inicial

    for i in range(2, len(x)):
        if method == "n":       # n*
            y[i] = (2 - 4*h**2) * y[i-1] - y[i-2]
        elif method == "n+1":
            y[i] = (2*y[i-1] - y[i-2]) / (1 + 4*h**2)
        elif method == "n-1":
            y[i] = 2*y[i-1] - (4*h**2 + 1) * y[i-2]
        else:
            raise ValueError("Método inválido. Use 'n', 'n+1' ou 'n-1'.")
    return x, y

# ==============================
# Plotagem
# ==============================
def plot_comparison(h, method):
    xa = np.arange(0, 10, 0.001)
    ya = analytical_solution(xa)
    xn, yn = numerical_method(h, method)

    plt.plot(xa, ya, label="Analítica", color='red', linewidth=2)
    plt.plot(xn, yn, label=f"Numérica ({method})", color='blue', linestyle='--', linewidth=1.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Comparação - Método {method}")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

# ==============================
# Execução
# ==============================
plot_comparison(0.01, method="n")
plot_comparison(0.01, method="n+1")
plot_comparison(0.01, method="n-1")
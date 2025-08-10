import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ======================== Funções utilitárias =========================

def central_difference_solution(T, h, y0=1.0, yp0=0.0):
    """
    Resolve y'' + 4y = 0 usando esquema de diferença central
    com passo h e condições iniciais y(0)=y0, y'(0)=yp0.
    """
    t = np.arange(0, T + 1e-12, h)
    N = len(t)
    y = np.zeros(N)
    y[0] = y0
    ypp0 = -4.0 * y0
    # Taylor para o primeiro passo
    if N > 1:
        y[1] = y0 + h*yp0 + 0.5*h*h*ypp0
    # Recorrência central
    for n in range(1, N-1):
        y[n+1] = 2.0*y[n] - y[n-1] + h*h * (-4.0 * y[n])
    return t, y


def compute_error(y_num, t):
    """Erro RMS (norma L2) entre solução numérica e exata."""
    y_exact = np.cos(2*t)
    E = np.sqrt(np.sum((y_exact - y_num)**2) / len(t))
    return E


# ======================== Código principal ============================
T = 10.0
hs = [0.1, 0.05, 0.025, 0.0125]
errors = []

# Calcula erros para cada passo
for h in hs:
    t, y_num = central_difference_solution(T, h)
    E = compute_error(y_num, t)
    errors.append(E)

# Estima ordem p
ps = []
for i in range(len(hs)-1):
    ratio_E = errors[i+1] / errors[i]
    ratio_h = hs[i+1] / hs[i]
    p = np.log(ratio_E) / np.log(ratio_h)
    ps.append(p)

# Monta tabela de resultados
df = pd.DataFrame({
    "h": hs,
    "E(L2,RMS)": errors,
    "p (entre h_i e h_{i+1})": ps + [np.nan]
})
print(df.round(10))

# ======================== Plot Erro x h (log-log) =====================
plt.figure(figsize=(7, 5))
plt.loglog(hs, errors, 'o-', label='Erro RMS (Diferença Central)')
# reta de referência h^2
ref_line = errors[0]*(np.array(hs)/hs[0])**2
plt.loglog(hs, ref_line, '--', label='Referência $h^2$')
plt.gca().invert_xaxis()
plt.xlabel('Passo $h$')
plt.ylabel('Erro RMS')
plt.title('Convergência - Diferença Central')
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()
plt.show()

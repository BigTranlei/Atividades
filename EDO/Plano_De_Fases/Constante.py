import numpy as np
import matplotlib.pyplot as plt
import math

# ==============================
# Função de teste e derivada exata
# ==============================
def f(x):
    return np.sin(x)

def df_exact(x):
    return np.cos(x)

# ==============================
# Diferenciação por diferença central
# ==============================
def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2*h)

# ==============================
# Cálculo do erro L2 RMS
# ==============================
hs = [0.1, 0.05, 0.025, 0.0125]
errors = []

for h in hs:
    x_vals = np.arange(0, 2*np.pi, h)
    df_num = central_diff(f, x_vals, h)
    df_true = df_exact(x_vals)
    e = df_num - df_true
    E_L2_RMS = np.sqrt(np.sum(e**2) / len(e))
    errors.append(E_L2_RMS)

# ==============================
# Cálculo da ordem p entre pares de h
# ==============================
ps = []
for i in range(len(errors) - 1):
    p_val = np.log(errors[i+1] / errors[i]) / np.log(hs[i+1] / hs[i])
    ps.append(p_val)
ps.append(np.nan)  # último não tem p

# ==============================
# Ajuste log–log para p e C
# ==============================
log_h = np.log(hs)
log_E = np.log(errors)
slope, intercept = np.polyfit(log_h, log_E, 1)
p_fit = slope
C_fit = math.exp(intercept)

# C calculada ponto a ponto
C_is = [errors[i] / (hs[i]**p_fit) for i in range(len(hs))]

# ==============================
# Impressão formatada
# ==============================
print(f"{'h':>10} {'E(L2,RMS)':>15} {'p (entre h_i e h_{i+1})':>25}")
for i in range(len(hs)):
    p_str = f"{ps[i]:.6f}" if not np.isnan(ps[i]) else "-"
    print(f"{hs[i]:10.4f} {errors[i]:15.6e} {p_str:>25}")

print("\nAjuste log–log:")
print(f"p ≈ {p_fit:.6f}")
print(f"C ≈ {C_fit:.6f}")
print(f"C_i (ponto a ponto) = {C_is}")

# ==============================
# Gráfico log–log
# ==============================
plt.figure()
plt.loglog(hs, errors, 'o-', label='Erro L2 RMS')
plt.loglog(hs, [C_fit*(h**p_fit) for h in hs], '--', label=f"Ajuste: $C h^p$\nC={C_fit:.3f}, p={p_fit:.3f}")
plt.loglog(hs, [hs[0]**2 * errors[0] / (hs[0]**2) * (h/hs[0])**2 for h in hs], ':', label='$\propto h^2$')
plt.xlabel('h')
plt.ylabel('Erro L2 RMS')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Métodos numéricos
# =========================
def euler_forward(f, t0, y0, v0, h, T):
    t = np.arange(t0, T+h, h)
    y = np.zeros_like(t)
    v = np.zeros_like(t)
    y[0], v[0] = y0, v0
    for n in range(len(t)-1):
        y[n+1] = y[n] + h*v[n]
        v[n+1] = v[n] + h*f(t[n], y[n], v[n])
    return t, y, v

def euler_backward(f, t0, y0, v0, h, T, tol=1e-10, maxit=20):
    t = np.arange(t0, T+h, h)
    y = np.zeros_like(t)
    v = np.zeros_like(t)
    y[0], v[0] = y0, v0
    for n in range(len(t)-1):
        Y, V = y[n], v[n]
        for _ in range(maxit):
            Y_new = y[n] + h*V
            V_new = v[n] + h*f(t[n+1], Y, V)
            if np.linalg.norm([Y_new-Y, V_new-V]) < tol:
                break
            Y, V = Y_new, V_new
        y[n+1], v[n+1] = Y, V
    return t, y, v

def central_diff(f, t0, y0, v0, h, T):
    t = np.arange(t0, T+h, h)
    y = np.zeros_like(t)
    y[0] = y0
    y[1] = y0 + h*v0 + 0.5*h**2*f(t0, y0, v0)
    for n in range(1, len(t)-1):
        v_nm = (y[n] - y[n-1]) / h
        y[n+1] = 2*y[n] - y[n-1] + h**2*f(t[n], y[n], v_nm)
    v = np.zeros_like(t)
    v[1:-1] = (y[2:] - y[:-2])/(2*h)
    v[0] = v0
    v[-1] = v[-2]
    return t, y, v

# =========================
# Funções auxiliares
# =========================
def erro_rms(y_num, t, y_exata):
    return np.sqrt(np.mean((y_exata(t) - y_num)**2))

def estimar_C_p(metodo, f, y_exata, y0, v0, T):
    hs = [0.1, 0.05, 0.025, 0.0125]
    erros = []
    for h in hs:
        t, y, v = metodo(f, 0, y0, v0, h, T)
        erros.append(erro_rms(y, t, y_exata))
    log_h = np.log(hs)
    log_E = np.log(erros)
    p, logC = np.polyfit(log_h, log_E, 1)
    return np.exp(logC), p, hs, erros

def campo_vetorial_auto(f_sys, ys, vs, dens=20):
    y_min, y_max = ys.min(), ys.max()
    v_min, v_max = vs.min(), vs.max()
    marg_y = 0.1*(y_max - y_min)
    marg_v = 0.1*(v_max - v_min)
    y_vals = np.linspace(y_min - marg_y, y_max + marg_y, dens)
    v_vals = np.linspace(v_min - marg_v, v_max + marg_v, dens)
    Y, V = np.meshgrid(y_vals, v_vals)
    dY = V
    dV = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            dV[i,j] = f_sys(0, Y[i,j], V[i,j])
    norm = np.sqrt(dY**2 + dV**2)
    norm[norm == 0] = 1
    return Y, V, dY/norm, dV/norm

# =========================
# DEFINA SUA EDO AQUI
# =========================
# Exemplo: y'' + 4y' + 5y = 2
#f = lambda t, y, v: -4*v - 5*y + 2

# Exemplo: y'' + 4 y' = 0  =>  y'' = -4 y'
f = lambda t, y, v: -4.0 * v

# Se tiver solução exata, defina-a; senão, use None
#y_exata = None
#v_exata = None

# Se souber a solução exata, defina-a (opcional)
# Para y'' + 4 y' = 0 com y(0)=1, y'(0)=0 a solução exata é y(t)=1
y_exata = lambda t: 1.0 + 0*t
v_exata = lambda t: 0.0 + 0*t

# Condições iniciais e parâmetros
y0, v0 = 1.0, 3.0
T, h = 10.0, 0.1

# =========================
# Cálculo numérico
# =========================
t1, y1, v1 = euler_forward(f, 0, y0, v0, h, T)
t2, y2, v2 = euler_backward(f, 0, y0, v0, h, T)
t3, y3, v3 = central_diff(f, 0, y0, v0, h, T)

# Se existir solução exata
if y_exata is not None:
    t_exact = np.linspace(0, T, 1000)
    y_exact_vals = y_exata(t_exact)
    v_exact_vals = v_exata(t_exact) if v_exata is not None else None
else:
    t_exact = None
    y_exact_vals = None
    v_exact_vals = None

# Campo vetorial cobrindo todo o plano de fase
ys_all = np.concatenate([y1, y2, y3] + ([y_exact_vals] if y_exact_vals is not None else []))
vs_all = np.concatenate([v1, v2, v3] + ([v_exact_vals] if v_exact_vals is not None else []))
Y, V, dY, dV = campo_vetorial_auto(f, ys_all, vs_all, dens=25)

# Plano de fase
plt.figure(figsize=(8,6))
plt.quiver(Y, V, dY, dV, angles='xy', alpha=0.3, color='gray')
plt.plot(y1, v1, 'b-o', markevery=5, label='Euler prog.')
plt.plot(y2, v2, 'orange', marker='s', markevery=5, label='Euler reg.')
plt.plot(y3, v3, 'g-', marker='.', markevery=5, label='Dif. central')
if y_exact_vals is not None and v_exact_vals is not None:
    plt.plot(y_exact_vals, v_exact_vals, 'k-', lw=2, label='Solução exata')
plt.plot(y0, v0, 'ks', label='início')
plt.xlabel("y")
plt.ylabel("y'")
plt.title(f"Plano de fase — T = {T}, h = {h}")
plt.legend()
plt.grid(True)

# Evolução no tempo
plt.figure()
plt.plot(t1, y1, 'b-', label='Euler prog.')
plt.plot(t2, y2, 'orange', label='Euler reg.')
plt.plot(t3, y3, 'g-', label='Dif. central')
if y_exact_vals is not None:
    plt.plot(t_exact, y_exact_vals, 'k--', label='Solução exata')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("Evolução no tempo")
plt.legend()
plt.grid(True)
plt.show()

# C e p para o método central
if y_exata is not None:
    C, p, hs, erros = estimar_C_p(central_diff, f, y_exata, y0, v0, T)
    print(f"Constante C ≈ {C:.6f}, ordem p ≈ {p:.6f}")
    print("hs:", hs)
    print("erros:", erros)

# Taxa de recorrência (linear com coef. constantes e homogênea)
if True:  # pode colocar detecção automática depois
    a = 2 - 4*h**2
    disc = a**2 - 4
    lambda1 = (a + np.sqrt(complex(disc))) / 2
    lambda2 = (a - np.sqrt(complex(disc))) / 2
    print(f"Taxas de recorrência λ1={lambda1}, λ2={lambda2}")
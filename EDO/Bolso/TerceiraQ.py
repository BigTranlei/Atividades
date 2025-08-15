import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Função para cada método
# ============================================================
def metodo_n(x_vals, h):
    y = np.zeros(len(x_vals))
    y[0] = 1
    y[1] = 1
    for j in range(2, len(x_vals)):
        y[j] = y[j-1] * (2 - 2 * h**2) - y[j-2] * (1 + h**2)
        y[j] /= (1 + h**2)
    return y

def metodo_nmais1(x_vals, h):
    y = np.zeros(len(x_vals))
    y[0] = 1
    y[1] = 1
    for j in range(2, len(x_vals)):
        aux = 2 * y[j-1] - y[j-2]
        y[j] = aux / (1 + 4 * h**2)
    return y

def metodo_nmenos1(x_vals, h):
    y = np.zeros(len(x_vals))
    y[0] = 1
    y[1] = 1
    for j in range(2, len(x_vals)):
        y[j] = 2 * y[j-1] - (4 * h**2 + 1) * y[j-2]
    return y

# ============================================================
# Lista de passos
# ============================================================
h_list = [0.01, 0.02, 0.03, 0.04, 0.05]

# Armazenamento dos erros
erro_n       = np.zeros(len(h_list))
erro_nmais1  = np.zeros(len(h_list))
erro_nmenos1 = np.zeros(len(h_list))

# ============================================================
# Loop para cada passo
# ============================================================
for i, h in enumerate(h_list):
    x_vals = np.arange(0, 2, h)
    y_ana = np.cos(2 * x_vals)

    y_n       = metodo_n(x_vals, h)
    y_nmais1  = metodo_nmais1(x_vals, h)
    y_nmenos1 = metodo_nmenos1(x_vals, h)

    # Erro médio quadrático
    erro_n[i]       = np.mean((y_ana - y_n)**2)
    erro_nmais1[i]  = np.mean((y_ana - y_nmais1)**2)
    erro_nmenos1[i] = np.mean((y_ana - y_nmenos1)**2)

# ============================================================
# Cálculo da ordem para cada método
# ============================================================
ordem_n       = np.zeros(len(h_list))
ordem_nmais1  = np.zeros(len(h_list))
ordem_nmenos1 = np.zeros(len(h_list))

for i in range(len(h_list)-1):
    ordem_n[i]       = np.log(erro_n[i+1] / erro_n[i]) / np.log(h_list[i+1] / h_list[i])
    ordem_nmais1[i]  = np.log(erro_nmais1[i+1] / erro_nmais1[i]) / np.log(h_list[i+1] / h_list[i])
    ordem_nmenos1[i] = np.log(erro_nmenos1[i+1] / erro_nmenos1[i]) / np.log(h_list[i+1] / h_list[i])

# ============================================================
# Plot Erro e Ordem
# ============================================================
plt.figure(figsize=(12, 5))

# Erro
plt.subplot(1, 2, 1)
plt.plot(h_list, erro_n, 'bo-', label="n* = n")
plt.plot(h_list, erro_nmais1, 'ro-', label="n* = n+1")
plt.plot(h_list, erro_nmenos1, 'go-', label="n* = n-1")
plt.title("Erro do Método em função do passo")
plt.xlabel("h")
plt.ylabel("Erro médio quadrático")
plt.legend()
plt.grid(True)

# Ordem
plt.subplot(1, 2, 2)
plt.plot(h_list, ordem_n, 'bo-', label="n* = n")
plt.plot(h_list, ordem_nmais1, 'ro-', label="n* = n+1")
plt.plot(h_list, ordem_nmenos1, 'go-', label="n* = n-1")
plt.title("Ordem do Método")
plt.xlabel("h")
plt.ylabel("Ordem")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
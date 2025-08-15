import numpy as np
import matplotlib.pyplot as plt


def f(t, y1, y2):
    return y2,  -4*y2 - 5*y1 #Isolar Alterar

# Parâmetros
h = 0.01
t = np.arange(0, 10, h)
y1 = np.zeros_like(t)  # posição
y2 = np.zeros_like(t)  # velocidade

# Condições iniciais Alterar
y1[0] = 1  # y(0)
y2[0] = 3 # y'(0)

# Método de Euler
for i in range(len(t)-1):
    dy1, dy2 = f(t[i], y1[i], y2[i])
    y1[i+1] = y1[i] + h * dy1
    y2[i+1] = y2[i] + h * dy2

# Solução analítica para comparação Alterar
#y_analitica = ((7/3) * np.exp(-1.0*t)) - ((4/3) * np.exp(-4.0*t)) # Distintas
#y_analitica = (1.0 + 5.0*t) * np.exp(-2.0*t) # Iguais

real, imaginario = -2, 1  # parte real e imaginária da raiz 
# Cálculo de C1 e C2
y_analitica = np.exp(real*t)*(1*np.cos(imaginario*t) + ((3 - real*1)/imaginario)*np.sin(imaginario*t))


# Plot
plt.plot(t, y1, label="Numérica (Euler)")
plt.plot(t, y_analitica, '--', label="Analítica")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
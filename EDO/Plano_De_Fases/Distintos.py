import numpy as np
import matplotlib.pyplot as plt

# parametros :

Passo = 0.1
Tempo = 10
Numero = int(Tempo/Passo)
Conjunto_Solucao = np.zeros((Numero+1, 2))  # vetor (Linhas,colunas) -> y ,y'
Conjunto_Solucao[0, :] = [1.0, 3.0]  # C.I

# def   sistema:
    #Def sistema
def f(x):
    x1, x2 = x
    return np.array([x2, -4*x1-5*x2])
    
    #Interações
for n in range(Numero):
    Conjunto_Solucao[n+1] = Conjunto_Solucao[n] + Passo*f(Conjunto_Solucao[n])

# campo
    #Grade
x1_valores = np.linspace(-6,6,20) #-> vertical
x2_valores = np.linspace(-3,3,20) #->horizontal
Conjunto_Solucao1, Conjunto_Solucao2 = np.meshgrid(x1_valores,x2_valores)
    
    #Campo vetores
dConjunto_Solucao1 = Conjunto_Solucao2
dConjunto_Solucao2 = -4*Conjunto_Solucao1-5*Conjunto_Solucao2

    #Normalizar
Nrm = np.sqrt(dConjunto_Solucao1**2 + dConjunto_Solucao2**2)
Nrm[Nrm==0] = 1.0
dConjunto_Solucao1n = dConjunto_Solucao1/Nrm 
dConjunto_Solucao2n = dConjunto_Solucao2/Nrm

# plot
plt.figure(figsize=(8,6))
plt.quiver(Conjunto_Solucao2,Conjunto_Solucao1,dConjunto_Solucao2n,dConjunto_Solucao1n, angles='xy' , alpha = 0.6)
plt.plot(Conjunto_Solucao[:,1], Conjunto_Solucao[:,0] ,'-o', markersize=3 , label = 'progressivo')
plt.plot(Conjunto_Solucao[0,1], Conjunto_Solucao[0,0], 'ks', label='início (1,3)')
plt.xlabel("x2 = y'")
plt.ylabel("x1 = y")
plt.title("Plano de fase (y' no eixo horizontal, y no vertical)")
plt.legend()
plt.grid(True)
plt.xlim(-3, 3)
plt.ylim(-6, 6)
plt.show()
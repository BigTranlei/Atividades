import numpy as np

# ==============================
# Malha 4x4
# ==============================
nx, ny = 4, 4
a, b = 3.0, 2.0   # dimensões (não precisamos de dx,dy pois é uniforme)

# Condições de contorno
V1 = 5.0   # y=0
V2 = 10.0  # x=a
V3 = 15.0  # y=b
V4 = 20.0  # x=0

# Matriz de potenciais (com bordas já impostas)
V = np.zeros((ny, nx))
V[0, :]   = V1
V[-1, :]  = V3
V[:, 0]   = V4
V[:, -1]  = V2

# ==============================
# Montagem do sistema linear
# ==============================
# Pontos internos: (1,1), (1,2), (2,1), (2,2)
# Variáveis: X = [V11, V12, V21, V22]

A = np.zeros((4,4))
b = np.zeros(4)

# Equação para (1,1)
A[0,0] = 4
A[0,1] = -1
A[0,2] = -1
b[0] = V[0,1] + V[1,0]  # vizinhos de borda

# Equação para (1,2)
A[1,1] = 4
A[1,0] = -1
A[1,3] = -1
b[1] = V[0,2] + V[1,3]

# Equação para (2,1)
A[2,2] = 4
A[2,0] = -1
A[2,3] = -1
b[2] = V[2,0] + V[3,1]

# Equação para (2,2)
A[3,3] = 4
A[3,1] = -1
A[3,2] = -1
b[3] = V[2,3] + V[3,2]

# ==============================
# Resolve o sistema
# ==============================
X = np.linalg.solve(A, b)

# Recoloca os valores na matriz V
V[1,1], V[1,2], V[2,1], V[2,2] = X

print("Matriz de Potenciais:")
print(V)
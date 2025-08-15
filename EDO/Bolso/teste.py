# We'll create a clean, readable 3D FDTD (Yee) skeleton for Maxwell's equations
# using only numpy + matplotlib. It supports:
# - Cartesian grid (x,y,z,t)
# - Isotropic materials (epsilon_r, sigma), mu = mu0
# - Source injection (soft source) at a point (Gaussian pulse)
# - Boundary conditions: PEC or Mur(1a) simple absorbing
# - Snapshots on a chosen plane for visualization (e.g., Ez at z = mid)
#
# We'll also run a *tiny* demo so it finishes quickly here.
# The solver is intentionally written to be legible over micro-optimizations.

import runpy
code = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Constantes físicas
c0  = 299792458.0
mu0 = 4e-7 * np.pi
eps0 = 1.0 / (mu0 * c0**2)

def gaussian_pulse(t, t0, spread):
    return np.exp(-((t - t0) / spread)**2)

class FDTD3D:
    """
    Solver FDTD 3D (Yee) para Maxwell em meios isotrópicos:
        ∂E/∂t = (1/eps) (∇×H) - (σ/eps) E
        ∂H/∂t = -(1/μ) (∇×E)
    - Malha Yee com passos uniformes dx, dy, dz
    - Condições de contorno: 'pec' ou 'mur' (1ª ordem simples)
    - Fonte: injeção "soft" em um ponto (em Ez por padrão)
    """
    def __init__(self, Nx, Ny, Nz, dx, dy, dz, dt, Nt,
                 eps_r=1.0, sigma=0.0,
                 bc='pec', source_loc=None, source_kind='Ez',
                 t0=None, spread=None):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.dt, self.Nt = dt, Nt
        self.bc = bc.lower()  # 'pec' ou 'mur'
        # Materiais (permitir escalar ou array do tamanho do domínio)
        if np.isscalar(eps_r):
            self.eps = eps0 * eps_r * np.ones((Nx, Ny, Nz))
        else:
            self.eps = eps0 * np.array(eps_r, dtype=float)
        if np.isscalar(sigma):
            self.sigma = sigma * np.ones((Nx, Ny, Nz))
        else:
            self.sigma = np.array(sigma, dtype=float)
        self.mu = mu0  # assumimos meios não magnéticos: mu = mu0

        # Campos (colocados na mesma grade para legibilidade; o esquema Yee padrão é implícito via diferenças)
        self.Ex = np.zeros((Nx, Ny, Nz))
        self.Ey = np.zeros((Nx, Ny, Nz))
        self.Ez = np.zeros((Nx, Ny, Nz))
        self.Hx = np.zeros((Nx, Ny, Nz))
        self.Hy = np.zeros((Nx, Ny, Nz))
        self.Hz = np.zeros((Nx, Ny, Nz))

        # Coeficientes de atualização para E (com condutividade sigma)
        self.Ce = (1.0 - 0.5 * (self.sigma * self.dt) / self.eps) / (1.0 + 0.5 * (self.sigma * self.dt) / self.eps)
        self.Ceh = self.dt / ( (1.0 + 0.5 * (self.sigma * self.dt) / self.eps) * self.eps )

        # Fonte
        if source_loc is None:
            self.src = (Nx//2, Ny//2, Nz//2)
        else:
            self.src = source_loc
        self.source_kind = source_kind  # 'Ex','Ey','Ez'
        self.t0 = t0 if t0 is not None else 6 * self.dt
        self.spread = spread if spread is not None else 2 * self.dt

        # Buffers p/ Mur (valores anteriores nas faces)
        if self.bc == 'mur':
            self.prev_Ex = self.Ex.copy()
            self.prev_Ey = self.Ey.copy()
            self.prev_Ez = self.Ez.copy()

    def curl_E(self):
        # ∇×E
        dEz_dy = (np.roll(self.Ez, -1, axis=1) - self.Ez) / self.dy
        dEy_dz = (np.roll(self.Ey, -1, axis=2) - self.Ey) / self.dz
        dEx_dz = (np.roll(self.Ex, -1, axis=2) - self.Ex) / self.dz
        dEz_dx = (np.roll(self.Ez, -1, axis=0) - self.Ez) / self.dx
        dEy_dx = (np.roll(self.Ey, -1, axis=0) - self.Ey) / self.dx
        dEx_dy = (np.roll(self.Ex, -1, axis=1) - self.Ex) / self.dy

        curl_x = dEz_dy - dEy_dz
        curl_y = dEx_dz - dEz_dx
        curl_z = dEy_dx - dEx_dy
        return curl_x, curl_y, curl_z

    def curl_H(self):
        # ∇×H
        dHz_dy = (np.roll(self.Hz, -1, axis=1) - self.Hz) / self.dy
        dHy_dz = (np.roll(self.Hy, -1, axis=2) - self.Hy) / self.dz
        dHx_dz = (np.roll(self.Hx, -1, axis=2) - self.Hx) / self.dz
        dHz_dx = (np.roll(self.Hz, -1, axis=0) - self.Hz) / self.dx
        dHy_dx = (np.roll(self.Hy, -1, axis=0) - self.Hy) / self.dx
        dHx_dy = (np.roll(self.Hx, -1, axis=1) - self.Hx) / self.dy

        curl_x = dHz_dy - dHy_dz
        curl_y = dHx_dz - dHz_dx
        curl_z = dHy_dx - dHx_dy
        return curl_x, curl_y, curl_z

    def apply_bc_pec(self):
        # Enforça PEC: componentes tangenciais de E = 0 nas fronteiras
        self.Ex[0,:,:] = 0.0; self.Ex[-1,:,:] = 0.0
        self.Ex[:,0,:] = 0.0; self.Ex[:,-1,:] = 0.0
        self.Ex[:,:,0] = 0.0; self.Ex[:,:,-1] = 0.0

        self.Ey[0,:,:] = 0.0; self.Ey[-1,:,:] = 0.0
        self.Ey[:,0,:] = 0.0; self.Ey[:,-1,:] = 0.0
        self.Ey[:,:,0] = 0.0; self.Ey[:,:,-1] = 0.0

        self.Ez[0,:,:] = 0.0; self.Ez[-1,:,:] = 0.0
        self.Ez[:,0,:] = 0.0; self.Ez[:,-1,:] = 0.0
        self.Ez[:,:,0] = 0.0; self.Ez[:,:,-1] = 0.0

    def apply_bc_mur(self):
        # Mur 1a ordem (simples) para Ez, Ex, Ey nas faces.
        # Fórmula: E(0,y,z,t+dt) = E(1,y,z,t) + (c dt - dx)/(c dt + dx) * (E(1,y,z,t+dt) - E(0,y,z,t))
        c = c0 / np.sqrt(1.0)  # assumindo eps_r=1 nas bordas
        rx = (c * self.dt - self.dx) / (c * self.dt + self.dx)
        ry = (c * self.dt - self.dy) / (c * self.dt + self.dy)
        rz = (c * self.dt - self.dz) / (c * self.dt + self.dz)

        # X-min e X-max
        self.Ex[0,:,:]   = self.prev_Ex[1,:,:] + rx * (self.Ex[1,:,:] - self.prev_Ex[0,:,:])
        self.Ex[-1,:,:]  = self.prev_Ex[-2,:,:] + rx * (self.Ex[-2,:,:] - self.prev_Ex[-1,:,:])
        self.Ey[0,:,:]   = self.prev_Ey[1,:,:] + rx * (self.Ey[1,:,:] - self.prev_Ey[0,:,:])
        self.Ey[-1,:,:]  = self.prev_Ey[-2,:,:] + rx * (self.Ey[-2,:,:] - self.prev_Ey[-1,:,:])
        self.Ez[0,:,:]   = self.prev_Ez[1,:,:] + rx * (self.Ez[1,:,:] - self.prev_Ez[0,:,:])
        self.Ez[-1,:,:]  = self.prev_Ez[-2,:,:] + rx * (self.Ez[-2,:,:] - self.prev_Ez[-1,:,:])

        # Y-min e Y-max
        self.Ex[:,0,:]   = self.prev_Ex[:,1,:] + ry * (self.Ex[:,1,:] - self.prev_Ex[:,0,:])
        self.Ex[:,-1,:]  = self.prev_Ex[:,-2,:] + ry * (self.Ex[:,-2,:] - self.prev_Ex[:,-1,:])
        self.Ey[:,0,:]   = self.prev_Ey[:,1,:] + ry * (self.Ey[:,1,:] - self.prev_Ey[:,0,:])
        self.Ey[:,-1,:]  = self.prev_Ey[:,-2,:] + ry * (self.Ey[:,-2,:] - self.prev_Ey[:,-1,:])
        self.Ez[:,0,:]   = self.prev_Ez[:,1,:] + ry * (self.Ez[:,1,:] - self.prev_Ez[:,0,:])
        self.Ez[:,-1,:]  = self.prev_Ez[:,-2,:] + ry * (self.Ez[:,-2,:] - self.prev_Ez[:,-1,:])

        # Z-min e Z-max
        self.Ex[:,:,0]   = self.prev_Ex[:,:,1] + rz * (self.Ex[:,:,1] - self.prev_Ex[:,:,0])
        self.Ex[:,:,-1]  = self.prev_Ex[:,:,-2] + rz * (self.Ex[:,:,-2] - self.prev_Ex[:,:,-1])
        self.Ey[:,:,0]   = self.prev_Ey[:,:,1] + rz * (self.Ey[:,:,1] - self.prev_Ey[:,:,0])
        self.Ey[:,:,-1]  = self.prev_Ey[:,:,-2] + rz * (self.Ey[:,:,-2] - self.prev_Ey[:,:,-1])
        self.Ez[:,:,0]   = self.prev_Ez[:,:,1] + rz * (self.Ez[:,:,1] - self.prev_Ez[:,:,0])
        self.Ez[:,:,-1]  = self.prev_Ez[:,:,-2] + rz * (self.Ez[:,:,-2] - self.prev_Ez[:,:,-1])

        # Atualiza "prev" para próximo passo
        self.prev_Ex[:] = self.Ex
        self.prev_Ey[:] = self.Ey
        self.prev_Ez[:] = self.Ez

    def step(self, n):
        # Atualiza H (meio passo)
        cex, cey, cez = self.curl_E()
        self.Hx -= (self.dt / self.mu) * cex
        self.Hy -= (self.dt / self.mu) * cey
        self.Hz -= (self.dt / self.mu) * cez

        # Atualiza E (passo inteiro) com condutividade
        chx, chy, chz = self.curl_H()
        self.Ex = self.Ce * self.Ex + self.Ceh * chx
        self.Ey = self.Ce * self.Ey + self.Ceh * chy
        self.Ez = self.Ce * self.Ez + self.Ceh * chz

        # Fonte (soft source)
        amp = gaussian_pulse(n*self.dt, self.t0, self.spread)
        i,j,k = self.src
        if self.source_kind == 'Ez':
            self.Ez[i,j,k] += amp
        elif self.source_kind == 'Ex':
            self.Ex[i,j,k] += amp
        elif self.source_kind == 'Ey':
            self.Ey[i,j,k] += amp

        # Condições de contorno
        if self.bc == 'pec':
            self.apply_bc_pec()
        elif self.bc == 'mur':
            self.apply_bc_mur()

    def run(self, snapshot_plane=('z', None), snapshot_every=10):
        """
        Executa Nt passos. Retorna lista de snapshots (campo escolhido em um plano).
        snapshot_plane: ('x'| 'y' | 'z', índice) — se None, usa centro.
        """
        axis, idx = snapshot_plane
        if idx is None:
            if axis == 'x': idx = self.Nx // 2
            if axis == 'y': idx = self.Ny // 2
            if axis == 'z': idx = self.Nz // 2

        snapshots = []
        for n in range(self.Nt):
            self.step(n)
            if (n % snapshot_every) == 0:
                if axis == 'x':
                    snapshots.append(self.Ez[idx,:,:].copy())
                elif axis == 'y':
                    snapshots.append(self.Ez[:,idx,:].copy())
                else:
                    snapshots.append(self.Ez[:,:,idx].copy())
        return snapshots

def cfl_dt(dx, dy, dz, safety=0.99):
    # Passo de tempo estável (CFL) para FDTD 3D uniforme
    inv_c = np.sqrt( (1/dx**2) + (1/dy**2) + (1/dz**2) )
    return safety / (c0 * inv_c)

def demo():
    # Domínio pequeno para rodar rápido
    Nx, Ny, Nz = 60, 60, 5
    dx = dy = dz = 1e-3  # 1 mm
    dt = cfl_dt(dx, dy, dz, safety=0.7)
    Nt = 200

    solver = FDTD3D(
        Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dy, dz=dz,
        dt=dt, Nt=Nt,
        eps_r=1.0, sigma=0.0,
        bc='mur',                  # ou 'pec'
        source_loc=(Nx//2, Ny//2, Nz//2),
        source_kind='Ez',
        t0=20*dt, spread=6*dt
    )

    snaps = solver.run(snapshot_plane=('z', Nz//2), snapshot_every=10)

    # Plota alguns snapshots de Ez no plano z = meio
    for idx, S in enumerate(snaps[:4]):
        plt.figure()
        plt.imshow(S.T, origin='lower', extent=[0, Nx*dx, 0, Ny*dy], aspect='equal')
        plt.colorbar(label='Ez [V/m]')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title(f'Ez no plano z=meio, snapshot {idx}')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    demo()
'''
with open('/mnt/data/em_fdtd3d.py', 'w', encoding='utf-8') as f:
    f.write(code)

# Run the tiny demo to show it works (kept lightweight)
runpy.run_path('/mnt/data/em_fdtd3d.py', run_name="__main__")

print("Arquivo salvo em /mnt/data/em_fdtd3d.py")

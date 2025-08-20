import numpy as np
import matplotlib.pyplot as plt

def upwind_deriv_x(E, vx, dx):
    if vx >= 0:
        return (E - np.roll(E, 1, axis=0)) / dx
    else:
        return (np.roll(E, -1, axis=0) - E) / dx

def upwind_deriv_y(E, vy, dy):
    if vy >= 0:
        return (E - np.roll(E, 1, axis=1)) / dy
    else:
        return (np.roll(E, -1, axis=1) - E) / dy

def evolve_jet(E0, eps, vx, vy, g, dx, dy, dt, n_steps, snap_steps=None):
    E = E0.copy()
    snapshots = {}

    for n in range(n_steps):
        dE_dx = upwind_deriv_x(E, vx, dx)
        dE_dy = upwind_deriv_y(E, vy, dy)
        loss = -g * eps * E

        E = E - dt * (vx * dE_dx + vy * dE_dy + loss)
        if snap_steps and n in snap_steps:
            snapshots[n] = E.copy()
    
    return snapshots

# ---------- Grid ----------
Nx, Ny = 200, 200
Lx, Ly = 10, 10
dx, dy = Lx/Nx, Ly/Ny
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# ---------- Medium energy density ----------
eps = np.sin(np.pi*X) * np.cos(np.pi*Y) # Should be from PART2

# ---------- Jet initial condition ----------
sigma_x = 0.1
sigma_y = 0.1
E0 = np.exp(-(X**2)/(2*sigma_x**2) -(Y**2/(2*sigma_y**2)))

# ---------- Parameters ----------
vx, vy = 1.0, 1.0
g = 0.5
CFL = 0.3
dt = CFL * min(dx/abs(vx) if vx!=0 else np.inf,
                dy/abs(vy) if vy!=0 else np.inf)
n_steps = 400
snap_steps = [0, n_steps/4, n_steps/2, 3*n_steps/4, n_steps-1]

# ---------- Run ----------
snaps = evolve_jet(E0, eps, vx, vy, g, dx, dy, dt, n_steps, snap_steps)

# ---------- Plots ----------
# 1) Medium
plt.figure(figsize=(6,4))
plt.imshow(eps.T, origin='lower', extent=[0,Lx,0,Ly], aspect='auto')
plt.title("Medium energy density")
plt.xlabel("x"); plt.ylabel("y")
plt.show()

# 2) Jet snapshots
for step, E_snap in snaps.items():
    plt.figure(figsize=(6,4))
    plt.imshow(E_snap.T, origin='lower', extent=[0,Lx,0,Ly], aspect='auto')
    plt.title(f"Jet energy density at step {step}")
    plt.xlabel("x"); plt.ylabel("y")
    plt.show()

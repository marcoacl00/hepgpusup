import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
import cmath

sigma = 2.099518748076819
eps = 4.642973059984742
hbar = 1
mu = 1

rmin = 2.0
rmax = 2.7
N = 128
r = np.linspace(rmin, rmax, N)
dr = r[1] - r[0]

dt = 0.01
tmax = 5
timesteps = int(tmax / dt)

# Gaussian wave packet parameters
r0 = sigma * 2**(1/6)
sigma_r = 0.05


def V(r):
    return 4 * eps * ((sigma / r)**12 - (sigma / r)**6)


def hamiltonian():
    T = np.zeros((N, N))
    np.fill_diagonal(T, -2)
    np.fill_diagonal(T[1:], 1)
    np.fill_diagonal(T[:, 1:], 1)
    T *= -hbar**2 / (2 * mu * dr**2)

    V_diag = np.diag(V(r))
    return T + V_diag


def gaussian_pulse(r):
    A = (1 / (2 * np.pi * sigma_r**2))**0.25
    psi = A * np.exp(-((r - r0)**2) / (4 * sigma_r**2))
    return psi.astype(complex)


def normalize(psi):
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dr)
    return psi / norm


def chebyshev_propagate_step(H_norm, psi, a, M):
    psi_next = jv(0, a) * psi
    T_prev = psi.copy()
    T_curr = -1j * H_norm @ T_prev
    psi_next += jv(1, a) * T_curr

    for k in range(2, M + 1):
        T_next = -1j * 2 * H_norm @ T_curr + T_prev
        psi_next += jv(k, a) * T_next
        T_prev, T_curr = T_curr, T_next

    return psi_next


H = hamiltonian()
eigvals = np.linalg.eigvalsh(H)
Emin, Emax = np.min(eigvals), np.max(eigvals)
dE = Emax - Emin
H_norm = 2 * (H - (Emin + dE / 2) * np.identity(N)) / dE

# Initial state
psi = normalize(gaussian_pulse(r))
expect_vals = []

for i in range(timesteps):
    t = i * dt
    a = dE * t * dt / (2 * hbar)
    psi_new = chebyshev_propagate_step(H_norm, psi, a, M=30)
    phase = cmath.exp(-1j * (Emin * t * dt / hbar + a))
    psi = normalize(phase * psi_new)

    exp_r = np.sum(r * np.abs(psi) ** 2) * dr
    expect_vals.append(exp_r)

    if i % 100 == 0:
        print(f"Progress: {i * dt:.2f} / {tmax}")

time_array = np.linspace(0, tmax, timesteps)
plt.plot(time_array, expect_vals)
plt.axhline(2.356, color='gray', linestyle='--', label='$r_{eq}$')
plt.xlabel('Time')
plt.ylabel(r'$\langle r \rangle(t)$')
plt.title('Oscillation of ⟨r⟩(t) in Lennard-Jones Potential')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

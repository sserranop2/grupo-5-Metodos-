import numpy as np
import matplotlib
matplotlib.use("Agg")  # sin ventanas
import matplotlib.pyplot as plt
from numba import njit


#1.a Alcanzar el equilibrio 

# Parámetros de simulación 
N      = 250      # lado de la red (N x N)
J      = 1.0        # acoplamiento
BETA   = 0.5        # β = 1/(kT)
STEPS  = 250000    
FIGOUT = "1.a.pdf" 

#  Utilidades del modelo 
@njit
def random_spins(N):
    s = np.empty((N, N), dtype=np.int8)
    for i in range(N):
        for j in range(N):
            s[i, j] = 1 if np.random.random() < 0.5 else -1
    return s

@njit
def neighbor_sum(spins, i, j):
    N = spins.shape[0]
    return (spins[(i+1) % N, j] + spins[(i-1) % N, j] +
            spins[i, (j+1) % N] + spins[i, (j-1) % N])

@njit
def energia_total(spins, J):
    N = spins.shape[0]
    e = 0.0
    for i in range(N):
        for j in range(N):
            e += -J * spins[i, j] * neighbor_sum(spins, i, j)
    return e

@njit
def metropolis_ising(spins, beta, steps, J):
    N   = spins.shape[0]
    eps = energia_total(spins, J)     # ε0 (doble conteo)
    M   = np.int64(spins.sum())           # M0

    E_series = np.empty(steps + 1, dtype=np.float64)
    M_series = np.empty(steps + 1, dtype=np.float64)
    normE = 4.0 * N * N
    normM = 1.0 * N * N

    E_series[0] = eps / normE
    M_series[0] = M   / normM

    for t in range(1, steps + 1):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        s = spins[i, j]
        S = neighbor_sum(spins, i, j)

        dH = 2.0 * J * s * S
        if dH <= 0.0 or np.random.random() < np.exp(-beta * dH):
            spins[i, j] = -s
            deps = 4.0 * J * s * S    # Δε (doble conteo)
            eps += deps
            M   += (-2 * s)           # ΔM

        E_series[t] = eps / normE
        M_series[t] = M   / normM

    return E_series, M_series

# Plot 
def plot_1a(spins_ini, E, M, spins_fin, fname=FIGOUT):
    import matplotlib.gridspec as gridspec
    steps = np.arange(len(E))

    # Usamos constrained_layout en vez de tight_layout
    fig = plt.figure(figsize=(12, 4), dpi=144, constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.8, 1.1], figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(spins_ini, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    ax0.set_title("Antes")
    ax0.set_xticks([]); ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(steps, E, lw=2.2, color="black", label="Energía")
    ax1.plot(steps, M, lw=1.6, color="#d62728", alpha=0.9, label="Magnetización")
    ax1.set_xlabel("Épocas")
    ax1.set_ylim(min(E.min(), M.min()) - 0.02, max(E.max(), M.max()) + 0.02)
    ax1.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax1.legend(loc="best", frameon=False)
    ax1.set_title("β=1/2, J=1")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(spins_fin, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    ax2.set_title("Después")
    ax2.set_xticks([]); ax2.set_yticks([])

    fig.suptitle(f"Ising 2D — N={N}", y=1.02, fontsize=11)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)

def simluacion_1a():
    np.random.seed()
    spins0 = random_spins(N)
    spins  = spins0.copy()

    E_series, M_series = metropolis_ising(spins, BETA, STEPS, J)

    plot_1a(spins0, E_series, M_series, spins, FIGOUT)
    # (Eliminado) np.save("1.a_final_state.npy", spins)

simluacion_1a()


#1.b Cambio de fase 
BETA_GRID_FINE   = np.arange(0.36, 0.5201, 0.005)
BETA_GRID_COARSE = np.concatenate([np.arange(0.10, 0.36, 0.02),
                                   np.arange(0.52, 0.91, 0.02)])
BETA_GRID        = np.unique(np.round(np.concatenate([BETA_GRID_COARSE, BETA_GRID_FINE]), 5))

# Barridos para equilibrio/medición (puedes afinarlos; ya van más rápido)
EQ_SWEEPS_FIRST  = 250
EQ_SWEEPS_NEXT   = 100
MEAS_SAMPLES     = 2000
STRIDE_SWEEPS    = 6

FIGOUT_1B        = "1.b.pdf"

@njit
def build_pbc_index(N):
    """Tablas de vecinos con PBC para evitar %N en el bucle."""
    ip = np.empty(N, dtype=np.int32)
    im = np.empty(N, dtype=np.int32)
    for i in range(N):
        ip[i] = i + 1 if i + 1 < N else 0
        im[i] = i - 1 if i - 1 >= 0 else N - 1
    return ip, im

@njit
def metropolis_sweeps_fast(spins, beta, J, nsweeps, eps, M, ip, im):
    """
    nsweeps barridos Metrópolis con:
      - LUT de aceptación (solo exp4 y exp8)
      - PBC via tablas ip/im (sin módulo en el bucle)
    Actualiza eps (doble conteo) y M.
    """
    N = spins.shape[0]
    exp4 = np.exp(-beta * 4.0 * J)
    exp8 = np.exp(-beta * 8.0 * J)

    for _ in range(nsweeps):
        # visitar N^2 sitios (orden aleatorio por elección aleatoria de (i,j))
        for _ in range(N * N):
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            s = spins[i, j]
            # suma vecinal sin %
            S = spins[ip[i], j] + spins[im[i], j] + spins[i, ip[j]] + spins[i, im[j]]
            dH = 2.0 * J * s * S

            # aceptación con LUT
            accept = False
            if dH <= 0.0:
                accept = True
            elif dH == 4.0 * J:
                accept = (np.random.random() < exp4)
            elif dH == 8.0 * J:
                accept = (np.random.random() < exp8)
            else:
                # (para robustez; no debería ocurrir)
                accept = (np.random.random() < np.exp(-beta * dH))

            if accept:
                spins[i, j] = -s
                eps += 4.0 * J * s * S   # Δε (doble conteo)
                M   += (-2 * s)          # ΔM

    return eps, M

@njit
def mc_energy_sweeps(spins, beta, J, eq_sweeps, samples, stride_sweeps, ip, im):
    """
    Equilibra eq_sweeps barridos y luego toma 'samples' mediciones de energía
    separadas por 'stride_sweeps' barridos. Devuelve la serie de energías normalizadas.
    """
    N   = spins.shape[0]
    eps = energia_total(spins, J)  # ε (doble conteo) consistente con 1.a
    M   = np.int64(spins.sum())
    # burn-in
    eps, M = metropolis_sweeps_fast(spins, beta, J, eq_sweeps, eps, M, ip, im)
    # medición
    Es = np.empty(samples, dtype=np.float64)
    normE = 4.0 * N * N
    for k in range(samples):
        eps, M = metropolis_sweeps_fast(spins, beta, J, stride_sweeps, eps, M, ip, im)
        Es[k] = eps / normE
    return Es

def simulacion_1b():
    np.random.seed()  # independiente de 1.a
    spins = random_spins(N)   # estado inicial aleatorio (~β=0)
    ip, im = build_pbc_index(N)
    Cv = np.empty(BETA_GRID.shape[0], dtype=np.float64)

    beta_c = 0.5 * np.log(1.0 + np.sqrt(2.0))
    first = True
    for idx, beta in enumerate(BETA_GRID):
        # burn-in extra cerca del crítico por 'critical slowing down'
        eq_sweeps = EQ_SWEEPS_FIRST if first else EQ_SWEEPS_NEXT
        if abs(beta - beta_c) < 0.03:
            eq_sweeps = max(eq_sweeps, EQ_SWEEPS_NEXT * 3)

        Es = mc_energy_sweeps(spins, beta, J, eq_sweeps,
                                         MEAS_SAMPLES, STRIDE_SWEEPS, ip, im)
        mu  = float(np.mean(Es))
        mu2 = float(np.mean(Es * Es))
        Cv[idx] = (beta**2) * (N * N) * (mu2 - mu * mu)
        first = False  # warm-start

    # gráfico
    fig = plt.figure(figsize=(8, 6), dpi=144, constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    idx = np.argsort(BETA_GRID)
    ax.plot(BETA_GRID[idx], Cv[idx], color="black", lw=1.8)
    ax.axvline(beta_c, color="red", alpha=0.6)
    ax.text(beta_c + 0.005, 0.97 * Cv.max(), "Critical point (theory)", color="red")
    ax.set_xlabel("Thermodynamic β")
    ax.set_ylabel("Specific heat from simulation")
    fig.savefig(FIGOUT_1B, bbox_inches="tight")
    plt.close(fig)

# Llamar 1.b tras 1.a
simulacion_1b()
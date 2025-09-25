import math
import numpy as np
import sys
import scipy
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
from dataclasses import dataclass
from typing import Tuple, Dict, List

# Parámetros del problema (tiempo: días; tasas: 1/día) ----------
A_por_dia = 1000.0   # creación de U por día (fuente)
B_por_dia = 20.0     # extracción de Pu por día (proporcional a Pu)

# Vidas medias y lambdas
T12_U_minutos = 23.4
T12_U_dias = T12_U_minutos / 60.0 / 24.0
T12_Np_dias = 2.36

LOG2 = math.log(2.0)
lambda_U  = LOG2 / T12_U_dias     
lambda_Np = LOG2 / T12_Np_dias    

# Condiciones iniciales y tiempo final
U_inicial  = 10.0
Np_inicial = 10.0
Pu_inicial = 10.0
tiempo_final_dias = 30.0

#Sistema determinista 
def derivadas(t, y):
    """Derivadas del sistema determinista: y = [U, Np, Pu]."""
    U, Np, Pu = y
    dU  = A_por_dia - lambda_U * U
    dNp = lambda_U * U - lambda_Np * Np
    dPu = lambda_Np * Np - B_por_dia * Pu
    return np.array([dU, dNp, dPu], dtype=float)

def estado_estable():
    """Estado estacionario analítico (f=0)."""
    U_estrella  = A_por_dia / lambda_U
    Np_estrella = A_por_dia / lambda_Np
    Pu_estrella = A_por_dia / B_por_dia
    return np.array([U_estrella, Np_estrella, Pu_estrella], dtype=float)

# Malla temporal con alta resolución inicial
def malla_tiempo(t_fin):
    """
    Muchos puntos cerca de 0 para capturar el transitorio rápido de U,
    luego una malla lineal hasta t_fin. Incluye 0 exactamente.
    """
    t0 = 1e-6  # d (~0.086 s) para evitar 0 en geomspace
    # Ventana "temprana" hasta 0.2 días (~4.8 h)
    t_early = np.geomspace(t0, min(0.2, t_fin), 400)
    t_early[0] = 0.0
    # Ventana "tardía" lineal
    t_late_start = t_early[-1] if t_fin > t_early[-1] else t_fin
    t_late = np.linspace(t_late_start, t_fin, 1200)
    t = np.unique(np.concatenate((t_early, t_late)))
    return t

#  ntegración con SciPy o con RK4 de respaldo ----------
def integrar(y0, t_fin):
    t_eval = malla_tiempo(t_fin)
    sol = solve_ivp(
        fun=derivadas,
        t_span=(0.0, t_fin),
        y0=y0,
        method="Radau",         
        atol=1e-9,
        rtol=1e-7,
        first_step=1e-4,        
        max_step=0.02,          
        t_eval=t_eval
    )
    return sol.t, sol.y.T


# ---------- Detección de cuasi-estacionariedad por variable ----------
def tiempos_cuasi_estacionario_por_variable(t, Y, atol=1e-4, rtol=1e-6):
    """
    Para cada X ∈ {U, Np, Pu} detecta el primer t donde |Xdot|/(1+|X|) ≤ atol+rtol.
    Devuelve (t_U, t_Np, t_Pu), NaN si no se cumple en el intervalo.
    """
    D = np.array([derivadas(tk, Yk) for tk, Yk in zip(t, Y)], dtype=float)  # (n,3)
    eps = atol + rtol
    t_star = []
    for i in range(3):
        Xi = Y[:, i]
        dXi = D[:, i]
        ratio = np.abs(dXi) / (1.0 + np.abs(Xi))
        idx = np.where(ratio <= eps)[0]
        if idx.size == 0:
            t_star.append(float('nan'))
        else:
            j = idx[0]
            if j == 0:
                t_star.append(t[0])
            else:
                # Interpolación lineal del cruce
                x0, x1 = ratio[j-1], ratio[j]
                if np.isfinite(x0) and np.isfinite(x1) and (x1 != x0):
                    f = (eps - x0) / (x1 - x0)
                    t_interp = t[j-1] + f * (t[j] - t[j-1])
                    t_star.append(float(t_interp))
                else:
                    t_star.append(float(t[j]))
    return tuple(t_star)

# Graficas
def _auto_limites(ax, y):
    y = np.asarray(y, dtype=float)
    y_min, y_max = float(np.min(y)), float(np.max(y))
    pad = 1e-12 if y_max == y_min else 0.05 * (y_max - y_min)
    ax.set_ylim(y_min - pad, y_max + pad)

def guardar_pdf_multipagina(t, Y, t_estrellas, y_estrella, archivo_pdf="2.a.pdf"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    Y_plot = np.maximum(Y, 0.0)

    nombres = ["U-239", "Np-239", "Pu-239"]
    series = [Y_plot[:,0], Y_plot[:,1], Y_plot[:,2]]
    ystars = [y_estrella[0], y_estrella[1], y_estrella[2]]

    with PdfPages(archivo_pdf) as pdf:
        for i in range(3):
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            ax.plot(t, series[i], lw=2)
            ax.axhline(ystars[i], ls="--", lw=1.1)
            ti = t_estrellas[i]
            if not math.isnan(ti):
                ax.axvline(ti, color="k", ls=":", lw=1.0)
            ax.set_title(f"2.a — {nombres[i]} (sistema determinista)")
            ax.set_xlabel("Tiempo [días]")
            ax.set_ylabel(f"{nombres[i]} [unid]")
            _auto_limites(ax, series[i])
            ax.grid(True, alpha=0.3)

            if i == 0:
                # Zoom temprano, elevado un poco para no chocar con el eje
                tau_U = 1.0 / lambda_U
                x1, x2 = 0.0, min(0.15, 5.0 * tau_U)
                mask = (t >= x1) & (t <= x2)
                if np.any(mask):
                    y_zoom = series[i][mask]
                    t_zoom = t[mask]
                    axins = ax.inset_axes([0.52, 0.18, 0.44, 0.44])  # (x,y,w,h) en fracción
                    axins.plot(t_zoom, y_zoom, lw=1.8)
                    axins.axhline(ystars[i], ls="--", lw=1.0)
                    axins.set_xlim(x1, x2)
                    y_min, y_max = float(np.min(y_zoom)), float(np.max(y_zoom))
                    pad = 0.05 * (y_max - y_min if y_max != y_min else 1.0)
                    axins.set_ylim(y_min - pad, y_max + pad)
                    axins.grid(True, alpha=0.3)
                    axins.set_title("Zoom (t<1)", fontsize=9)
                    axins.tick_params(labelsize=8)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

#Cargar codigo
def simulacion_2a():
    y0 = np.array([U_inicial, Np_inicial, Pu_inicial], dtype=float)

    t, Y = integrar(y0, tiempo_final_dias)

    y_star = estado_estable()
    tU, tNp, tPu = tiempos_cuasi_estacionario_por_variable(t, Y, atol=1e-4, rtol=1e-6)
    guardar_pdf_multipagina(t, Y, (tU, tNp, tPu), y_star, archivo_pdf="2.a.pdf")

    # Chequeos de coherencia teórica (errores relativos al final)
    err_rel = np.abs((Y[-1] - y_star) / (1.0 + np.abs(y_star)))
    print("Equilibrio analítico: U*={:.6f}, Np*={:.6f}, Pu*={:.6f}".format(*y_star))
    print("Tiempos cuasi-estacionarios: tU*={:.4f} d, tNp*={:.4f} d, tPu*={:.4f} d".format(tU, tNp, tPu))
    print("Error relativo al final (U, Np, Pu):", err_rel)
    print("Gráfica multipágina guardada en 2.a.pdf")

simulacion_2a()


#2.b Ecuación diferencial estocástica
def mu(y):
    """Drift vector μ(y) = f(y)."""
    U, Np, Pu = y
    return np.array([
        A_por_dia - lambda_U * U,
        lambda_U * U - lambda_Np * Np,
        lambda_Np * Np - B_por_dia * Pu
    ], dtype=float)

def sigma(y):
    """
    Volatilidad σ(y) por variable (Chemical Langevin simplificado):
    raíz de la suma de tasas que modifican cada especie en ±1.
    """
    U, Np, Pu = y
    sU  = max(A_por_dia + lambda_U * U, 0.0)
    sNp = max(lambda_U * U + lambda_Np * Np, 0.0)
    sPu = max(lambda_Np * Np + B_por_dia * Pu, 0.0)
    return np.array([math.sqrt(sU), math.sqrt(sNp), math.sqrt(sPu)], dtype=float)

def sde_rk2_trayectorias(y0, t0, t_fin, dt, n_tray):
    """
    RK2 estocástico (Platen) tal como en el enunciado:
      K1 = dt μ(y) + (W - S) sqrt(dt) σ(y)
      K2 = dt μ(y + K1) + (W + S) sqrt(dt) σ(y + K1)
      y_{n+1} = y_n + 0.5 (K1 + K2)
    * W ~ N(0,1) y S ∈ {-1,1} son ESCALARES comunes al vector en cada paso.
    * Mantiene no-negatividad con una barrera en 0.
    """
    n = int(math.ceil((t_fin - t0) / dt)) + 1
    t = np.linspace(t0, t_fin, n)
    Y = np.zeros((n_tray, n, 3), dtype=float)
    sqrt_dt = math.sqrt(dt)

    for m in range(n_tray):
        y = np.array(y0, dtype=float)
        Y[m, 0] = y
        for k in range(1, n):
            mu1 = mu(y)
            sig1 = sigma(y)
            # Un W y un S por paso (compartidos por todas las variables):
            W = np.random.randn()
            S = -1.0 if np.random.random() < 0.5 else 1.0

            K1 = dt * mu1 + (W - S) * (sqrt_dt * sig1)
            y1 = y + K1

            mu2 = mu(y1)
            sig2 = sigma(y1)
            K2 = dt * mu2 + (W + S) * (sqrt_dt * sig2)

            y = y + 0.5 * (K1 + K2)
            # no-negatividad (evita valores físicamente inválidos)
            y = np.maximum(y, 0.0)
            # saneo muy pequeño por redondeo:
            y[np.abs(y) < 1e-15] = 0.0
            Y[m, k] = y

    return t, Y

def guardar_pdf_2b(t, Y_det, trayect, archivo_pdf="2.b.pdf"):
    nombres = ["U-239", "Np-239", "Pu-239"]

    with PdfPages(archivo_pdf) as pdf:
        for i, nombre in enumerate(nombres):
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            # 5 trayectorias
            for m in range(trayect.shape[0]):
                ax.plot(t, trayect[m, :, i], lw=1.2, alpha=0.85, zorder=1)
            # determinista encima
            ax.plot(t, Y_det[:, i], lw=2.4, zorder=3)
            ax.set_title(f"2.b — {nombre}: 5 trayectorias SDE (RK2) + determinista")
            ax.set_xlabel("Tiempo [días]")
            ax.set_ylabel(f"{nombre} [unid]")
            _auto_limites(ax, np.vstack([trayect[:, :, i], Y_det[:, i]]))
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

  

def simulacion_2b():
    y0 = np.array([U_inicial, Np_inicial, Pu_inicial], dtype=float)
    t_det, Y_det = integrar(y0, tiempo_final_dias)

    # Paso de integración SDE (respetando la escala 1/lambda_U ~ 0.0234 d)
    dt_sde = 1e-3  # 1.44 min
    n_tray = 5
    t_sde, trayect = sde_rk2_trayectorias(y0, 0.0, tiempo_final_dias, dt_sde, n_tray)

    # Interpolación si las mallas difieren
    if not np.array_equal(t_sde, t_det):
        Y_det_interp = np.column_stack([
            np.interp(t_sde, t_det, Y_det[:, 0]),
            np.interp(t_sde, t_det, Y_det[:, 1]),
            np.interp(t_sde, t_det, Y_det[:, 2]),
        ])
    else:
        Y_det_interp = Y_det

    guardar_pdf_2b(t_sde, Y_det_interp, trayect, archivo_pdf="2.b.pdf")
    print("2.b → 2.b.pdf generado con 5 trayectorias SDE + solución determinista.")

simulacion_2b()





#2.c Simulación exacta
def gillespie_un_path(tmax, y0, rng):
    """
    Reacciones (tasas en 1/día):
      1) +U                 r1 = A_por_dia
      2) U -> Np            r2 = lambda_U  * U
      3) Np -> Pu           r3 = lambda_Np * Np
      4) extracción de Pu   r4 = B_por_dia * Pu   
    """
    t = 0.0
    y = np.array(y0, dtype=int)
    ts = [0.0]
    Ys = [y.copy()]

    while t < tmax:
        U, Np, Pu = y
        r1 = A_por_dia
        r2 = lambda_U  * max(U,  0)
        r3 = lambda_Np * max(Np, 0)
        r4 = B_por_dia * max(Pu, 0)

        rates = np.array([r1, r2, r3, r4], dtype=float)
        s = rates.sum()
        if s <= 0.0:
            break

        # Tiempo al siguiente evento
        tau = rng.exponential(1.0 / s)
        t_next = t + tau
        if t_next > tmax:
            ts.append(tmax)
            Ys.append(y.copy())
            break

        # Elegir reacción
        r_idx = rng.choice(4, p=rates / s)

        # Aplicar reacción
        if r_idx == 0:          # creación U
            y[0] += 1
        elif r_idx == 1:        # U -> Np
            if y[0] > 0:
                y[0] -= 1; y[1] += 1
        elif r_idx == 2:        # Np -> Pu
            if y[1] > 0:
                y[1] -= 1; y[2] += 1
        elif r_idx == 3:        # extracción Pu
            if y[2] > 0:
                y[2] -= 1

        t = t_next
        ts.append(t)
        Ys.append(y.copy())

    return np.array(ts), np.array(Ys, dtype=float)


def guardar_pdf_2c_multipagina(t_det, Y_det, paths, archivo_pdf="2.c.pdf"):
    """
    Crea un PDF multipágina:
      - Página 1: U-239 (5 trayectorias + determinista)
      - Página 2: Np-239
      - Página 3: Pu-239
    """
    nombres = ["U-239", "Np-239", "Pu-239"]

    with PdfPages(archivo_pdf) as pdf:
        for i, nombre in enumerate(nombres):
            fig, ax = plt.subplots(figsize=(8.5, 5.5))

            # 5 trayectorias Gillespie
            for ts, Ys in paths:
                ax.step(ts, Ys[:, i], where='post', alpha=0.6, lw=1.0)

            # Solución determinista encima
            ax.plot(t_det, Y_det[:, i], lw=2.6, alpha=0.95, label="Determinista")

            ax.set_title(f"2.c — {nombre}: 5 trayectorias (Gillespie) + determinista")
            ax.set_xlabel("Tiempo [días]")
            ax.set_ylabel(f"{nombre} [unid]")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def simulacion_2c_multipagina(n_paths=5, archivo_pdf="2.c.pdf", seed=12345):
    rng = np.random.default_rng(seed)

    # Solución determinista
    t_det, Y_det = integrar([U_inicial, Np_inicial, Pu_inicial], tiempo_final_dias)

    # 5 trayectorias de Gillespie
    paths = [gillespie_un_path(tiempo_final_dias,
                               [U_inicial, Np_inicial, Pu_inicial],
                               rng)
             for _ in range(n_paths)]

    # Guardar PDF multipágina
    guardar_pdf_2c_multipagina(t_det, Y_det, paths, archivo_pdf=archivo_pdf)
    print(f"2.c (multipágina) → {archivo_pdf} generado.")


# Ejecutar 2.c
simulacion_2c_multipagina()



#2.d Probabilidad de concentración crítica

# Parámetros del problema
T_MAX_DAYS = 30.0
PU_THRESHOLD = 80.0

U0 = 10.0
NP0 = 10.0
PU0 = 10.0

A = 1000.0
B = 20.0

t12_U_min = 23.4
t12_U_days = t12_U_min / (60.0 * 24.0)
t12_NP_days = 2.36

lam_U = math.log(2.0) / t12_U_days
lam_NP = math.log(2.0) / t12_NP_days



# ========================
def freq_estimate(k: int, N: int) -> Tuple[float, float, Tuple[float, float]]:
    if N <= 0:
        return 0.0, 0.0, (0.0, 0.0)
    p = k / N
    se = math.sqrt(max(p * (1.0 - p) / N, 0.0))
    z = 1.96
    lo = max(0.0, p - z * se)
    hi = min(1.0, p + z * se)
    return p, se, (lo, hi)

def bayes_beta_ci(k: int, N: int, alpha_prior: float = 1.0, beta_prior: float = 1.0, q_lo=0.025, q_hi=0.975):
    a = alpha_prior + k
    b = beta_prior + (N - k)
    rng = np.random.default_rng(12345)
    samples = rng.beta(a, b, size=200000)
    mean = a / (a + b)
    lo = np.quantile(samples, q_lo)
    hi = np.quantile(samples, q_hi)
    return mean, (float(lo), float(hi))



# =========================
def ode_rhs(u, np_, pu):
    dU = A - lam_U * u
    dNp = B + lam_U * u - lam_NP * np_
    dPu = lam_NP * np_
    return dU, dNp, dPu

def rk4_deterministic(dt=1e-3):
    t = 0.0
    u, np_, pu = U0, NP0, PU0
    crossed = (pu >= PU_THRESHOLD)
    while t < T_MAX_DAYS and not crossed:
        k1 = ode_rhs(u, np_, pu)
        k2 = ode_rhs(u + 0.5*dt*k1[0], np_ + 0.5*dt*k1[1], pu + 0.5*dt*k1[2])
        k3 = ode_rhs(u + 0.5*dt*k2[0], np_ + 0.5*dt*k2[1], pu + 0.5*dt*k2[2])
        k4 = ode_rhs(u + dt*k3[0], np_ + dt*k3[1], pu + dt*k3[2])
        u  += (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        np_+= (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        pu += (dt/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        u = max(u, 0.0); np_ = max(np_, 0.0); pu = max(pu, 0.0)
        t += dt
        if pu >= PU_THRESHOLD:
            crossed = True
    return crossed


# ======================
def sde_step(u, np_, pu, dt, rng: np.random.Generator):
    a1 = A
    a2 = lam_U * max(u, 0.0)
    a3 = B
    a4 = lam_NP * max(np_, 0.0)

    du_dt  = +a1 - a2
    dnp_dt = +a2 + a3 - a4
    dpu_dt = +a4

    dW = rng.normal(0.0, math.sqrt(dt), size=4)
    du_sto  = (+1.0*math.sqrt(a1)*dW[0]) + (-1.0*math.sqrt(a2)*dW[1])
    dnp_sto = (+1.0*math.sqrt(a2)*dW[1]) + (+1.0*math.sqrt(a3)*dW[2]) + (-1.0*math.sqrt(a4)*dW[3])
    dpu_sto = (+1.0*math.sqrt(a4)*dW[3])

    u_new  = u  + du_dt*dt  + du_sto
    np_new = np_ + dnp_dt*dt + dnp_sto
    pu_new = pu + dpu_dt*dt + dpu_sto

    if u_new < 0: u_new = 0.0
    if np_new < 0: np_new = 0.0
    if pu_new < 0: pu_new = 0.0

    return u_new, np_new, pu_new

def run_sde(N: int = 1000, dt: float = 1e-3, seed: int = 123):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(N):
        t = 0.0
        u, np_, pu = U0, NP0, PU0
        crossed = (pu >= PU_THRESHOLD)
        while t < T_MAX_DAYS and not crossed:
            u, np_, pu = sde_step(u, np_, pu, dt, rng)
            t += dt
            if pu >= PU_THRESHOLD:
                crossed = True
        hits += int(crossed)
    return hits



# ================
def gillespie_once(rng: np.random.Generator) -> bool:
    t = 0.0
    u, np_, pu = U0, NP0, PU0
    if pu >= PU_THRESHOLD:
        return True

    while t < T_MAX_DAYS:
        a1 = A
        a2 = lam_U * u
        a3 = B
        a4 = lam_NP * np_

        a0 = a1 + a2 + a3 + a4
        if a0 <= 0.0:
            break

        tau = rng.exponential(1.0 / a0)
        t += tau
        if t > T_MAX_DAYS:
            break

        r = rng.uniform(0.0, a0)
        if r < a1:
            u += 1.0
        elif r < a1 + a2:
            if u >= 1.0:
                u -= 1.0
                np_ += 1.0
        elif r < a1 + a2 + a3:
            np_ += 1.0
        else:
            if np_ >= 1.0:
                np_ -= 1.0
                pu += 1.0

        if pu >= PU_THRESHOLD:
            return True

    return False

def run_gillespie(N: int = 1000, seed: int = 321) -> int:
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(N):
        if gillespie_once(rng):
            hits += 1
    return hits

# ========================
def main_point_2d(N: int = 1000, dt_det: float = 1e-3, dt_sde: float = 1e-3) -> Dict[str, Dict]:
    det_cross = rk4_deterministic(dt_det)
    k_det = int(det_cross)
    N_det = 1

    k_sde = run_sde(N=N, dt=dt_sde, seed=20240925)
    N_sde = N

    k_gill = run_gillespie(N=N, seed=20240925)
    N_gill = N

    results = {}
    for label, k, NN in [
        ("Determinista (2.a)", k_det, N_det),
        ("SDE (2.b)", k_sde, N_sde),
        ("Gillespie (2.c)", k_gill, N_gill),
    ]:
        p_hat, se, (lo, hi) = freq_estimate(k, NN)
        mean_b, (blo, bhi) = bayes_beta_ci(k, NN, 1.0, 1.0)
        results[label] = {
            "k": k,
            "N": NN,
            "p_hat": p_hat,
            "se": se,
            "ci95_norm": (lo, hi),
            "bayes_mean": mean_b,
            "bayes_ci95": (blo, bhi)
        }
    return results

# Ejecutar y guardar archivo
# =====================================
if __name__ == "__main__":
    results = main_point_2d(N=1000, dt_det=1e-3, dt_sde=1e-3)

    lines: List[str] = []
    lines.append("Punto 2.d — Probabilidad de alcanzar Pu >= 80 en [0, 30 días]\n")
    lines.append(f"Parámetros: A={A:.6g} día^-1, B={B:.6g} día^-1, λ_U={lam_U:.6g} día^-1, λ_Np={lam_NP:.6g} día^-1; "
                 f"U0=Np0=Pu0=10; Umbral=80; Horiz=30 días.\n")
    for label, r in results.items():
        p = 100.0 * r["p_hat"]
        se = 100.0 * r["se"]
        lo_n = 100.0 * r["ci95_norm"][0]
        hi_n = 100.0 * r["ci95_norm"][1]
        pb = 100.0 * r["bayes_mean"]
        lo_b = 100.0 * r["bayes_ci95"][0]
        hi_b = 100.0 * r["bayes_ci95"][1]
        if r["N"] == 1:
            lines.append(f"- {label}: trayectoria única → {'CRUZA' if r['k']==1 else 'NO CRUZA'} (p≈{p:.3f}%).\n")
        else:
            lines.append(
                f"- {label}: k={r['k']}, N={r['N']} → p̂={p:.3f}% (SE={se:.3f}%), "
                f"IC95≈[{lo_n:.3f}%, {hi_n:.3f}%]; "
                f"Bayes (Beta[1+k,1+N−k]): media={pb:.3f}%, IC95≈[{lo_b:.3f}%, {hi_b:.3f}%].\n"
            )

    lines.append("\nDiscusión breve: Las tres aproximaciones concuerdan cualitativamente en la dinámica ∅→U→Np→Pu con fuentes A y B; ")
    lines.append("la SSA (Gillespie) sirve como ‘ground truth’ discreto, mientras que la SDE aproxima bien para poblaciones moderadas. ")
    lines.append("El modelo determinista no entrega probabilidad (0%/100% según cruce), por lo que la incertidumbre se reporta solo para los enfoques estocásticos.\n")

    with open("2.d.txt", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("\n".join(lines[:12]))
    print("...")
    print("Archivo guardado en 2.d.txt")
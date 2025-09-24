import math
import numpy as np
import sys
import scipy
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

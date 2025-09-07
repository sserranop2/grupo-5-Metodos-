#Librerias
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import math
from matplotlib.ticker import ScalarFormatter

# 1. Cantidades conservadas
def zoom(ax, y):
    """Auto-zoom al rango real de la serie para ver fluctuaciones pequeñas."""
    y = np.asarray(y)
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return
    if y_max == y_min:
        pad = 1e-12
    else:
        pad = 0.05 * (y_max - y_min)  # 5% de margen
    ax.set_ylim(y_min - pad, y_max + pad)

#1.a Sistema depredador-presa
@dataclass
class LVParams:
    alpha: float = 2.0
    beta:  float = 1.5
    gamma: float = 0.3
    delta: float = 0.4
    x0: float = 3.0   # decazorros
    y0: float = 2.0   # kiloconejos
    t_max: float = 50.0

def lv_rhs(t, z, p: LVParams):
    x, y = z
    return [p.alpha*x - p.beta*x*y,
            -p.gamma*y + p.delta*x*y]

def lv_invariant(x, y, p: LVParams):
    # V = δx − γ ln x + βy − α ln y  (protege log(0))
    x_ = np.clip(x, 1e-12, None)
    y_ = np.clip(y, 1e-12, None)
    return p.delta*x_ - p.gamma*np.log(x_) + p.beta*y_ - p.alpha*np.log(y_)

def run_1a(save_as="1.a.pdf"):
    p = LVParams()
    t_span = (0.0, p.t_max)
    t_eval = np.linspace(*t_span, 4000)

    sol = solve_ivp(
        fun=lv_rhs, t_span=t_span, y0=[p.x0, p.y0], t_eval=t_eval,
        args=(p,), method="RK45", rtol=1e-9, atol=1e-12
    )
    t = sol.t
    x, y = sol.y
    V = lv_invariant(x, y, p)

    # --- 2 subplots: (a) x(t)/y(t) con ejes y separados, (b) V(t) ---
    fig, (ax_pop, ax_V) = plt.subplots(2, 1, figsize=(8, 8), sharex=False)

    # Eje izquierdo: x(t) [decazorros]
    l1, = ax_pop.plot(t, x, color="tab:blue", label="x(t) – presas")
    ax_pop.set_xlabel("t")
    ax_pop.set_ylabel("x(t) [decazorros]", color=l1.get_color())
    ax_pop.tick_params(axis="y", labelcolor=l1.get_color())
    ax_pop.grid(True, alpha=0.3)

    # Eje derecho: y(t) [kiloconejos]
    ax_pop_r = ax_pop.twinx()
    l2, = ax_pop_r.plot(t, y, color="tab:orange", label="y(t) – depredadores")
    ax_pop_r.set_ylabel("y(t) [kiloconejos]", color=l2.get_color())
    ax_pop_r.tick_params(axis="y", labelcolor=l2.get_color())

    # Leyenda combinada coherente con ambos ejes
    lines = [l1, l2]
    ax_pop.legend(lines, [ln.get_label() for ln in lines], loc="upper right")

    
    ax_V.plot(t, V, color="tab:blue")
    ax_V.set_xlabel("t")
    # >>> etiqueta pedida en el eje y usando mathtext <<<
    ax_V.set_ylabel(r"$V=\delta x-\gamma\,\ln x+\beta y-\alpha \ln y$", labelpad=4)
    ax_V.ticklabel_format(style="plain", useOffset=False)
    ax_V.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_as, dpi=200, bbox_inches="tight")
    plt.close(fig)

run_1a("1.a.pdf")

#1.b Problema de Landau
@dataclass
class LandauParams:
    c: float = 1.0
    q: float = 7.5284
    B0: float = 0.438
    E0: float = 0.7423
    m:  float = 3.8428
    k:  float = 1.0014
    t_max: float = 30.0
    # CI (no especificadas): pequeña perturbación para evitar equilibrio trivial
    x0: float = 0.1
    y0: float = 0.0
    vx0: float = 0.0
    vy0: float = 0.0

def landau_rhs(t, u, p: LandauParams):
    """ Ecuaciones del enunciado (c=1):
        m x¨ = qE0[sin(kx) + k x cos(kx)] - qB0 y˙
        m y¨ = qB0 x˙
    """
    x, y, vx, vy = u
    Fx = p.q * p.E0 * (math.sin(p.k * x) + p.k * x * math.cos(p.k * x))
    ax = (Fx - p.q * p.B0 * vy / p.c) / p.m
    ay = (p.q * p.B0 * vx / p.c) / p.m
    return [vx, vy, ax, ay]

def landau_invariants(u_t, p: LandauParams):
    """ Devuelve Π_y y K+U a lo largo del tiempo. """
    x, y, vx, vy = u_t
    Piy = p.m * vy - p.q * p.B0 * x / p.c
    K   = 0.5 * p.m * (vx**2 + vy**2)
    U   = - p.q * p.E0 * x * np.sin(p.k * x)
    return Piy, K + U


def run_1b(save_as="1.b.pdf"):
    p = LandauParams()
    t_span = (0.0, p.t_max)
    t_eval = np.linspace(*t_span, 6000)  # malla densa para invariantes
    u0 = [p.x0, p.y0, p.vx0, p.vy0]

    sol = solve_ivp(
        fun=landau_rhs, t_span=t_span, y0=u0, t_eval=t_eval,
        args=(p,), method="RK45", rtol=1e-9, atol=1e-12
    )
    t = sol.t
    Piy, Etot = landau_invariants(sol.y, p)

    # --- 2 subplots: Π_y(t) y K+U(t), ambos con zoom automático del eje y ---
    fig, (ax_pi, ax_E) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax_pi.plot(t, Piy, color="tab:green", linewidth=1.6)
    ax_pi.set_ylabel(r"$\Pi_y = m\,\dot y - \frac{qB_0}{c}\,x$")
    ax_pi.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax_pi.grid(True, alpha=0.3)
    zoom(ax_pi, Piy)

    ax_E.plot(t, Etot, color="tab:red", linewidth=1.6)
    ax_E.set_xlabel("t")
    ax_E.set_ylabel(r"$K+U=\frac{m}{2}(\dot x^2+\dot y^2)-qE_0\,x\sin(kx)$")
    ax_E.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax_E.grid(True, alpha=0.3)
    zoom(ax_E, Etot)

    fig.tight_layout()
    fig.savefig(save_as, dpi=220, bbox_inches="tight")
    plt.close(fig)

run_1b("1.b.pdf")


#1.c Sistema binario
class TwoBodyParams:
    G: float = 1.0
    m: float = 1.7
    r1_0: tuple[float, float] = (0.0, 0.0)
    r2_0: tuple[float, float] = (1.0, 1.0)
    v1_0: tuple[float, float] = (0.0, 0.5)
    v2_0: tuple[float, float] = (0.0, -0.5)
    t_max: float = 10.0

def twobody_rhs(t, y, p: TwoBodyParams):
    # y = [x1,y1,x2,y2, vx1,vy1,vx2,vy2]
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = y
    rx, ry = (x2 - x1), (y2 - y1)
    r2 = rx*rx + ry*ry
    r3 = (r2 + 1e-12)**1.5  # evita /0 si r -> 0
    # m1 = m2 = m
    ax1 =  p.G * p.m * rx / r3
    ay1 =  p.G * p.m * ry / r3
    ax2 = -p.G * p.m * rx / r3
    ay2 = -p.G * p.m * ry / r3
    return [vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2]

def twobody_invariants(y_t, p: TwoBodyParams):
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = y_t
    rx, ry = (x2 - x1), (y2 - y1)
    r = np.sqrt(rx*rx + ry*ry)
    # Energía total (atractiva): K + U con U = -G m^2 / r
    K = 0.5 * p.m * (vx1*vx1 + vy1*vy1 + vx2*vx2 + vy2*vy2)
    U = - p.G * (p.m**2) / r
    E = K + U
    # Momento angular total (z)
    Lz = p.m * (x1*vy1 - y1*vx1) + p.m * (x2*vy2 - y2*vx2)
    return E, Lz

def run_1c(save_as="1.c.pdf"):
    p = TwoBodyParams()
    y0 = [p.r1_0[0], p.r1_0[1], p.r2_0[0], p.r2_0[1],
          p.v1_0[0], p.v1_0[1], p.v2_0[0], p.v2_0[1]]
    t_span = (0.0, p.t_max)
    t_eval = np.linspace(*t_span, 5000)

    sol = solve_ivp(
        fun=twobody_rhs, t_span=t_span, y0=y0, t_eval=t_eval,
        args=(p,), method="RK45", rtol=1e-10, atol=1e-13  # más estricto para conservar
    )
    t = sol.t
    E, Lz = twobody_invariants(sol.y, p)

    # --- 2 subplots: E(t) y Lz(t) ---
    fig, (axE, axL) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axE.plot(t, E, color="tab:red", lw=1.8)
    axE.set_ylabel("Energía total")
    axE.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    axE.grid(True, alpha=0.3)
    zoom(axE, E)

    axL.plot(t, Lz, color="tab:green", lw=1.8)
    axL.set_xlabel("t")
    axL.set_ylabel(r"$L_z$")
    axL.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    axL.grid(True, alpha=0.3)
    zoom(axL, Lz)

    fig.tight_layout()
    fig.savefig(save_as, dpi=220, bbox_inches="tight")
    plt.close(fig)

run_1c("1.c.pdf")
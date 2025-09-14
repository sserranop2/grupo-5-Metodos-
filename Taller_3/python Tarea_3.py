#Librerias
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import math
from matplotlib.ticker import ScalarFormatter
import csv
from scipy.optimize import brentq
from scipy.special import betainc
import warnings
warnings.filterwarnings('ignore')
import pandas as pd


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

#pto 3

hbar = 0.1
a = 0.8
x0 = 10.0

def V(x):
    return (1.0 - np.exp(a*(x - x0)))**2 - 1.0

def rhs(x, y, E):
    psi, phi = y
    dpsi = phi
    dphi = ((V(x) - E) / (hbar**2)) * psi
    return np.array([dpsi, dphi], dtype=float)

def rk4_step(fun, x, y, h, E):
    k1 = fun(x, y, E)
    k2 = fun(x + 0.5*h, y + 0.5*h*k1, E)
    k3 = fun(x + 0.5*h, y + 0.5*h*k2, E)
    k4 = fun(x + h, y + h*k3, E)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def turning_points(E, x_min=0.0, x_max=20.0, N=4000):
    xs = np.linspace(x_min, x_max, N+1)
    vals = [V(x) - E for x in xs]
    roots = []
    for i in range(N):
        if vals[i] == 0:
            roots.append(xs[i])
        elif vals[i]*vals[i+1] < 0:
            a_, b_ = xs[i], xs[i+1]
            fa, fb = vals[i], vals[i+1]
            for _ in range(30):
                m = 0.5*(a_ + b_)
                fm = V(m) - E
                if fa*fm <= 0:
                    b_, fb = m, fm
                else:
                    a_, fa = m, fm
            roots.append(0.5*(a_ + b_))
    if len(roots) >= 2:
        return roots[0], roots[-1]
    return None, None

def shoot_norm(E, dx=0.01):
    x1, x2 = turning_points(E)
    if x1 is None:
        return np.inf, None, None, None
    xL = x1 - 2.0
    xR = x2 + 1.0
    y = np.array([0.0, 1e-8], dtype=float)
    xs = [xL]
    psis = [y[0]]
    x = xL
    steps = int(math.ceil((xR - xL)/dx))
    h = (xR - xL)/steps
    for _ in range(steps):
        if abs(y[0]) > 1e6 or abs(y[1]) > 1e6:
            return 1e12, None, None, None
        y = rk4_step(rhs, x, y, h, E)
        x += h
        xs.append(x)
        psis.append(float(y[0]))
    psi_end, phi_end = y[0], y[1]
    norm = math.hypot(psi_end, phi_end)
    return norm, np.array(xs), np.array(psis), (xL, xR)

def find_local_minima(E_grid, norms):
    mins = []
    for i in range(1, len(E_grid)-1):
        if norms[i] < norms[i-1] and norms[i] < norms[i+1]:
            mins.append(i)
    return mins

def golden_section_minimize(f, a_, b_, tol=1e-8, maxit=80):
    gr = (math.sqrt(5)-1)/2
    c = b_ - gr*(b_ - a_)
    d = a_ + gr*(b_ - a_)
    fc = f(c)
    fd = f(d)
    for _ in range(maxit):
        if abs(b_ - a_) < tol:
            break
        if fc < fd:
            b_, d, fd = d, c, fc
            c = b_ - gr*(b_ - a_)
            fc = f(c)
        else:
            a_, c, fc = c, d, fd
            d = a_ + gr*(b_ - a_)
            fd = f(d)
    if fc < fd:
        return c, fc
    else:
        return d, fd

def compute_all():
    E_grid = np.linspace(-0.99, -1e-5, 1200)
    norms = []
    for E in E_grid:
        n, *_ = shoot_norm(E)
        norms.append(n)
    norms = np.array(norms)
    idxs = find_local_minima(E_grid, norms)
    Es_num = []
    Psis = []
    Xs = []
    for i in idxs:
        i0 = max(i-2, 0)
        i1 = min(i+2, len(E_grid)-1)
        aE = E_grid[i0]
        bE = E_grid[i1]
        def f(E):
            n, *_ = shoot_norm(E)
            return n
        E_star, _ = golden_section_minimize(f, aE, bE, tol=1e-10)
        norm, xs, psis, _ = shoot_norm(E_star)
        if math.isfinite(norm):
            Es_num.append(E_star)
            Xs.append(xs)
            Psis.append(psis)
    order = np.argsort(Es_num)
    Es_num = [Es_num[i] for i in order]
    Xs     = [Xs[i] for i in order]
    Psis   = [Psis[i] for i in order]
    return Es_num, Xs, Psis, (E_grid, norms)

Es_num, Xs, Psis, scan = compute_all()

lam = 1.0/(a*hbar)
n_max = int(math.floor((lam - 0.5) + 1e-12))

def E_teo(n):
    return (2*lam - (n + 0.5))*(n + 0.5)/(lam**2) - 1.0

Es_teo = [E_teo(n) for n in range(n_max)]

m = min(len(Es_num), len(Es_teo))
df = pd.DataFrame({
    "n": range(m),
    "E_num": [Es_num[i] for i in range(m)],
    "E_teo": [Es_teo[i] for i in range(m)],
})
df["Diff_%"] = (abs(df["E_num"] - df["E_teo"]) / df["E_teo"].abs().clip(lower=1e-14)) * 100.0

df_out = df.copy()
df_out.to_csv("3.txt", sep="\t", index=False, float_format="%.10f")
print("Archivo escrito: 3.txt")

xx = np.linspace(0, 12, 1000)
Vv = V(xx)

plt.figure(figsize=(7.5, 5.2))
plt.plot(xx, Vv, color='k', linewidth=2.0, label='Morse potential')

to_plot = min(len(Es_num), n_max)

def spacing_local(k):
    up = Es_num[k+1] - Es_num[k] if k+1 < to_plot else np.inf
    dn = Es_num[k]   - Es_num[k-1] if k-1 >= 0      else np.inf
    return min(up, dn)

for n in range(to_plot):
    E = Es_num[n]
    x = Xs[n]; psi = Psis[n]
    msk = (x >= 0) & (x <= 12)
    if not np.any(msk):
        continue
    x_cut  = x[msk]
    psi_cut = psi[msk]
    s = 0.45 * spacing_local(n)
    psi_vis = psi_cut/(np.max(np.abs(psi_cut)) + 1e-15) * s + E
    plt.plot(x_cut, psi_vis, lw=1.4)
    plt.hlines(E, 0, 12, linestyles='dotted', linewidth=0.8, color='0.7')

plt.xlim(0, 12);  plt.ylim(-1.15, 0.05)
plt.xlabel("x");  plt.ylabel("Energy")
plt.legend(loc="lower left")
plt.grid(alpha=0.3); plt.tight_layout()

plt.savefig("3.pdf", bbox_inches="tight")




#Pto 4
def f(state, alpha):
    """Campo vectorial. state = [theta, r, Pth, Pr]."""
    theta, r, Pth, Pr = state
    den = 1.0 + r
    if den <= 1e-8:
        den = 1e-8
    dtheta = Pth / (den**2)
    dr = Pr
    dPth = - (alpha**2) * den * np.sin(theta)
    dPr  =   (alpha**2) * np.cos(theta) - r + (Pth**2) / (den**3)
    return np.array([dtheta, dr, dPth, dPr], dtype=float)

def rk4_step(state, h, alpha):
    k1 = f(state, alpha)
    k2 = f(state + 0.5*h*k1, alpha)
    k3 = f(state + 0.5*h*k2, alpha)
    k4 = f(state + h*k3, alpha)
    return state + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def theta_dot(state, alpha):
    _, r, Pth, _ = state
    den = max(1e-8, 1.0 + r)
    return Pth / (den**2)

def wrap_pi(x):
    return np.arctan2(np.sin(x), np.cos(x))

def H(state, alpha):
    theta, r, Pth, Pr = state
    den = 1.0 + r
    if den <= 1e-8:
        den = 1e-8
    return 0.5*Pr**2 + 0.5*(Pth**2)/(den**2) + 0.5*r**2 - (alpha**2)*den*np.cos(theta)

def drift_energia_abs(alpha=1.1, tmax=2000.0, dt=0.02):
    state = np.array([np.pi/2, 0.0, 0.0, 0.0], float)
    H0 = H(state, alpha)
    t = 0.0
    max_abs = 0.0
    while t < tmax:
        state = rk4_step(state, dt, alpha)
        Hi = H(state, alpha)
        max_abs = max(max_abs, abs(Hi - H0))
        t += dt
    rel = max_abs / (1.0 + abs(H0))
    print(f"[alpha={alpha:.3f}] |H-H0|_max ≈ {max_abs:.3e}   (rel≈{rel:.3e}, H0={H0:.3e})")

def grafica_energia(alpha=1.1, tmax=200.0, dt=0.02):
    state = np.array([np.pi/2, 0.0, 0.0, 0.0], float)
    H0 = H(state, alpha)
    ts, Hs = [0.0], [H0]
    t = 0.0
    while t < tmax:
        state = rk4_step(state, dt, alpha)
        t += dt
        ts.append(t); Hs.append(H(state, alpha))
    plt.figure(figsize=(6,3))
    plt.plot(ts, np.array(Hs) - H0, lw=1)
    plt.xlabel("t"); plt.ylabel("H(t) - H0")
    plt.title(f"Energía - alpha={alpha:.3f}")
    plt.tight_layout(); plt.show()

def seccion_poincare(alpha, tmax=1_000.0, dt=0.02,
                     thetas=(0.0,), sentido="ambos", max_puntos=np.inf):
    """
    Devuelve arrays (r, Pr) cuando θ cruza cualquiera de las θ_k en 'thetas'.
    - sentido: "ambos" (default), "+" (sólo dθ/dt>0) o "-" (sólo dθ/dt<0)
    - thetas: iterable de valores objetivo para θ_k
    CI del enunciado: θ0 = π/2, r0=0, Pθ0=0, Pr0=0
    """
    thetas = np.atleast_1d(np.array(thetas, float))
    state = np.array([np.pi/2, 0.0, 0.0, 0.0], dtype=float)
    t = 0.0
    r_sec, Pr_sec = [], []
    while t < tmax and len(r_sec) < max_puntos:
        s0 = state
        th0 = s0[0]
        s1 = rk4_step(s0, dt, alpha)
        th1 = s1[0]
        for thk in thetas:
            phi0 = wrap_pi(th0 - thk)
            phi1 = wrap_pi(th1 - thk)
            if (phi0 <= 0.0 and phi1 > 0.0) or (phi0 >= 0.0 and phi1 < 0.0):
                denom = (phi1 - phi0)
                if abs(denom) < 1e-14:
                    continue
                frac = np.clip(-phi0/denom, 0.0, 1.0)
                sc = s0 + frac*(s1 - s0)
                v = theta_dot(sc, alpha)
                if (sentido == "+") and not (v > 0.0):
                    continue
                if (sentido == "-") and not (v < 0.0):
                    continue
                r_sec.append(sc[1])
                Pr_sec.append(sc[3])
                if len(r_sec) >= max_puntos:
                    break
        state = s1
        t += dt
    return np.array(r_sec), np.array(Pr_sec)

def run_experimento(
    alphas=np.linspace(1.0, 1.2, 9),
    tmax=2_000.0,
    dt=0.02,
    warmup=300.0,
    puntos_max_por_alpha=np.inf,
    num_secciones=1,
    thetas=None,
    sentido="ambos",
    archivo_salida="secciones.pdf"
):
    if thetas is None:
        thetas = np.linspace(0.0, 2*np.pi, num_secciones, endpoint=False)
    for a in alphas:
        seccion_poincare(a, tmax=warmup, dt=dt, thetas=thetas, sentido=sentido, max_puntos=0)
    plt.figure(figsize=(7, 5))
    for a in alphas:
        r_pts, Pr_pts = seccion_poincare(
            a, tmax=tmax, dt=dt, thetas=thetas, sentido=sentido,
            max_puntos=puntos_max_por_alpha
        )
        if len(r_pts) == 0:
            print(f"Advertencia: no hubo cruces para alpha={a:.3f}")
            continue
        plt.scatter(r_pts, Pr_pts, s=6, label=fr"$\alpha={a:.3f}$", alpha=0.75)
    plt.xlabel(r"$r$")
    plt.ylabel(r"$P_r$")
    plt.title(r"Sección de Poincaré: $\theta=0$ (ambos sentidos)")
    plt.legend(loc="best", markerscale=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(archivo_salida, dpi=300)
    plt.show()
    print(f"Figura guardada en: {archivo_salida}")

drift_energia_abs(alpha=1.10, tmax=2_000.0, dt=0.02)
grafica_energia(alpha=1.10, tmax=400.0, dt=0.02)

alphas = np.linspace(1.0, 1.2, 9)
run_experimento(
    alphas=alphas,
    tmax=10_000.0,
    dt=0.02,
    warmup=300.0,
    puntos_max_por_alpha=np.inf,
    thetas=(0.0,),
    num_secciones=1,
    sentido="ambos",
    archivo_salida="4.pdf"
)


#Punto 5
def punto_5():
    """
    Punto 5: Circuito genético oscilatorio
    """
    print("Resolviendo Punto 5: Circuito genético oscilatorio...")
    
    def genetic_circuit(t, y, alpha, beta, alpha0, n):
        """
        Sistema de ecuaciones para el circuito genético
        y = [m1, m2, m3, p1, p2, p3]
        """
        m1, m2, m3, p1, p2, p3 = y
        
        # Ecuaciones para mRNA (índices cíclicos: p0 ≡ p3)
        dm1dt = alpha / (1 + p3**n) + alpha0 - m1
        dm2dt = alpha / (1 + p1**n) + alpha0 - m2
        dm3dt = alpha / (1 + p2**n) + alpha0 - m3
        
        # Ecuaciones para proteínas
        dp1dt = -beta * (p1 - m1)
        dp2dt = -beta * (p2 - m2)
        dp3dt = -beta * (p3 - m3)
        
        return [dm1dt, dm2dt, dm3dt, dp1dt, dp2dt, dp3dt]
    
    # Parámetros
    n = 2
    alphas = np.logspace(0, 5, 50)  # α ∈ [1, 10^5]
    betas = np.logspace(0, 2, 50)   # β ∈ [1, 100]
    t_max = 400
    
    # Matriz para almacenar amplitudes
    amplitudes = np.zeros((len(betas), len(alphas)))
    
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            alpha0 = alpha / 1000
            
            # Condiciones iniciales (pequeñas perturbaciones para evitar equilibrio trivial)
            y0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            
            try:
                # Resolver el sistema
                t_eval = np.linspace(0, t_max, 10000)
                sol = solve_ivp(genetic_circuit, [0, t_max], y0, 
                              args=(alpha, beta, alpha0, n), 
                              t_eval=t_eval, method='LSODA', 
                              rtol=1e-8, atol=1e-10)
                
                if sol.success:
                    # Extraer p3 y calcular amplitud de las últimas oscilaciones
                    p3 = sol.y[5]  # p3 está en el índice 5
                    
                    # Tomar la última parte de la simulación (después de ~10 oscilaciones)
                    start_idx = len(t_eval) // 2  # Segunda mitad
                    p3_final = p3[start_idx:]
                    
                    # Calcular amplitud como diferencia entre max y min
                    amplitude = np.max(p3_final) - np.min(p3_final)
                    amplitudes[i, j] = amplitude
                else:
                    amplitudes[i, j] = np.nan
                    
            except Exception as e:
                amplitudes[i, j] = np.nan
    
    # Reemplazar valores muy pequeños o NaN
    amplitudes[amplitudes < 1e-10] = 1e-10
    amplitudes[np.isnan(amplitudes)] = 1e-10
    
    # Crear la gráfica
    plt.figure(figsize=(10, 8))
    
    # Usar log10 de la amplitud para el color
    log_amplitudes = np.log10(amplitudes)
    
    # Crear meshgrid para pcolormesh
    Alpha, Beta = np.meshgrid(alphas, betas)
    
    plt.pcolormesh(Alpha, Beta, log_amplitudes, shading='auto', cmap='viridis')
    plt.colorbar(label='log₁₀(Amplitud)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('α')
    plt.ylabel('β')
    plt.title('Amplitud de oscilación de p₃ en circuito genético')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('5.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Punto 5 completado. Gráfica guardada en '5.pdf'")

def punto_6():
    """
    Punto 6: Teoría de vigas quasiestáticas
    """
    print("Resolviendo Punto 6: Teoría de vigas quasiestáticas...")
    
    def beam_equations(x, y):
        """
        Sistema de ecuaciones de Timoshenko-Ehrenfest
        y = [phi, dphi_dx, EI_dphi_dx, w]
        """
        phi, dphi_dx, EI_dphi_dx, w = y
        
        # Parámetros
        E, I, A, G, kappa = 1, 1, 1, 1, 5/6
        
        # Fuerza aplicada usando beta incompleta regularizada
        q_x = betainc(3, 6, x)
        
        # Ecuaciones del sistema
        d_phi = dphi_dx
        d_dphi_dx = EI_dphi_dx / (E * I)
        d_EI_dphi_dx = q_x
        d_w = phi - EI_dphi_dx / (kappa * A * G)
        
        return [d_phi, d_dphi_dx, d_EI_dphi_dx, d_w]
    
    # Dominio de la viga
    x_span = [0, 1]
    x_eval = np.linspace(0, 1, 1000)
    
    # Condiciones iniciales (todas cero como sugiere la pista)
    y0 = [0, 0, 0, 0]  # [phi(0), phi'(0), (EI*phi')(0), w(0)]
    
    # Resolver el sistema
    sol = solve_ivp(beam_equations, x_span, y0, t_eval=x_eval, 
                   method='RK45', rtol=1e-8, atol=1e-10)
    
    if not sol.success:
        print("Error en la solución del sistema de vigas")
        return
    
    # Extraer soluciones
    phi = sol.y[0]
    w = sol.y[3]
    
    # Definir geometría de la viga
    y_top = 0.2
    y_bottom = -0.2
    
    # Crear puntos para la visualización de la viga
    n_points = len(x_eval)
    
    # Viga antes de la deformación (rectángulo)
    x_original = x_eval
    y_top_original = np.full(n_points, y_top)
    y_bottom_original = np.full(n_points, y_bottom)
    
    # Viga después de la deformación
    # Deformación horizontal: u_x(x,y) = -y*phi(x)
    # Deformación vertical: u_y(x,y) = w(x)
    
    # Puntos deformados para la superficie superior (y = y_top)
    x_top_def = x_eval - y_top * phi  # x + u_x
    y_top_def = y_top_original + w     # y + u_y
    
    # Puntos deformados para la superficie inferior (y = y_bottom)
    x_bottom_def = x_eval - y_bottom * phi  # x + u_x
    y_bottom_def = y_bottom_original + w    # y + u_y
    
    # Crear la gráfica
    plt.figure(figsize=(12, 8))
    
    # Viga original
    plt.fill_between(x_original, y_bottom_original, y_top_original, 
                     alpha=0.3, color='gray', label='Viga original')
    plt.plot(x_original, y_top_original, 'k--', linewidth=1, alpha=0.7)
    plt.plot(x_original, y_bottom_original, 'k--', linewidth=1, alpha=0.7)
    
    # Viga deformada
    plt.fill_between(x_top_def, y_bottom_def, y_top_def, 
                     alpha=0.6, color='red', label='Viga deformada')
    plt.plot(x_top_def, y_top_def, 'r-', linewidth=2, label='Superficie superior deformada')
    plt.plot(x_bottom_def, y_bottom_def, 'r-', linewidth=2, label='Superficie inferior deformada')
    
    # Línea central deformada
    x_center_def = x_eval
    y_center_def = w
    plt.plot(x_center_def, y_center_def, 'b-', linewidth=2, label='Línea central deformada')
    
    # Configuración de la gráfica
    plt.xlabel('x (longitud de la viga)')
    plt.ylabel('y (altura)')
    plt.title('Deformación de viga con carga distribuida\n(Teoría de Timoshenko-Ehrenfest)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Añadir anotaciones
    plt.annotate('Extremo fijo', xy=(0, 0), xytext=(0.1, 0.3),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
    plt.annotate('Extremo libre', xy=(1, w[-1]), xytext=(0.8, w[-1] + 0.1),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('6.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Punto 6 completado. Gráfica guardada en '6.pdf'")
    print(f"Deformación máxima vertical: {np.max(np.abs(w)):.6f}")
    print(f"Deformación máxima horizontal: {np.max(np.abs(phi)):.6f}")

if __name__ == "__main__":
    # Resolver punto 5
    punto_5()
    
    # Resolver punto 6
    punto_6()
    
    print("\n¡Puntos 5 y 6 completados exitosamente!")
    print("Archivos generados:")
    print("- 5.pdf: Mapa de amplitudes del circuito genético")
    print("- 6.pdf: Deformación de la viga")


#7 Perfil de densidad estelar
# --- utilidades compactas ---
def is_int(n: float) -> bool:          # ¿n entero?
    return float(n).is_integer()

def series_center(n, x):               # expansión regular en x≈0
    return (1 - x*x/6 + n*x**4/120, -x/3 + n*x**3/30)

# evento: θ=0 (descendiendo)
def ev_zero(x, y): return y[0]
ev_zero.terminal, ev_zero.direction = True, -1.0

# refinamiento de raíz sobre la solución densa
def refine_root(sol, a, b):
    f = lambda xx: float(sol.sol(xx)[0])
    return float(brentq(f, a, b, xtol=1e-14, rtol=1e-12))

def solve_polytrope(n, x_plot_max=32.0, x_table_max=None):
    """Devuelve (x*, θ'(x*), M, ρc/<ρ>, (xx,θ)) con una única integración SciPy."""
    if x_table_max is None: x_table_max = 120.0 if n <= 4.5 else 600.0
    eps0 = 1e-8
    y0 = series_center(n, eps0)

    # RHS seguro hasta el primer cero: clamp cuando n no entero y θ<0
    def rhs_upto_zero(x, y):
        th, thp = y
        thn = th**n if (is_int(n) or th >= 0.0 or n == 0.0) else 0.0
        return (thp, -(2/x)*thp - (1.0 if n==0.0 else thn))

    sol = solve_ivp(rhs_upto_zero, (eps0, x_table_max), y0,
                    events=ev_zero, method="DOP853",
                    rtol=1e-11, atol=1e-14, max_step=0.02, dense_output=True)

    # ¿hubo cruce?
    if sol.t_events[0].size:
        x_star = float(sol.t_events[0][0])
        th_star, thp_star = sol.sol(x_star)
        M  = - x_star**2 * thp_star
        Rr = - (1/3) * (x_star / thp_star)

        # curva hasta x* y extensión (solo para la figura)
        xs = np.linspace(sol.t[0], min(x_star, x_plot_max), 1500)
        th = sol.sol(xs)[0]
        if x_star < x_plot_max:
            δ = 1e-8*(1+x_star)
            x0 = x_star + δ
            y0_ext = (thp_star*δ, thp_star)
            # tras el cero: entero→θ^n; fraccionario→|θ|^n
            def rhs_ext(x, y):
                th, thp = y
                base = th**n if is_int(n) else abs(th)**n
                return (thp, -(2/x)*thp - base)
            sol2 = solve_ivp(rhs_ext, (x0, x_plot_max), y0_ext,
                             method="DOP853", rtol=1e-11, atol=1e-14,
                             max_step=0.02, dense_output=True)
            xs2 = np.linspace(x0, x_plot_max, 900)
            th2 = sol2.sol(xs2)[0]
            xs, th = np.concatenate([xs, xs2]), np.concatenate([th, th2])
        return x_star, float(thp_star), float(M), float(Rr), (xs, th)

    # sin cruce (≈ n=5): radio ∞; masa por límite; ρc/<ρ>=∞
    X  = sol.t[-1]
    thp_last = float(sol.y[1, -1])
    M = - X**2 * thp_last
    xs = np.linspace(sol.t[0], x_plot_max, 1500)
    th = sol.sol(xs)[0]
    return float('inf'), thp_last, float(M), float('inf'), (xs, th)

def run_7(save_csv="7.csv", save_fig="7.pdf"):
    x_plot_max = 32.0
    ns = [0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0]

    rows = [("Índice n","Radio x*","Masa (∝ -x*^2 θ'(x*))","ρ_c/⟨ρ⟩")]
    curves = {}
    for n in ns:
        x_star, thp_star, M, Rr, (xx, th) = solve_polytrope(
            n, x_plot_max=x_plot_max, x_table_max=(120.0 if n <= 4.5 else 600.0)
        )
        fmt = lambda v: "inf" if (isinstance(v,float) and np.isinf(v)) else f"{v:.5f}"
        rows.append((f"{n:.1f}", fmt(x_star), fmt(M), fmt(Rr)))
        curves[n] = (xx, th)

    with open(save_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    fig, ax = plt.subplots(figsize=(19, 9), constrained_layout=True)
    palette = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple",
               "sienna","tab:pink","tab:gray"]
    for i, n in enumerate(ns):
        xx, th = curves[n]
        ax.plot(xx, th, lw=2, color=palette[i%len(palette)], label=f"{n:g}")
    ax.set(xlim=(0, x_plot_max), ylim=(-0.4, 1.0),
           xlabel="Dimensionless radius", ylabel=r"$\theta(x)$")
    ax.axhline(0, color="k", lw=0.9, ls="--", alpha=0.45)
    ax.legend(title="Polytropic index", ncols=8, frameon=False, fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    fig.savefig(save_fig, dpi=200)
    plt.close(fig)

run_7("7.csv", "7.pdf")
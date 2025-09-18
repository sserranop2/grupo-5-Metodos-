#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Punto_2.py — Patrones de Turing (Reacción–Difusión 2D, BC periódicas)
Imágenes nítidas (viridis) con ejes/Colorbar y pie con fórmula + constantes.
"""

from dataclasses import dataclass
from textwrap import dedent
import numpy as np
import matplotlib
matplotlib.use("Agg")  # sin ventanas
import matplotlib.pyplot as plt
import os
import warnings

# SciPy es opcional: si está, usamos gaussian_filter para un look más suave
try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------- Parámetros y utils ---------------------------

@dataclass
class RDParams:
    L: float = 3.0
    N: int = 128
    alpha: float = 2.8e-4
    beta: float  = 5.0e-2
    dt: float = None
    tmax: float = 15.0
    method: str = "heun"        # "euler" o "heun"
    seed: int = 7
    noise_sigma: float = 0.02
    check_every: int = 100
    tol_rms: float = 1e-5

    def derive_dt(self) -> float:
        dx = self.L / self.N
        return 0.5 * (dx*dx) / (4.0 * max(self.alpha, self.beta))

def sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.,=" else "_" for c in s)

# ------------------- Operador y reacciones (nombres originales) -------------------

def laplacian_periodic(Z: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(Z, +1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, +1, axis=1) + np.roll(Z, -1, axis=1) -
            4.0*Z) / (dx*dx)

def reaction_slide(u, v):
    F = u - u**3 - v - 0.05
    G = 10.0*(u - v)
    return F, G

def reaction_gray_scott(u, v, f=0.0367, k=0.0649):
    F = -u*v*v + f*(1.0 - u)
    G =  u*v*v - (f + k)*v
    return F, G

def reaction_brusselator(u, v, A=1.0, B=3.2):
    F = A - (B+1.0)*u + (u*u*v)
    G = B*u - (u*u*v)
    return F, G

def model_formula(model: str, kw: dict) -> str:
    kw = {} if kw is None else kw
    if model == "slide":
        return r"F(u,v)=u-u^3-v-0.05;  G(u,v)=10(u-v)"
    elif model == "gray_scott":
        f = kw.get("f", 0.0367); k = kw.get("k", 0.0649)
        return rf"F=-u v^2+{f}(1-u);  G=u v^2-({f}+{k})v"
    elif model == "brusselator":
        A = kw.get("A", 1.0); B = kw.get("B", 3.2)
        return rf"F={A}-({B}+1)u+u^2 v;  G={B}u-u^2 v"
    else:
        return ""

def _const_string(model: str, params: RDParams, kw: dict) -> str:
    kw = {} if kw is None else kw
    if model == "gray_scott":
        return f"α={params.alpha}, β={params.beta}, f={kw.get('f',0.0367)}, k={kw.get('k',0.0649)}"
    if model == "brusselator":
        return f"α={params.alpha}, β={params.beta}, A={kw.get('A',1.0)}, B={kw.get('B',3.2)}"
    return f"α={params.alpha}, β={params.beta}"

# ------------------------------- Simulador -------------------------------

def simulate(params: RDParams, model="slide", model_kwargs=None, max_steps=None):
    if model_kwargs is None:
        model_kwargs = {}
    rng = np.random.default_rng(params.seed)

    N, L = params.N, params.L
    dx = L / N
    dt = params.dt if params.dt is not None else params.derive_dt()
    steps = int(np.ceil(params.tmax / dt)) if max_steps is None else max_steps

    # equilibrio homogéneo aproximado
    if model == "slide":
        u0 = -np.cbrt(0.05); v0 = u0
    else:
        u0 = 0.0; v0 = 0.0

    u = (u0 + params.noise_sigma * rng.standard_normal((N, N))).astype(np.float64, copy=False)
    v = (v0 + params.noise_sigma * rng.standard_normal((N, N))).astype(np.float64, copy=False)

    # selector de reacciones
    if model == "slide":
        react = lambda uu, vv: reaction_slide(uu, vv)
    elif model == "gray_scott":
        react = lambda uu, vv: reaction_gray_scott(uu, vv, **model_kwargs)
    elif model == "brusselator":
        react = lambda uu, vv: reaction_brusselator(uu, vv, **model_kwargs)
    else:
        raise ValueError("Unknown model")

    last_u = u.copy(); last_v = v.copy()
    for n in range(steps):
        lap_u = laplacian_periodic(u, dx)
        lap_v = laplacian_periodic(v, dx)
        Fu, Gv = react(u, v)

        if params.method == "euler":
            u_new = u + dt*(params.alpha*lap_u + Fu)
            v_new = v + dt*(params.beta *lap_v + Gv)
        else:
            # Heun (RK2)
            u_p = u + dt*(params.alpha*lap_u + Fu)
            v_p = v + dt*(params.beta *lap_v + Gv)

            lap_u_p = laplacian_periodic(u_p, dx)
            lap_v_p = laplacian_periodic(v_p, dx)
            Fu_p, Gv_p = react(u_p, v_p)

            Ru  = params.alpha*lap_u   + Fu
            Rv  = params.beta *lap_v   + Gv
            RuP = params.alpha*lap_u_p + Fu_p
            RvP = params.beta *lap_v_p + Gv_p

            u_new = u + 0.5*dt*(Ru + RuP)
            v_new = v + 0.5*dt*(Rv + RvP)

        u, v = u_new, v_new

        if (n+1) % params.check_every == 0:
            rms_u = float(np.sqrt(np.mean((u - last_u)**2)))
            rms_v = float(np.sqrt(np.mean((v - last_v)**2)))
            last_u[...] = u; last_v[...] = v
            if rms_u < params.tol_rms and rms_v < params.tol_rms:
                break

    meta = {"dt": dt, "steps": n+1, "dx": dx}
    return u, v, meta

# ------------------------ Guardado (ejes + pie fuera) ------------------------

def _upsample_nearest(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    return np.kron(arr, np.ones((factor, factor), dtype=arr.dtype))

def _auto_contrast_limits(arr: np.ndarray, qlow=0.02, qhigh=0.98):
    lo, hi = np.quantile(arr, [qlow, qhigh])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    return lo, hi

def save_field_image(u, L, out_png, title=None, footer_text=None,
                     cmap="viridis", dpi=340, figsize=(6.4, 6.0), upsample=3,
                     smooth_sigma=0.6, show_axes=True, show_colorbar=True,
                     footer_margin=0.15):
    """
    Imagen nítida con ejes y colorbar; pie (fórmula + constantes) debajo, sin superponer.
    """
    U = u
    if _HAS_SCIPY and smooth_sigma and smooth_sigma > 0:
        U = gaussian_filter(U, sigma=smooth_sigma)

    Uplot = _upsample_nearest(U, upsample)
    vmin, vmax = _auto_contrast_limits(Uplot, 0.02, 0.98)

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=False)
    ax = fig.add_subplot(1,1,1)

    im = ax.imshow(Uplot, origin="lower", extent=[0, L, 0, L],
                   cmap=cmap, interpolation="bilinear", aspect="equal",
                   vmin=vmin, vmax=vmax)

    if show_axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    else:
        ax.set_xticks([]); ax.set_yticks([])

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("u(x,y)")

    # Reservar espacio inferior para el pie
    fig.subplots_adjust(bottom=footer_margin)

    if footer_text:
        fig.text(0.5, footer_margin*0.5, footer_text, ha="center", va="center", fontsize=10)

    if title:
        ax.set_title(title)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def write_caption(out_txt, text):
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)

def run_and_save(model, params: RDParams, model_kwargs=None, title="patron",
                 out_dir=".", upsample=3, cmap="viridis"):
    if model_kwargs is None:
        model_kwargs = {}
    u, v, meta = simulate(params, model=model, model_kwargs=model_kwargs)

    base = f"2_{sanitize(title)}"
    png_path = os.path.join(out_dir, base + ".png")
    txt_path = os.path.join(out_dir, base + ".txt")

    footer = model_formula(model, model_kwargs)
    consts = _const_string(model, params, model_kwargs)
    footer_full = f"{footer}    |    {consts}"

    save_field_image(u, params.L, png_path, title=None, footer_text=footer_full,
                     cmap=cmap, upsample=upsample, dpi=340, figsize=(6.4,6.0),
                     smooth_sigma=0.6, show_axes=True, show_colorbar=True,
                     footer_margin=0.16)

    caption = dedent(f"""
        {title}
        Modelo: {model}.
        Parámetros: α={params.alpha}, β={params.beta}, L={params.L}, N={params.N}.
        Discretización: dx={meta['dx']:.6g}, dt={meta['dt']:.6g}, método={params.method}, pasos={meta['steps']}.
        Condiciones de frontera: periódicas. CI: ruido gaussiano σ={params.noise_sigma}, semilla={params.seed}.
        Funciones: {footer}
        Constantes: {consts}
        Extra (modelo): {model_kwargs}
    """).strip()
    write_caption(txt_path, caption)

    return {"png": png_path, "txt": txt_path, "meta": meta}

# --------------------------------- Casos ---------------------------------

def main():
    # 1) Modelo del enunciado (laberintos)
    params1 = RDParams(alpha=2.8e-4, beta=5.0e-2, N=128, tmax=15.0,
                       method="heun", seed=7, noise_sigma=0.02)
    run_and_save("slide", params1, title="laberintos_slide", upsample=3, cmap="viridis")

    # 2) Gray–Scott (motas)
    params2 = RDParams(alpha=2.8e-4, beta=5.0e-2, N=128, tmax=12.0,
                       method="heun", seed=3, noise_sigma=0.02)
    run_and_save("gray_scott", params2,
                 model_kwargs={"f": 0.0367, "k": 0.0649},
                 title="motas_gray_scott", upsample=3, cmap="viridis")

    # 3) Brusselator (puede oscilar)
    params3 = RDParams(alpha=2.8e-4, beta=5.0e-2, N=128, tmax=12.0,
                       method="heun", seed=5, noise_sigma=0.02)
    run_and_save("brusselator", params3,
                 model_kwargs={"A": 1.0, "B": 3.2},
                 title="brusselator_oscila", upsample=3, cmap="viridis")

if __name__ == "__main__":
    np.seterr(all="ignore")
    main()

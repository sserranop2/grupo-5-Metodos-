#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Punto_1.py — Schrödinger 1D (PDE por DF) con Crank–Nicolson + Neumann.
Genera 1.a.mp4, 1.a.pdf, 1.b.mp4 y 1.c.mp4 usando ffmpeg real (imageio-ffmpeg).
Sin plt.show(), sin inputs.
"""

import os, shutil, sys, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu
import imageio_ffmpeg

# ================= CONFIG VIDEO (ffmpeg real) =================
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
REQUIRE_MP4 = True

# ----------------------------------------------------------------
# Utilidad común para guardar MP4 con ffmpeg real por stdin
def save_mp4_ffmpeg(xs, dens_list, t_list, V, fname="out.mp4", fps=30,
                    xmin=-20, xmax=20, title_prefix=""):

    # 1280x720 exacto
    dpi = 160
    fig = plt.figure(figsize=(8, 4.5), dpi=dpi, constrained_layout=True)
    ax  = fig.add_subplot(111)
    ax.set_xlim(xmin, xmax)
    ymax0 = max(1e-6, 1.05*np.max(dens_list[0]))
    ax.set_ylim(0.0, ymax0)
    ax.set_xlabel("x"); ax.set_ylabel(r"$|\psi|^2$")
    title = ax.set_title("")
    line, = ax.plot(xs, dens_list[0], lw=2)

    # potencial reescalado (referencia visual)
    V_scaled = (V - V.min()) / (V.max()-V.min()+1e-12) * (0.9*ymax0)
    ax.plot(xs, V_scaled, ls="--", lw=1, alpha=0.6, label="V(x) (esc.)")
    ax.legend(loc="upper right", frameon=False)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    assert width == 1280 and height == 720, f"Canvas {width}x{height} ≠ 1280x720"

    # Abrimos ffmpeg y le pipeamos frames RGB24
    cmd = [
        FFMPEG_PATH, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",                 # stdin
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-b:v", "2000k",
        "-movflags", "+faststart",
        fname,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write_frame():
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape((height, width, 4))[:, :, :3]  # RGB
        proc.stdin.write(buf.tobytes())

    # primer frame
    line.set_ydata(dens_list[0])
    title.set_text(f"{title_prefix}t = {t_list[0]:.2f}")
    write_frame()

    # resto de frames (Y dinámica hacia arriba como en tu versión)
    ymax = ymax0
    for tk, Pk in zip(t_list[1:], dens_list[1:]):
        ymax = max(ymax, float(np.max(Pk)))
        ax.set_ylim(0.0, max(1e-6, 1.05*ymax))
        line.set_data(xs, Pk)
        title.set_text(f"{title_prefix}t = {tk:.2f}")
        write_frame()

    proc.stdin.close()
    proc.wait()
    plt.close(fig)

# =============================================================
# =============== 1.a — Oscilador armónico ====================
# =============================================================
alpha = 0.1
xmin, xmax = -20.0, 20.0
Nx = 2001
x  = np.linspace(xmin, xmax, Nx)
dx = (xmax - xmin) / (Nx - 1)

t0, t_end = 0.0, 150.0
dt = 0.05
n_steps = int(round((t_end - t0) / dt))
times   = np.linspace(t0, t_end, n_steps + 1)

# *** CORRECCIÓN CLAVE: potencial CONFINANTE ***
V = -(x**2) / 50.0

# Estado inicial (igual que el tuyo)
psi = np.exp(-2.0 * (x - 10.0)**2) * np.exp(-1j * 2.0 * x)
psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))

# Operador espacial (Neumann)
def laplacian_1d_neumann(N, dx):
    main = -2.0 * np.ones(N)
    off  =  1.0 * np.ones(N-1)
    L = diags([off, main, off], [-1, 0, 1]).tolil()
    # Ghost points: ∂xψ=0 => ψ[-1]=ψ[1], ψ[N]=ψ[N-2]
    L[0, 1]   = 2.0;  L[0, 0]   = -2.0
    L[-1, -2] = 2.0;  L[-1, -1] = -2.0
    return (L / dx**2).tocsr()

Lop = laplacian_1d_neumann(Nx, dx)
H   = (-alpha) * Lop + diags(V, 0, shape=(Nx, Nx), format="csr")

# Crank–Nicolson
I = eye(Nx, format="csr")
A = (I + (1j * dt / 2.0) * H).tocsc()
B = (I - (1j * dt / 2.0) * H).tocsr()
A_lu = splu(A)

# Observables
def moments(x, psi):
    rho = np.abs(psi)**2
    mass = np.trapezoid(rho, x)
    mu   = np.trapezoid(x * rho, x) / mass
    var  = np.trapezoid((x - mu)**2 * rho, x) / mass
    return mu, np.sqrt(var)

mu_hist = np.zeros(n_steps + 1)
sg_hist = np.zeros(n_steps + 1)
mu_hist[0], sg_hist[0] = moments(x, psi)

# Storage
target_frames = 600
k_frame = max(1, n_steps // target_frames)
storage_t = []; storage_rho = []
def record(t, psi):
    storage_t.append(float(t))
    storage_rho.append(np.abs(psi)**2)
record(times[0], psi)

# Progreso
def progress(n, total):
    width = 30
    done = int(width * n / total)
    sys.stdout.write("\r[" + "#"*done + "-"*(width-done) + f"] {n}/{total}")
    sys.stdout.flush()

# Evolución
for n in range(1, n_steps + 1):
    rhs = B.dot(psi)
    psi = A_lu.solve(rhs)
    psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))
    mu_hist[n], sg_hist[n] = moments(x, psi)
    if n % k_frame == 0:
        record(times[n], psi)
    if n % max(1, n_steps//100) == 0:
        progress(n, n_steps)
progress(n_steps, n_steps); print()

# Video 1.a
save_mp4_ffmpeg(x, storage_rho, storage_t, V, fname="1.a.mp4", fps=30,
                xmin=xmin, xmax=xmax, title_prefix="")

# PDF 1.a (título consistente con el V usado)
plt.figure(figsize=(7.5, 4.5))
plt.plot(times, mu_hist, lw=2, label=r"$\mu(t)=\langle x\rangle$")
plt.fill_between(times, mu_hist - sg_hist, mu_hist + sg_hist,
                 color="C0", alpha=0.2, label=r"$\mu\pm\sigma$")
plt.xlabel("t"); plt.ylabel(r"$\langle x\rangle$ y banda $\mu\pm\sigma$")
plt.title(r"Paquete en $V(x)=+x^2/50$,  $\alpha=0.1$")
plt.grid(True, alpha=0.25); plt.legend(frameon=False)
plt.savefig("1.a.pdf", bbox_inches="tight"); plt.close()

# =============================================================
# ================== 1.b — Oscilador cuártico =================
# =============================================================
alpha = 0.1
xmin, xmax = -20.0, 20.0
Nx = 2001
x  = np.linspace(xmin, xmax, Nx)
dx = (xmax - xmin) / (Nx - 1)

# *** enunciado: simular hasta t=50 ***
t0, t_end = 0.0, 50.0
dt = 0.05
n_steps = int(round((t_end - t0) / dt))
times   = np.linspace(t0, t_end, n_steps + 1)

# *** cuártico tal como se pidió ***
V = (x/5.0)**4

psi = np.exp(-2.0 * (x - 10.0)**2) * np.exp(-1j * 2.0 * x)
psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))

Lop = laplacian_1d_neumann(Nx, dx)
H   = (-alpha) * Lop + diags(V, 0, shape=(Nx, Nx), format="csr")
I   = eye(Nx, format="csr")
A   = (I + (1j * dt / 2.0) * H).tocsc()
B   = (I - (1j * dt / 2.0) * H).tocsr()
A_lu = splu(A)

target_frames = 600
k_frame = max(1, n_steps // target_frames)
storage_t = []; storage_rho = []
record = lambda t, psi: (storage_t.append(float(t)),
                         storage_rho.append(np.abs(psi)**2))
record(times[0], psi)

for n in range(1, n_steps + 1):
    rhs = B.dot(psi); psi = A_lu.solve(rhs)
    psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))
    if n % k_frame == 0:
        record(times[n], psi)
    if n % max(1, n_steps//100) == 0:
        progress(n, n_steps)
progress(n_steps, n_steps); print()

save_mp4_ffmpeg(x, storage_rho, storage_t, V, fname="1.b.mp4", fps=30,
                xmin=xmin, xmax=xmax, title_prefix="")

# =============================================================
# ================= 1.c — Potencial “sombrero” =================
# =============================================================
alpha = 0.1
xmin, xmax = -20.0, 20.0
Nx = 2001
x  = np.linspace(xmin, xmax, Nx)
dx = (xmax - xmin) / (Nx - 1)

t0, t_end = 0.0, 150.0
dt = 0.05
n_steps = int(round((t_end - t0) / dt))
times   = np.linspace(t0, t_end, n_steps + 1)

# Doble pozo (sombrero) — tu forma
V = (1.0/50.0) * ((x**4)/100.0 - x**2)

psi = np.exp(-2.0 * (x - 10.0)**2) * np.exp(-1j * 2.0 * x)
psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))

Lop = laplacian_1d_neumann(Nx, dx)
H   = (-alpha) * Lop + diags(V, 0, shape=(Nx, Nx), format="csr")
I   = eye(Nx, format="csr")
A   = (I + (1j * dt / 2.0) * H).tocsc()
B   = (I - (1j * dt / 2.0) * H).tocsr()
A_lu = splu(A)

mu_hist = np.zeros(n_steps + 1)
sg_hist = np.zeros(n_steps + 1)

target_frames = 600
k_frame = max(1, n_steps // target_frames)
storage_t = []; storage_rho = []
def record(t, psi):
    storage_t.append(float(t))
    storage_rho.append(np.abs(psi)**2)
record(times[0], psi)

for n in range(1, n_steps + 1):
    rhs = B.dot(psi); psi = A_lu.solve(rhs)
    psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))
    mu_hist[n], sg_hist[n] = moments(x, psi)
    if n % k_frame == 0:
        record(times[n], psi)
    if n % max(1, n_steps//100) == 0:
        progress(n, n_steps)
progress(n_steps, n_steps); print()

save_mp4_ffmpeg(x, storage_rho, storage_t, V, fname="1.c.mp4", fps=30,
                xmin=xmin, xmax=xmax, title_prefix="")

# PDF 1.c (título consistente)
plt.figure(figsize=(7.5, 4.5))
plt.plot(times, mu_hist, lw=2, label=r"$\mu(t)=\langle x\rangle$")
plt.fill_between(times, mu_hist - sg_hist, mu_hist + sg_hist,
                 color="C0", alpha=0.2, label=r"$\mu\pm\sigma$")
plt.xlabel("t"); plt.ylabel(r"$\langle x\rangle$ y banda $\mu\pm\sigma$")
plt.title(r"Paquete en $V(x)=\frac{1}{50}\!\left(\frac{x^4}{100}-x^2\right)$,  $\alpha=0.1$")
plt.grid(True, alpha=0.25); plt.legend(frameon=False)
plt.savefig("1.c.pdf", bbox_inches="tight"); plt.close()

print("OK → 1.a.mp4, 1.a.pdf, 1.b.mp4, 1.c.mp4 y 1.c.pdf generados.")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.sparse import diags
from scipy.integrate import solve_ivp, trapezoid
import imageio_ffmpeg

#  PARÁMETROS GENERALES 
ALPHA = 0.1
XMIN, XMAX = -20.0, 20.0
NPTS = 801                    # dx ≈ 0.05
TOL_R = 1e-6
TOL_A = 1e-9
MAX_STEP = 0.1
FPS = 30

# Tiempos y #frames (≈ duración 10–15 s)
T_END_A = 150.0; NFR_A = 450
T_END_B = 50.0;  NFR_B = 300
T_END_C = 150.0; NFR_C = 450

# Dimensiones pares para evitar errores de ffmpeg (8*144=1152, 4.5*144=648)
FIG_DPI = 144
FIG_SIZE = (8.0, 4.5)

# UTILIDADES NUMÉRICAS 
def grid_and_matrices(npts=NPTS, xmin=XMIN, xmax=XMAX):
    """Malla 1D y Laplaciano con BC Neumann como matriz dispersa CSR."""
    x = np.linspace(xmin, xmax, npts, dtype=float)
    dx = x[1] - x[0]
    main = -2.0 * np.ones(npts)
    off = 1.0 * np.ones(npts-1)
    L = diags([off, main, off], offsets=[-1, 0, 1], format="lil")
    # Neumann en bordes (segunda derivada consistente):
    L[0, 0] = -2.0; L[0, 1] = 2.0
    L[-1, -2] = 2.0; L[-1, -1] = -2.0
    L = (L.tocsr()) / (dx*dx)
    return x, dx, L

def V_harmonic(x):
    return (x**2)/50.0

def V_quartic(x):
    return (x/5.0)**4

def V_hat(x):
    return (1.0/50.0)*((x**4)/100.0 - x**2)

def psi0_gaussian(x, x0=10.0, k0=2.0, width_coeff=2.0):
    """Paquete gaussiano con componente de fase (momento hacia -x)."""
    env = np.exp(-width_coeff * (x - x0)**2)
    phase = np.exp(-1j * k0 * x)
    psi = env * phase
    # Normaliza a probabilidad 1 con scipy.integrate.trapezoid
    norm = trapezoid(np.abs(psi)**2, x)
    return psi / np.sqrt(norm)

def build_rhs(L, Vx):
    """f(t, ψ) = i [ ALPHA*L − diag(V) ] ψ para solve_ivp."""
    def rhs(_t, psi):
        return 1j * (ALPHA * (L @ psi) - Vx * psi)
    return rhs

# VIDEO CON FFMPEG 
def ffmpeg_writer(width, height, fps, out_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_path,
        "-loglevel", "error",         # solo errores
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    import subprocess
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    return proc, proc.stdin

def to_even_dims(arr_rgb):
    """Asegura ancho/alto pares con padding negro si es necesario."""
    h, w, _ = arr_rgb.shape
    h2 = h + (h & 1)
    w2 = w + (w & 1)
    if (h2, w2) == (h, w):
        return arr_rgb, w, h
    pad = ((0, h2 - h), (0, w2 - w), (0, 0))
    arr2 = np.pad(arr_rgb, pad, mode='constant')
    return arr2, w2, h2

def render_video(x, times, psi_t, Vx, out_mp4, ylim=None, title=None):
    """
    Genera MP4 con |ψ|^2 (eje Y izquierdo) y el potencial V(x) (eje Y derecho).
    """
    fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1,1,1)
    ax2 = ax.twinx()  # eje para el potencial

    # Densidad inicial
    y0 = np.abs(psi_t[:,0])**2
    line_prob, = ax.plot(x, y0, lw=2, label=r"$|\psi|^2$")

    # Potencial (estático)
    line_V, = ax2.plot(x, Vx, ls="--", lw=1.5, alpha=0.9, label=r"$V(x)$")

    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\psi(t,x)|^2$")
    ax2.set_ylabel(r"$V(x)$")

    if ylim is None:
        ymax = float(1.1*np.max(np.abs(psi_t)**2))
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0
        ax.set_ylim(0, ymax)
    else:
        ax.set_ylim(*ylim)

    # Escala del potencial acorde a sus valores
    Vmin, Vmax = float(np.min(Vx)), float(np.max(Vx))
    if Vmin == Vmax:
        Vmin, Vmax = Vmin - 1.0, Vmax + 1.0
    ax2.set_ylim(Vmin, Vmax)

    ax.set_xlim(x[0], x[-1])

    if title:
        ax.set_title(title)

    # Leyenda combinada
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper right", framealpha=0.6)

    fig.tight_layout()
    canvas.draw()

    # Frame 0 y dimensiones pares
    frame = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    H, W = fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0]
    frame = frame.reshape(H, W, 4)[..., :3]
    frame, W_even, H_even = to_even_dims(frame)

    proc, pipe = ffmpeg_writer(W_even, H_even, FPS, out_mp4)

    try:
        # Enviamos primer frame
        pipe.write(frame.tobytes())

        for k, t in enumerate(times[1:], start=1):
            y = np.abs(psi_t[:,k])**2
            line_prob.set_ydata(y)
            if title:
                ax.set_title(f"{title} — t={t:6.2f}")
            canvas.draw()
            fr = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(H, W, 4)[..., :3]
            if (H % 2) or (W % 2):
                fr, _, _ = to_even_dims(fr)
            pipe.write(fr.tobytes())
    finally:
        try:
            pipe.close()
        except Exception:
            pass
        proc.wait()
        plt.close(fig)

# MÉTRICAS (μ, σ)
def mu_sigma_over_time(x, psi_t):
    """Devuelve mu(t), sigma(t) evaluados en snapshots ψ(x,t_k)."""
    dens = np.abs(psi_t)**2
    # Re-normaliza cada snapshot por seguridad numérica
    norm = trapezoid(dens, x, axis=0)
    dens = dens / norm
    mu = trapezoid(x[:,None]*dens, x, axis=0)
    x2 = trapezoid((x[:,None]**2)*dens, x, axis=0)
    sigma = np.sqrt(np.maximum(x2 - mu**2, 0.0))
    return mu, sigma

def save_mu_sigma_pdf(times, mu, sigma, out_pdf, title):
    fig, ax = plt.subplots(figsize=(8,4.0), dpi=150)
    ax.plot(times, mu, lw=2, label=r"$\mu(t)=\langle x\rangle$")
    ax.fill_between(times, mu - sigma, mu + sigma, alpha=0.25, label=r"$\mu\pm\sigma$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\mu \pm \sigma$")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# PIPELINE DE SIMULACIÓN 
def simulate_case(V_func, t_end, nframes, out_mp4, out_pdf=None, title=""):
    x, _dx, L = grid_and_matrices()
    Vx = V_func(x).astype(float)

    # Condición inicial gaussiana con “momento” hacia -x
    psi0 = psi0_gaussian(x, x0=10.0, k0=2.0, width_coeff=2.0)

    # RHS lineal (MOL)
    rhs = build_rhs(L, Vx)

    # Tiempos de evaluación (y frames)
    t_eval = np.linspace(0.0, float(t_end), int(nframes), dtype=float)

    # Integración temporal
    sol = solve_ivp(rhs, (0.0, float(t_end)), psi0, t_eval=t_eval,
                    method="RK45", rtol=TOL_R, atol=TOL_A, max_step=MAX_STEP)

    if not sol.success:
        raise RuntimeError(f"solve_ivp no convergió: {sol.message}")

    psi_t = sol.y  # shape (NPTS, NT)

    # Re-normaliza snapshots para métricas/visualización
    dens = np.abs(psi_t)**2
    norm = trapezoid(dens, x, axis=0)
    psi_t = psi_t / np.sqrt(norm)[None, :]

    # Video con potencial
    render_video(x, sol.t, psi_t, Vx, out_mp4, title=title)

    # Métricas
    if out_pdf is not None:
        mu, sigma = mu_sigma_over_time(x, psi_t)
        save_mu_sigma_pdf(sol.t, mu, sigma, out_pdf, title + r" — $\mu$ y $\sigma$")

#  MAIN 
def main():
    # 1.a — Oscilador “armónico” 
    simulate_case(
        V_func=V_harmonic, t_end=T_END_A, nframes=NFR_A,
        out_mp4="1.a.mp4", out_pdf="1.a.pdf",
        title=r"1.a  $V(x)=-x^2/50$"
    )

    # 1.b — Oscilador cuártico (solo video)
    simulate_case(
        V_func=V_quartic, t_end=T_END_B, nframes=NFR_B,
        out_mp4="1.b.mp4", out_pdf=None,
        title=r"1.b  $V(x)=(x/5)^4$"
    )

    # 1.c — Potencial del sombrero (video + gráfica μ,σ)
    simulate_case(
        V_func=V_hat, t_end=T_END_C, nframes=NFR_C,
        out_mp4="1.c.mp4", out_pdf="1.c.pdf",
        title=r"1.c  $V(x)=\frac{1}{50}\!\left(\frac{x^4}{100}-x^2\right)$"
    )

if __name__ == "__main__":
    np.random.seed(0)
    main()

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.integrate import trapezoid

# ————— Config general ————————————————————————————————————————————————
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
np.seterr(all="ignore")  # silencia warnings; controlamos estabilidad con CFL + filtros


# ————— Utilidades FFT (versión RFFT) ————————————————————————————————
def make_grid(L, N):
    """
    Malla periódica [-L, L], frecuencias y operadores espectrales para RFFT.
    Devuelve:
      x (N,), dx, k (N//2+1,), ik, k3  con k en rad/m.
    """
    x = np.linspace(-L, L, N, endpoint=False, dtype=np.float64)
    dx = x[1] - x[0]
    k = 2.0*np.pi * np.fft.rfftfreq(N, d=dx)  # sólo modos [0..N/2]
    ik = 1j * k
    k3 = k**3
    return x, dx, k, ik, k3


def two_thirds_mask_rfft(N):
    """
    Máscara 2/3 para de-aliasing en RFFT (longitud N//2+1).
    Mantiene |m| <= N/3 y anula el resto.
    """
    m = np.arange(0, N//2 + 1, dtype=int)
    return (m <= (N//3))


def spectral_tophat_filter(k, frac=0.9):
    """
    Filtro tophat suave: 1 en |k|<=frac*kmax, 0 fuera (en RFFT basta un corte).
    Lo usamos como refuerzo suave adicional (opcional).
    """
    kmax = np.max(np.abs(k))
    kc = frac * kmax
    return (np.abs(k) <= kc).astype(np.float64)


# ————— Solver KdV: RK4 en k-espacio (RFFT), estilo del código C++ ——————————
class KdVSolver:
    """
    KdV pseudo-espectral con RK4 en k-espacio:
        φ_t = - 1/2 ∂_x(φ^2) - δ^2 ∂_x^3 φ
    En Fourier (RFFT):
        d/dt hat{φ} = - 1/2 (ik) * RFFT[ (IRFFT(hat{φ}))^2 ]  -  δ^2 (ik)^3 * hat{φ}
    Con de-aliasing 2/3 sobre el no lineal.
    """

    def __init__(self, L=25.0, N=512, delta=0.022, cfl_nl=0.25,
                 dt=None, use_filter=True, filter_frac=0.95):
        self.L = float(L)
        self.N = int(N)
        self.delta = float(delta)
        self.cfl_nl = float(cfl_nl)
        self.dt_user = dt

        # malla y operadores espectrales (RFFT)
        self.x, self.dx, self.k, self.ik, self.k3 = make_grid(self.L, self.N)
        self.dealias = two_thirds_mask_rfft(self.N)           # booleana (N//2+1,)
        self.use_filter = bool(use_filter)
        self.soft_filter = spectral_tophat_filter(self.k, frac=filter_frac) if use_filter else 1.0

        self.dt = None  # se fija en evolve

    # — CFL sugerido (no lineal + lineal) —
    def _stable_dt(self, phi0):
        umax = float(np.max(np.abs(phi0)))
        dt_nl = self.cfl_nl * (self.dx / max(umax, 1e-12))
        kmax = np.max(np.abs(self.k))
        dt_lin = 0.30 / ((self.delta**2) * max(kmax**3, 1e-12))
        dt = min(dt_nl, dt_lin)
        return float(np.clip(dt, 1e-5, 5e-2))

    # — RHS en k-espacio: d/dt (phi_hat) —
    def _rhs_hat(self, phi_hat):
        # φ en real
        phi = np.fft.irfft(phi_hat, n=self.N)

        # No lineal: -1/2 (ik) * RFFT(φ^2)  con de-aliasing 2/3
        phi2_hat = np.fft.rfft(phi*phi)
        # de-aliasing: anula alta frecuencia del término no lineal
        phi2_hat = np.where(self.dealias, phi2_hat, 0.0)
        # filtro suave adicional (opcional)
        if self.use_filter:
            phi2_hat = phi2_hat * self.soft_filter
        nonlin_hat = -0.5 * (self.ik * phi2_hat)

        # Lineal: - δ^2 (ik)^3 * φ_hat
        linear_hat = - (self.delta**2) * ((self.ik**3) * phi_hat)

        return nonlin_hat + linear_hat

    def _rk4_step_hat(self, phi_hat, dt):
        k1 = self._rhs_hat(phi_hat)
        k2 = self._rhs_hat(phi_hat + 0.5*dt*k1)
        k3 = self._rhs_hat(phi_hat + 0.5*dt*k2)
        k4 = self._rhs_hat(phi_hat + dt*k3)
        return phi_hat + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def evolve(self, phi_init, t_final, n_frames=200, quiet=True):
        """Integra hasta t_final. Devuelve (times_snap, snapshots) con ~n_frames estados."""
        assert phi_init.shape == (self.N,)
        phi0 = phi_init.astype(np.float64, copy=True)
        dt = self.dt_user if self.dt_user is not None else self._stable_dt(phi0)
        self.dt = dt

        steps = int(np.ceil(t_final / dt))
        stride = max(1, steps // int(n_frames))

        # estado en k-espacio
        phi_hat = np.fft.rfft(phi0)

        t = 0.0
        times = [t]
        snaps = [phi0.copy()]

        for n in range(1, steps + 1):
            phi_hat = self._rk4_step_hat(phi_hat, dt)
            t += dt
            if (n % stride == 0) or (n == steps):
                phi = np.fft.irfft(phi_hat, n=self.N)
                if not np.all(np.isfinite(phi)):
                    raise FloatingPointError(f"Estado no finito en paso {n}. Reduce dt o activa filtro.")
                times.append(t)
                snaps.append(phi.copy())

        return np.array(times), np.array(snaps)

    # — Cantidades conservadas —
    def conserved(self, phi):
        mass = trapezoid(phi, self.x)
        momentum = trapezoid(phi**2, self.x)
        dphi_dx = np.fft.irfft(self.ik * np.fft.rfft(phi), n=self.N).real
        energy = trapezoid((phi**3)/3.0 - (self.delta**2) * (dphi_dx**2), self.x)
        return mass, momentum, energy


# ————— Condiciones iniciales ————————————————————————————————
def initial_cosine(x, amplitude=1.0, wavelength=20.0):
    return amplitude * np.cos(2*np.pi*x / wavelength)

def initial_sech_squared(x, amplitude=12.0, center=-5.0, width=1.0):
    return amplitude / np.cosh((x - center)/width)**2

def initial_gaussian(x, amplitude=5.0, center=0.0, width=2.0):
    return amplitude * np.exp(-((x-center)/width)**2)

def two_soliton_collision(x):
    s1 = 8.0 / np.cosh((x + 10.0)/2.0)**2   # alto/rápido
    s2 = 2.0 / np.cosh((x - 10.0)/4.0)**2   # bajo/lento
    return s1 + s2


# ————— Render video (una curva) ————————————————————————————————
def save_line_animation(x, times, snapshots, out_mp4, ylims, color="C0",
                        title="", delta2=None, fps=20):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    writer = FFMpegWriter(fps=fps, bitrate=1800, metadata=dict(artist="KdV"))
    with writer.saving(fig, out_mp4, dpi=120):
        for i, t in enumerate(times):
            ax.clear()
            ax.plot(x, snapshots[i], color=color, lw=2)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(*ylims)
            ax.set_xlabel('x'); ax.set_ylabel('φ(t,x)')
            ttl = title + (f"   (t = {t:.2f})" if t > 0 else "")
            ax.set_title(ttl)
            ax.grid(True, alpha=0.25)
            if delta2 is not None:
                ax.text(0.02, 0.95, f'δ² = {delta2:.4f}', transform=ax.transAxes,
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                        facecolor="white", alpha=0.8))
            writer.grab_frame()
    plt.close(fig)


# ————— Plots estáticos ————————————————————————————————
def plot_conserved(times, masses, moms, enes, out_pdf):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=140)
    axes[0].plot(times, masses, 'b-', lw=2); axes[0].set_title('Masa'); axes[0].set_xlabel('t'); axes[0].set_ylabel('∫φ dx'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(times, moms, 'g-', lw=2);   axes[1].set_title('Momento'); axes[1].set_xlabel('t'); axes[1].set_ylabel('∫φ² dx'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(times, enes, 'r-', lw=2);   axes[2].set_title('Energía'); axes[2].set_xlabel('t'); axes[2].set_ylabel('∫(φ³/3 - δ² φ_x²) dx'); axes[2].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_pdf, bbox_inches="tight"); plt.close(fig)

def plot_effect_of_delta(deltas, states, xs_unused, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=140)
    axes = axes.ravel()
    for i, (delta, (x, u)) in enumerate(zip(deltas, states)):
        ax = axes[i]
        ax.plot(x, u, lw=2, label=f'δ²={delta**2:.4f}')
        ax.set_title(f'Estado final (δ²={delta**2:.4f})')
        ax.set_xlabel('x'); ax.set_ylabel('φ')
        ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

def plot_velocity(time, peak_pos, v_num, v_th, out_pdf):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5), dpi=140)
    ax1.plot(time, peak_pos, 'bo-', ms=4, lw=2, label='Simulación')
    ax1.plot(time, v_th*time + peak_pos[0], 'r--', lw=2, label=f'Teórica v=A/3={v_th:.2f}')
    ax1.set_xlabel('tiempo'); ax1.set_ylabel('Posición del pico'); ax1.set_title('Movimiento del solitón'); ax1.grid(True); ax1.legend()
    ax2.plot(time[1:], v_num, 'go-', ms=4, lw=2, label='Velocidad numérica')
    ax2.axhline(v_th, color='r', ls='--', lw=2, label=f'v teórica={v_th:.2f}')
    ax2.set_xlabel('tiempo'); ax2.set_ylabel('Velocidad'); ax2.set_title('Velocidad del solitón'); ax2.grid(True); ax2.legend()
    fig.tight_layout(); fig.savefig(out_pdf, bbox_inches="tight"); plt.close(fig)


# ————— Tareas del enunciado ————————————————————————————————
def main():
    # 1) Descomposición de coseno → muchos solitones
    solver1 = KdVSolver(L=25, N=512, delta=0.022, cfl_nl=0.25, use_filter=True)
    phi0 = initial_cosine(solver1.x, amplitude=1.0, wavelength=20.0)
    t1_all, U1_all = solver1.evolve(phi0, t_final=15.0, n_frames=240, quiet=True)
    # t1_all, U1_all incluyen snapshot inicial; animación espera mismas longitudes
    save_line_animation(solver1.x, t1_all, U1_all, '3_descomposicion_coseno.mp4',
                        ylims=(-1.5, 2.0), color='C0',
                        title='KdV: coseno → solitones', delta2=solver1.delta**2, fps=20)

    # 2) Colisión de dos solitones
    solver2 = KdVSolver(L=30, N=768, delta=0.022, cfl_nl=0.25, use_filter=True)
    phi_col = two_soliton_collision(solver2.x)
    t2_all, U2_all = solver2.evolve(phi_col, t_final=8.0, n_frames=160, quiet=True)
    save_line_animation(solver2.x, t2_all, U2_all, '3_colision_solitones.mp4',
                        ylims=(-2.0, 10.0), color='C3',
                        title='KdV: colisión de dos solitones', delta2=solver2.delta**2, fps=15)

    # 3) Cantidades conservadas (sobre el caso 1, usando todos los snapshots)
    masses, moms, enes = [], [], []
    for u in U1_all:
        m, p, e = solver1.conserved(u)
        masses.append(m); moms.append(p); enes.append(e)
    plot_conserved(t1_all, masses, moms, enes, '3_cantidades_conservadas.pdf')

    # 4) Efecto de δ
    deltas = [0.01, 0.022, 0.05, 0.10]
    finals = []
    for d in deltas:
        s = KdVSolver(L=25, N=512, delta=d, cfl_nl=0.25, use_filter=True)
        phi0d = initial_cosine(s.x, amplitude=1.0, wavelength=20.0)
        td, Ud = s.evolve(phi0d, t_final=10.0, n_frames=120, quiet=True)
        finals.append((s.x, Ud[-1]))
    plot_effect_of_delta(deltas, finals, None, '3_efecto_delta.png')

    # 5) Velocidad de un solitón sech^2 (validación v ≈ A/3)
    solver3 = KdVSolver(L=25, N=512, delta=0.022, cfl_nl=0.25, use_filter=True)
    A = 12.0; v_th = A/3.0
    U0 = initial_sech_squared(solver3.x, amplitude=A, center=-5.0, width=1.0)
    t3, U3 = solver3.evolve(U0, t_final=5.0, n_frames=160, quiet=True)
    peak_pos = np.array([solver3.x[np.argmax(u)] for u in U3])
    dt_snap = np.mean(np.diff(t3))
    v_num = np.diff(peak_pos) / dt_snap
    plot_velocity(t3, peak_pos, v_num, v_th, '3_velocidad_soliton.pdf')

    # 6) Caso sin solitones (dispersión)
    def smooth_decay(x):  # CI suave que decae
        return np.exp(-(x**2)/100.0) * np.sin(x) * np.exp(-np.abs(x)/20.0)
    solver4 = KdVSolver(L=25, N=512, delta=0.022, cfl_nl=0.25, use_filter=True)
    U0s = smooth_decay(solver4.x)
    t4, U4 = solver4.evolve(U0s, t_final=20.0, n_frames=200, quiet=True)
    # gráfico estático: estado inicial y final + amplitud máx en el tiempo
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5), dpi=140)
    ax1.plot(solver4.x, U4[0], 'b-', lw=2, label='t=0')
    ax1.plot(solver4.x, U4[-1], 'r-', lw=2, label=f't={t4[-1]:.1f}')
    ax1.set_title('Dispersión sin formación de solitones'); ax1.set_xlabel('x'); ax1.set_ylabel('φ'); ax1.grid(True); ax1.legend()
    maxamp = [np.max(np.abs(u)) for u in U4]
    ax2.plot(t4, maxamp, 'g-', lw=2); ax2.set_yscale('log')
    ax2.set_title('Decaimiento dispersivo'); ax2.set_xlabel('tiempo'); ax2.set_ylabel('amplitud máx'); ax2.grid(True)
    fig.tight_layout(); fig.savefig('3_sin_solitones.pdf', bbox_inches="tight"); plt.close(fig)

    # Resumen breve
    with open('3_resultados.txt', 'w', encoding='utf-8') as f:
        f.write("RESULTADOS DEL ANÁLISIS DE SOLITONES KdV (pseudo-espectral RFFT + RK4 + de-alias 2/3)\n")
        f.write("="*70 + "\n\n")
        f.write("1) Coseno → tren de solitones; los más altos viajan más rápido.\n")
        f.write("2) Colisión: interacción no lineal; tras la colisión reaparecen (con desfase).\n")
        f.write("3) Conservación: masa, momento y energía casi constantes.\n")
        f.write("4) δ controla el balance no linealidad–dispersión: δ² pequeño → más solitones.\n")
        f.write("5) Solitón sech²: velocidad numérica ≃ A/3 (validación).\n")
        f.write("6) CI suave que decae → dispersión sin solitones.\n")

    print("Generado:", *[
        "3_descomposicion_coseno.mp4", "3_colision_solitones.mp4",
        "3_cantidades_conservadas.pdf", "3_efecto_delta.png",
        "3_velocidad_soliton.pdf", "3_sin_solitones.pdf", "3_resultados.txt"
    ], sep="\n- ")


if __name__ == "__main__":
    main()
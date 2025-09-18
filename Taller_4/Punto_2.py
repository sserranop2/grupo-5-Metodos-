<<<<<<< HEAD
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
=======
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import laplace
import os

class TuringPatternSimulator:
    def __init__(self, nx=256, ny=256, dx=1.0, dy=1.0):
        """
        Inicializar el simulador de patrones de Turing
        """
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.x = np.linspace(0, nx*dx, nx)
        self.y = np.linspace(0, ny*dy, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def laplacian_2d(self, field):
        """Calcular el laplaciano en 2D usando diferencias finitas"""
        # Usando scipy.ndimage.laplace que maneja bien las condiciones de frontera
        return laplace(field) / (self.dx * self.dy)
    
    def reaction_diffusion_step(self, u, v, dt, alpha, beta, F_func, G_func):
        """Un paso de la simulación de reacción-difusión"""
        # Términos de difusión
        diff_u = alpha * self.laplacian_2d(u)
        diff_v = beta * self.laplacian_2d(v)
        
        # Términos de reacción
        F_uv = F_func(u, v)
        G_uv = G_func(u, v)
        
        # Actualización usando Euler hacia adelante
        u_new = u + dt * (diff_u + F_uv)
        v_new = v + dt * (diff_v + G_uv)
        
        return u_new, v_new
    
    def simulate(self, u0, v0, t_final, dt, alpha, beta, F_func, G_func, 
                 save_interval=100):
        """Ejecutar la simulación completa"""
        steps = int(t_final / dt)
        u, v = u0.copy(), v0.copy()
        
        # Almacenar algunos frames para animación si es necesario
        u_history = []
        v_history = []
        times = []
        
        print(f"Simulando {steps} pasos temporales...")
        
        for step in range(steps):
            u, v = self.reaction_diffusion_step(u, v, dt, alpha, beta, F_func, G_func)
            
            # Guardar cada save_interval pasos
            if step % save_interval == 0:
                u_history.append(u.copy())
                v_history.append(v.copy())
                times.append(step * dt)
                
            # Progreso cada 10% de la simulación
            if step % (steps // 10) == 0:
                print(f"Progreso: {100*step/steps:.1f}%")
        
        return u, v, u_history, v_history, times

def create_initial_conditions(nx, ny, noise_level=0.1, seed=42):
    """Crear condiciones iniciales con ruido gaussiano"""
    np.random.seed(seed)
    
    # Estado base con pequeñas perturbaciones aleatorias
    u0 = np.ones((ny, nx)) + noise_level * np.random.randn(ny, nx)
    v0 = np.ones((ny, nx)) + noise_level * np.random.randn(ny, nx)
    
    return u0, v0

def create_localized_initial_conditions(nx, ny, num_seeds=5):
    """Crear condiciones iniciales con perturbaciones localizadas"""
    u0 = np.ones((ny, nx))
    v0 = np.ones((ny, nx))
    
    np.random.seed(42)
    for _ in range(num_seeds):
        # Posición aleatoria
        cx = np.random.randint(nx//4, 3*nx//4)
        cy = np.random.randint(ny//4, 3*ny//4)
        
        # Crear perturbación gaussiana
        Y, X = np.ogrid[:ny, :nx]
        mask = (X - cx)**2 + (Y - cy)**2 <= 20**2
        u0[mask] += 0.5 * np.exp(-((X[mask] - cx)**2 + (Y[mask] - cy)**2) / 100)
        v0[mask] += 0.3 * np.exp(-((X[mask] - cx)**2 + (Y[mask] - cy)**2) / 100)
    
    return u0, v0

# Definir diferentes sistemas de reacción
def classic_turing_system():
    """Sistema clásico de Turing del taller"""
    def F(u, v):
        return u - u**3 - v - 0.05
    
    def G(u, v):
        return 10*(u - v)
    
    return F, G, {"alpha": 0.00028, "beta": 0.05, "name": "Classic_Turing"}

def fitzHugh_nagumo_system():
    """Sistema FitzHugh-Nagumo modificado"""
    a, b, tau = 0.1, 0.1, 10
    
    def F(u, v):
        return u - u**3/3 - v
    
    def G(u, v):
        return (u + a - b*v) / tau
    
    return F, G, {"alpha": 0.001, "beta": 0.01, "name": "FitzHugh_Nagumo"}

def brusselator_system():
    """Sistema Brusselator"""
    A, B = 1.0, 3.0
    
    def F(u, v):
        return A + u**2 * v - B*u - u
    
    def G(u, v):
        return B*u - u**2 * v
    
    return F, G, {"alpha": 0.002, "beta": 0.008, "name": "Brusselator"}

def gray_scott_system():
    """Sistema Gray-Scott"""
    f, k = 0.04, 0.06
    
    def F(u, v):
        return -u*v**2 + f*(1-u)
    
    def G(u, v):
        return u*v**2 - (f+k)*v
    
    return F, G, {"alpha": 0.002, "beta": 0.001, "name": "Gray_Scott"}

def schnakenberg_system():
    """Sistema Schnakenberg"""
    a, b = 0.1, 0.9
    
    def F(u, v):
        return a - u + u**2 * v
    
    def G(u, v):
        return b - u**2 * v
    
    return F, G, {"alpha": 0.005, "beta": 0.1, "name": "Schnakenberg"}

def gierer_meinhardt_system():
    """Sistema Gierer-Meinhardt simplificado"""
    a, b, c = 0.01, 0.1, 0.1
    
    def F(u, v):
        return a*u**2/v - b*u + c
    
    def G(u, v):
        return u**2 - v
    
    return F, G, {"alpha": 0.001, "beta": 0.02, "name": "Gierer_Meinhardt"}

def plot_final_pattern(u_final, v_final, title, filename, params=None):
    """Graficar el patrón final"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Campo u
    im1 = axes[0].imshow(u_final, cmap='viridis', origin='lower')
    axes[0].set_title(f'{title} - Campo u')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # Campo v
    im2 = axes[1].imshow(v_final, cmap='plasma', origin='lower')
    axes[1].set_title(f'{title} - Campo v')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    # Añadir información de parámetros
    if params:
        info_text = f"α={params.get('alpha', 'N/A')}, β={params.get('beta', 'N/A')}"
        fig.suptitle(f"{title}\n{info_text}", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def create_pattern_animation(u_history, v_history, times, title, filename):
    """Crear animación del desarrollo del patrón"""
    if len(u_history) < 10:  # No crear animación si hay muy pocos frames
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Configurar los límites de color
    u_min, u_max = np.min(u_history), np.max(u_history)
    v_min, v_max = np.min(v_history), np.max(v_history)
    
    im1 = axes[0].imshow(u_history[0], cmap='viridis', origin='lower', 
                         vmin=u_min, vmax=u_max)
    axes[0].set_title('Campo u')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(v_history[0], cmap='plasma', origin='lower',
                         vmin=v_min, vmax=v_max)
    axes[1].set_title('Campo v')
    plt.colorbar(im2, ax=axes[1])
    
    time_text = fig.suptitle(f'{title} - t = {times[0]:.2f}')
    
    def animate(frame):
        im1.set_array(u_history[frame])
        im2.set_array(v_history[frame])
        time_text.set_text(f'{title} - t = {times[frame]:.2f}')
        return [im1, im2, time_text]
    
    # Reducir frames para animación más manejable
    frame_skip = max(1, len(u_history) // 50)
    frames_to_use = range(0, len(u_history), frame_skip)
    
    anim = FuncAnimation(fig, animate, frames=frames_to_use, 
                        interval=100, blit=False, repeat=False)
    
    try:
        Writer = plt.matplotlib.animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(filename, writer=writer)
    except:
        print(f"No se pudo guardar la animación {filename}")
    
    plt.close()

def run_simulation(system_func, initial_func, title, grid_size=256, t_final=15, dt=0.001):
    """Ejecutar una simulación completa"""
    print(f"\n=== Simulando {title} ===")
    
    # Obtener sistema de reacción
    F, G, params = system_func()
    
    # Crear simulador
    simulator = TuringPatternSimulator(grid_size, grid_size, dx=3.0/grid_size, dy=3.0/grid_size)
    
    # Condiciones iniciales
    u0, v0 = initial_func(grid_size, grid_size)
    
    # Verificar estabilidad (condición CFL aproximada)
    max_diff = max(params["alpha"], params["beta"])
    dt_max = 0.25 * min(simulator.dx**2, simulator.dy**2) / max_diff
    if dt > dt_max:
        dt = dt_max * 0.8
        print(f"Ajustando dt a {dt:.6f} para estabilidad")
    
    # Ejecutar simulación
    u_final, v_final, u_history, v_history, times = simulator.simulate(
        u0, v0, t_final, dt, params["alpha"], params["beta"], F, G
    )
    
    # Guardar resultados
    filename_base = f"2_{params['name']}"
    
    # Imagen final
    plot_final_pattern(u_final, v_final, title, f"{filename_base}.png", params)
    
    # Animación (opcional, solo para algunos casos)
    if len(u_history) > 10:
        try:
            create_pattern_animation(u_history, v_history, times, title, f"{filename_base}.mp4")
        except:
            print(f"No se pudo crear animación para {title}")
    
    print(f"Simulación {title} completada")
    return u_final, v_final

def main():
    """Función principal - ejecutar todas las simulaciones"""
    print("Iniciando simulaciones de patrones de Turing")
    
    # Lista de sistemas a simular
    simulations = [
        (classic_turing_system, create_initial_conditions, "Rayas_Clasicas", 256, 15),
        (fitzHugh_nagumo_system, create_initial_conditions, "Ondas_FitzHugh", 256, 20),
        (brusselator_system, create_initial_conditions, "Puntos_Brusselator", 256, 10),
        (gray_scott_system, create_initial_conditions, "Coral_GrayScott", 256, 25),
        (schnakenberg_system, create_initial_conditions, "Hexagonos_Schnakenberg", 256, 30),
        (gierer_meinhardt_system, create_localized_initial_conditions, "Activador_Inhibidor", 256, 40),
    ]
    
    # Ejecutar simulaciones
    for system_func, initial_func, name, size, t_max in simulations:
        try:
            run_simulation(system_func, initial_func, name, size, t_max)
        except Exception as e:
            print(f"Error en simulación {name}: {e}")
            continue
    
    # Crear documento de texto con explicaciones
    create_results_documentation()
    
    print("\nTodas las simulaciones de patrones de Turing completadas")

def create_results_documentation():
    """Crear documentación de los resultados"""
    documentation = """
PATRONES DE TURING - RESULTADOS

1. Rayas_Clasicas (Sistema original del taller):
   - F(u,v) = u - u³ - v - 0.05, G(u,v) = 10(u - v)
   - α=0.00028, β=0.05
   - Produce rayas y patrones similares a piel de animales

2. Ondas_FitzHugh (FitzHugh-Nagumo):
   - Sistema de excitación-inhibición
   - α=0.001, β=0.01
   - Genera ondas viajeras y patrones espirales

3. Puntos_Brusselator:
   - A=1.0, B=3.0, F(u,v) = A + u²v - Bu - u
   - α=0.002, β=0.008
   - Crea patrones de puntos y estructuras hexagonales

4. Coral_GrayScott:
   - f=0.04, k=0.06, sistema feed-kill
   - α=0.002, β=0.001
   - Simula crecimiento tipo coral o bacterias

5. Hexagonos_Schnakenberg:
   - a=0.1, b=0.9, F(u,v) = a - u + u²v
   - α=0.005, β=0.1
   - Produce patrones hexagonales regulares

6. Activador_Inhibidor (Gierer-Meinhardt):
   - Sistema activador-inhibidor clásico
   - α=0.001, β=0.02
   - Genera patrones de puntos con inhibición lateral
"""
    
    with open("2_resultados.txt", "w") as f:
        f.write(documentation)

if __name__ == "__main__":
    main()
>>>>>>> 75eea000e5c4c2387babcf12c3a9bf383bb658ef

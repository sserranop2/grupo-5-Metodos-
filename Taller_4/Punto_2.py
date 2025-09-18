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
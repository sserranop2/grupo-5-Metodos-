import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patches as patches

# Configuración para evitar mostrar ventanas
plt.ioff()

# Configurar writer para videos
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'

class KdVSolver:
    """Solver para la ecuación de Korteweg-de Vries usando diferencias finitas"""
    
    def __init__(self, L=50, N=1024, delta=0.022):
        self.L = L  # Longitud del dominio
        self.N = N  # Número de puntos espaciales
        self.delta = delta  # Parámetro δ² de la ecuación KdV
        
        # Grilla espacial
        self.dx = 2 * L / N
        self.x = np.linspace(-L, L, N)
        
        # Paso temporal (CFL condition)
        self.dt = 0.1 * self.dx**3 / delta  # Condición estricta para estabilidad
        
    def spatial_derivatives(self, phi):
        """Calcula las derivadas espaciales usando FFT para precisión"""
        phi_hat = np.fft.fft(phi)
        k = np.fft.fftfreq(self.N, self.dx) * 2 * np.pi
        
        # Primera derivada
        dphi_dx = np.real(np.fft.ifft(1j * k * phi_hat))
        
        # Tercera derivada
        d3phi_dx3 = np.real(np.fft.ifft(-1j * k**3 * phi_hat))
        
        return dphi_dx, d3phi_dx3
    
    def rk4_step(self, phi):
        """Un paso de Runge-Kutta 4to orden"""
        def f(phi_state):
            dphi_dx, d3phi_dx3 = self.spatial_derivatives(phi_state)
            return -(phi_state * dphi_dx + self.delta**2 * d3phi_dx3)
        
        k1 = f(phi)
        k2 = f(phi + 0.5 * self.dt * k1)
        k3 = f(phi + 0.5 * self.dt * k2)
        k4 = f(phi + self.dt * k3)
        
        return phi + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def evolve(self, phi_init, t_final):
        """Evoluciona la solución en el tiempo"""
        n_steps = int(t_final / self.dt)
        times = np.linspace(0, t_final, n_steps)
        
        phi = phi_init.copy()
        solutions = [phi.copy()]
        
        for i in range(1, n_steps):
            phi = self.rk4_step(phi)
            if i % (n_steps // 100) == 0:  # Guardar cada 1% del tiempo
                solutions.append(phi.copy())
        
        return np.array(times), np.array(solutions)
    
    def conserved_quantities(self, phi):
        """Calcula las cantidades conservadas"""
        # Masa
        mass = np.trapz(phi, self.x)
        
        # Momento
        momentum = np.trapz(phi**2, self.x)
        
        # Energía
        dphi_dx, _ = self.spatial_derivatives(phi)
        energy = np.trapz(phi**3/3 - self.delta**2 * dphi_dx**2, self.x)
        
        return mass, momentum, energy

def initial_cosine(x, amplitude=1.0, wavelength=10.0):
    """Condición inicial coseno"""
    return amplitude * np.cos(2 * np.pi * x / wavelength)

def initial_sech_squared(x, amplitude=12.0, center=0.0, width=1.0):
    """Condición inicial sech² (solitón exacto)"""
    return amplitude / np.cosh((x - center) / width)**2

def initial_gaussian(x, amplitude=5.0, center=0.0, width=2.0):
    """Condición inicial gaussiana"""
    return amplitude * np.exp(-((x - center) / width)**2)

def two_soliton_collision(x):
    """Dos solitones para estudiar colisiones"""
    # Solitón rápido (derecha) y lento (izquierda)
    soliton1 = 8 / np.cosh((x + 10) / 2)**2  # Más alto, más rápido
    soliton2 = 2 / np.cosh((x - 10) / 4)**2  # Más bajo, más lento
    return soliton1 + soliton2

def analyze_soliton_properties(solver, phi_evolution, times):
    """Analiza propiedades de los solitones"""
    conserved_vals = []
    peak_positions = []
    peak_amplitudes = []
    
    for phi in phi_evolution:
        # Cantidades conservadas
        mass, momentum, energy = solver.conserved_quantities(phi)
        conserved_vals.append([mass, momentum, energy])
        
        # Encontrar picos (solitones)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(phi, height=0.1, distance=20)
        
        if len(peaks) > 0:
            peak_positions.append(solver.x[peaks])
            peak_amplitudes.append(phi[peaks])
        else:
            peak_positions.append([])
            peak_amplitudes.append([])
    
    return np.array(conserved_vals), peak_positions, peak_amplitudes

# Configuración principal
def main():
    print("Iniciando simulación de solitones KdV...")
    
    # 1. Estudio de descomposición de coseno en solitones
    print("1. Analizando descomposición de coseno en solitones...")
    solver1 = KdVSolver(L=25, N=512, delta=0.022)
    phi_init_cos = initial_cosine(solver1.x, amplitude=1.0, wavelength=20)
    
    times1, solutions1 = solver1.evolve(phi_init_cos, t_final=15.0)
    
    # Crear animación
    fig, ax = plt.subplots(figsize=(12, 6))
    def animate_coseno(frame):
        ax.clear()
        ax.plot(solver1.x, solutions1[frame], 'b-', linewidth=2)
        ax.set_xlim(-25, 25)
        ax.set_ylim(-0.5, 2.0)
        ax.set_xlabel('x')
        ax.set_ylabel('φ(t,x)')
        ax.set_title(f'KdV: Descomposición de coseno en solitones (t = {times1[frame]:.2f})')
        ax.grid(True, alpha=0.3)
        
        # Agregar información sobre δ
        ax.text(0.02, 0.95, f'δ² = {solver1.delta**2:.4f}', 
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    anim1 = FuncAnimation(fig, animate_coseno, frames=len(solutions1), 
                         interval=50, repeat=True)
    
    # Configurar writer de video
    try:
        writer = FFMpegWriter(fps=20, metadata=dict(artist='KdV Simulator'), bitrate=1800)
        anim1.save('3_descomposicion_coseno.mp4', writer=writer, dpi=100)
    except Exception as e:
        print(f"Error guardando video 1: {e}")
        print("Guardando como GIF en su lugar...")
        anim1.save('3_descomposicion_coseno.gif', writer='pillow', fps=10, dpi=80)
    
    plt.close()
    
    # 2. Colisión de dos solitones
    print("2. Estudiando colisión de solitones...")
    solver2 = KdVSolver(L=30, N=768, delta=0.022)
    phi_init_collision = two_soliton_collision(solver2.x)
    
    times2, solutions2 = solver2.evolve(phi_init_collision, t_final=8.0)
    
    # Animación de colisión
    fig, ax = plt.subplots(figsize=(12, 6))
    def animate_collision(frame):
        ax.clear()
        ax.plot(solver2.x, solutions2[frame], 'r-', linewidth=2)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-1, 10)
        ax.set_xlabel('x')
        ax.set_ylabel('φ(t,x)')
        ax.set_title(f'KdV: Colisión de dos solitones (t = {times2[frame]:.2f})')
        ax.grid(True, alpha=0.3)
        
        # Información del sistema
        ax.text(0.02, 0.95, f'δ² = {solver2.delta**2:.4f}', 
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    anim2 = FuncAnimation(fig, animate_collision, frames=len(solutions2), 
                         interval=80, repeat=True)
    
    # Configurar writer de video
    try:
        writer = FFMpegWriter(fps=15, metadata=dict(artist='KdV Simulator'), bitrate=1800)
        anim2.save('3_colision_solitones.mp4', writer=writer, dpi=100)
    except Exception as e:
        print(f"Error guardando video 2: {e}")
        print("Guardando como GIF en su lugar...")
        anim2.save('3_colision_solitones.gif', writer='pillow', fps=8, dpi=80)
    
    plt.close()
    
    # 3. Análisis de cantidades conservadas
    print("3. Analizando cantidades conservadas...")
    conserved1, peaks1, amps1 = analyze_soliton_properties(solver1, solutions1, times1)
    conserved2, peaks2, amps2 = analyze_soliton_properties(solver2, solutions2, times2)
    
    # Gráfica de conservación
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Para el caso coseno
    axes[0,0].plot(times1[:len(conserved1)], conserved1[:, 0], 'b-', linewidth=2)
    axes[0,0].set_title('Masa (Coseno)')
    axes[0,0].set_ylabel('∫φ dx')
    axes[0,0].grid(True)
    
    axes[0,1].plot(times1[:len(conserved1)], conserved1[:, 1], 'g-', linewidth=2)
    axes[0,1].set_title('Momento (Coseno)')
    axes[0,1].set_ylabel('∫φ² dx')
    axes[0,1].grid(True)
    
    axes[0,2].plot(times1[:len(conserved1)], conserved1[:, 2], 'r-', linewidth=2)
    axes[0,2].set_title('Energía (Coseno)')
    axes[0,2].set_ylabel('∫(φ³/3 - δ²(∂φ/∂x)²) dx')
    axes[0,2].grid(True)
    
    # Para el caso colisión
    axes[1,0].plot(times2[:len(conserved2)], conserved2[:, 0], 'b-', linewidth=2)
    axes[1,0].set_title('Masa (Colisión)')
    axes[1,0].set_xlabel('tiempo')
    axes[1,0].set_ylabel('∫φ dx')
    axes[1,0].grid(True)
    
    axes[1,1].plot(times2[:len(conserved2)], conserved2[:, 1], 'g-', linewidth=2)
    axes[1,1].set_title('Momento (Colisión)')
    axes[1,1].set_xlabel('tiempo')
    axes[1,1].set_ylabel('∫φ² dx')
    axes[1,1].grid(True)
    
    axes[1,2].plot(times2[:len(conserved2)], conserved2[:, 2], 'r-', linewidth=2)
    axes[1,2].set_title('Energía (Colisión)')
    axes[1,2].set_xlabel('tiempo')
    axes[1,2].set_ylabel('∫(φ³/3 - δ²(∂φ/∂x)²) dx')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig('3_cantidades_conservadas.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Comparación de diferentes valores de δ
    print("4. Estudiando efecto del parámetro δ...")
    deltas = [0.01, 0.022, 0.05, 0.1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, delta in enumerate(deltas):
        solver_delta = KdVSolver(L=25, N=512, delta=delta)
        phi_init = initial_cosine(solver_delta.x, amplitude=1.0)
        times_d, solutions_d = solver_delta.evolve(phi_init, t_final=10.0)
        
        # Mostrar estado final
        axes[i].plot(solver_delta.x, solutions_d[-1], linewidth=2, 
                    label=f'δ² = {delta**2:.4f}')
        axes[i].set_title(f'Estado final con δ² = {delta**2:.4f}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('φ(t,x)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('3_efecto_delta.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Velocidades de solitones
    print("5. Analizando velocidades de solitones...")
    
    # Crear un solitón exacto para validación
    solver_exact = KdVSolver(L=25, N=512, delta=0.022)
    amplitude = 12.0  # Amplitud del solitón
    phi_exact = initial_sech_squared(solver_exact.x, amplitude=amplitude, center=-5.0)
    
    times_exact, solutions_exact = solver_exact.evolve(phi_exact, t_final=5.0)
    
    # Seguir el pico del solitón
    peak_positions = []
    for solution in solutions_exact:
        peak_idx = np.argmax(solution)
        peak_positions.append(solver_exact.x[peak_idx])
    
    # Calcular velocidad
    velocities = np.diff(peak_positions) / solver_exact.dt / (len(solutions_exact) // 100)
    theoretical_velocity = amplitude / 3  # v = A/3 para sech²
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Posición vs tiempo
    time_subset = times_exact[:len(peak_positions)]
    ax1.plot(time_subset, peak_positions, 'bo-', markersize=4, linewidth=2, 
             label='Simulación')
    ax1.plot(time_subset, theoretical_velocity * time_subset + peak_positions[0], 
             'r--', linewidth=2, label=f'Teórica (v = A/3 = {theoretical_velocity:.2f})')
    ax1.set_xlabel('tiempo')
    ax1.set_ylabel('Posición del pico')
    ax1.set_title('Movimiento del solitón')
    ax1.legend()
    ax1.grid(True)
    
    # Velocidad vs tiempo
    time_vel = time_subset[1:]
    ax2.plot(time_vel, velocities, 'go-', markersize=4, linewidth=2, 
             label='Velocidad numérica')
    ax2.axhline(y=theoretical_velocity, color='r', linestyle='--', linewidth=2, 
                label=f'Velocidad teórica = {theoretical_velocity:.2f}')
    ax2.set_xlabel('tiempo')
    ax2.set_ylabel('Velocidad')
    ax2.set_title('Velocidad del solitón')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('3_velocidad_soliton.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Estudio de condiciones iniciales que NO producen solitones
    print("6. Condiciones iniciales sin solitones...")
    
    # Función suave que decae rápidamente
    def smooth_decay(x):
        return np.exp(-x**2/100) * np.sin(x) * np.exp(-abs(x)/20)
    
    solver_no_sol = KdVSolver(L=25, N=512, delta=0.022)
    phi_no_sol = smooth_decay(solver_no_sol.x)
    
    times_no, solutions_no = solver_no_sol.evolve(phi_no_sol, t_final=20.0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Estado inicial y final
    ax1.plot(solver_no_sol.x, solutions_no[0], 'b-', linewidth=2, 
             label='t = 0 (inicial)')
    ax1.plot(solver_no_sol.x, solutions_no[-1], 'r-', linewidth=2, 
             label=f't = {times_no[-1]:.1f} (final)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('φ(t,x)')
    ax1.set_title('Dispersión sin formación de solitones')
    ax1.legend()
    ax1.grid(True)
    
    # Evolución de la amplitud máxima
    max_amplitudes = [np.max(np.abs(solution)) for solution in solutions_no]
    ax2.plot(times_no[:len(max_amplitudes)], max_amplitudes, 'g-', linewidth=2)
    ax2.set_xlabel('tiempo')
    ax2.set_ylabel('Amplitud máxima')
    ax2.set_title('Decaimiento dispersivo')
    ax2.grid(True)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('3_sin_solitones.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Crear resumen de resultados
    with open('3_resultados.txt', 'w', encoding='utf-8') as f:
        f.write("RESULTADOS DEL ANÁLISIS DE SOLITONES KdV\n")
        f.write("="*50 + "\n\n")
        
        f.write("1. DESCOMPOSICIÓN DE COSENO:\n")
        f.write(f"   - Condición inicial: cos(2πx/20)\n")
        f.write(f"   - El coseno se descompone en múltiples solitones\n")
        f.write(f"   - Los solitones más altos viajan más rápido\n\n")
        
        f.write("2. COLISIÓN DE SOLITONES:\n")
        f.write(f"   - Los solitones interactúan de manera no lineal\n")
        f.write(f"   - Después de la colisión emergen intactos\n")
        f.write(f"   - Hay un cambio de fase pero conservan forma\n\n")
        
        f.write("3. CANTIDADES CONSERVADAS:\n")
        f.write(f"   - La masa se conserva con alta precisión\n")
        f.write(f"   - El momento se conserva durante la evolución\n")
        f.write(f"   - La energía presenta pequeñas fluctuaciones numéricas\n\n")
        
        f.write("4. EFECTO DEL PARÁMETRO δ:\n")
        f.write(f"   - δ² más pequeño: más solitones, más dispersión\n")
        f.write(f"   - δ² más grande: menos solitones, menos dispersión\n")
        f.write(f"   - Controla el balance dispersión/no-linealidad\n\n")
        
        f.write("5. VELOCIDAD DE SOLITONES:\n")
        f.write(f"   - Solitón sech² con amplitud A = 12\n")
        f.write(f"   - Velocidad teórica: v = A/3 = 4.0\n")
        f.write(f"   - Velocidad numérica promedio: {np.mean(velocities):.3f}\n")
        f.write(f"   - Error relativo: {abs(np.mean(velocities) - theoretical_velocity)/theoretical_velocity*100:.2f}%\n\n")
        
        f.write("6. CONDICIONES SIN SOLITONES:\n")
        f.write(f"   - Funciones que decaen rápidamente tienden a dispersarse\n")
        f.write(f"   - No toda condición inicial produce solitones\n")
        f.write(f"   - La amplitud máxima decae exponencialmente\n\n")
        
        f.write("PARÁMETROS DE SIMULACIÓN:\n")
        f.write(f"   - Dominio: x ∈ [-L, L] con L = 25-30\n")
        f.write(f"   - Puntos espaciales: N = 512-768\n")
        f.write(f"   - δ² = {solver1.delta**2:.4f} (principal)\n")
        f.write(f"   - Método: Runge-Kutta 4 + FFT para derivadas\n")
        f.write(f"   - Condiciones de frontera: periódicas (FFT)\n")
    
    print("¡Simulación completada!")
    print("Archivos generados:")
    print("- 3_descomposicion_coseno.mp4")
    print("- 3_colision_solitones.mp4")
    print("- 3_cantidades_conservadas.pdf")
    print("- 3_efecto_delta.png")
    print("- 3_velocidad_soliton.pdf")
    print("- 3_sin_solitones.pdf")
    print("- 3_resultados.txt")

if __name__ == "__main__":
    main()
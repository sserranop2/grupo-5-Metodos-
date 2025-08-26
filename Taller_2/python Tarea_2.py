#Métodos Computacionales 2: Taller 2 - Fourier
#Grupo #5

#Librerias
import numpy as np
from typing import Iterable
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import os

#1. Intuición e interpretación (Transformada general)

#1.a. Límite de Nyquist
#1.a.a. Implementación
def Fourier_transform(t: np.ndarray, y: np.ndarray, freqs: Iterable[float],
                      max_matrix_bytes: int = int(1e8)) -> np.ndarray:
    """
    Calcula F_k = sum_{i=0..N-1} y_i * exp(-2j*pi * f_k * t_i) para cada f_k en `freqs`.

    Comportamiento: Se realiza una estimacion del tiempo de calculo y uso de memoria. dependiendo del tamaño de datos y 
    la cantidad de frecuencias a evaluar.
      - Si la matriz completa (M x N) cabe en memoria (estimada por max_matrix_bytes),
        se construye la matriz exponencial con np.outer y se realiza un único producto matricial
        (vectorizado, sin bucles Python por elemento). (Muy rápido pero consume más memoria).
      - Si no cabe, se procesa por bloques de frecuencias. Cada bloque construye una matriz
        (K x N) que sí cabe en memoria y se multiplica por y (vectorizado por bloque).
        Esto evita construir la matriz (M x N) completa a la vez. (Mas lento pero no consume tanta memoria).
    

    Args:
      t: array-like de tiempos, forma (N,)
      y: array-like de medidas, forma (N,)
      freqs: array-like de frecuencias donde evaluar, forma (M,)
      max_matrix_bytes: umbral (bytes) para decidir si construir la matriz completa.
                        Aproximación: complex128 ocupa 16 bytes/elemento.

    Retorna:
      np.ndarray complejo de forma (M,) con F(f_k) para cada frecuencia f_k.
    """
    t = np.asarray(t).ravel()
    y = np.asarray(y).ravel()
    freqs = np.asarray(freqs).ravel()

    if t.size != y.size:
        raise ValueError("t and y must have the same length")

    N = t.size
    M = freqs.size

    # Estimación memoria en bytes para una matriz complex128 de tamaño M x N
    estimated_bytes = int(M) * int(N) * 16

    if estimated_bytes <= max_matrix_bytes:
        # Construcción única: (M, N) @ (N,) -> (M,)
        E = np.exp(-2j * np.pi * np.outer(freqs, t))
        return E.dot(y).astype(np.complex128)

    # Si no cabe, procesar por bloques de frecuencias.
    # Calcular cuántas frecuencias por bloque podemos permitir (al menos 1)
    bytes_per_row = int(N) * 16  # bytes para una fila (una frecuencia)
    chunk_M = max(1, int(max_matrix_bytes // bytes_per_row))

    Fk = np.empty(M, dtype=np.complex128)
    for start in range(0, M, chunk_M):
        stop = min(M, start + chunk_M)
        freqs_block = freqs[start:stop]                     # shape (K,)
        # matriz exponencial del bloque: shape (K, N)
        E_block = np.exp(-2j * np.pi * np.outer(freqs_block, t))
        F_block = E_block.dot(y)                            # shape (K,)
        Fk[start:stop] = F_block

    return Fk

#1.a.b. Prueba
# Función proporcionada en el enunciado
def generate_data(tmax, dt, A, freq, noise):
    ts = np.arange(0, tmax + dt, dt)
    return ts, np.random.normal(loc=A * np.sin(2 * np.pi * ts * freq), scale=noise)

# Parámetros 
tmax = 1.0
dt = 1e-3
A = 1.0
freq_signal = 150.0
noise = 0.1

# Generar datos
t, y = generate_data(tmax, dt, A, freq_signal, noise)

# Eje de frecuencias 
Fs = 1.0 / dt
nyquist = Fs / 2.0
nyquist_2_7 = 2.7 * nyquist
freqs = np.linspace(0.0, nyquist_2_7, 400)

# Calcular transformada 
Fk = Fourier_transform(t, y, freqs)

# Graficar 
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(freqs, np.abs(Fk))
ax.axvline(nyquist, color='r', linestyle='--', linewidth=1, label=f'Nyquist = {nyquist:.2f} Hz')
ax.axvline(nyquist_2_7, color='g', linestyle='--', linewidth=1, label=f'2.7×Nyquist = {nyquist_2_7:.2f} Hz')

ax.set_xlabel('Frecuencia (Hz)')
ax.set_ylabel('|F(f)|')
ax.set_title('Espectro hasta 2.7×Nyquist')
ax.grid(alpha=0.3)
ax.legend()

# Guardar como PDF
fig.savefig('1.a.pdf', bbox_inches='tight')
plt.close(fig)

#1.b. Signal-to-noise
def sn_to_no(tmax=1.0, dt=1e-3, A=1.0, freq=150, noise=0.1,
                  sn_min=0.01, sn_max=1.0, n=20, out_pdf="1.b_SNfreq_vs_SNtime.pdf"):
    sn_times = np.logspace(np.log10(sn_min), np.log10(sn_max), n)
    sn_freqs = np.zeros(n)
    for i, sn_t in enumerate(sn_times):
        A = sn_t * noise
        t, y = generate_data(tmax, dt, A, freq, noise)
        mag = np.abs(np.fft.rfft(y))
        peak = np.argmax(mag[1:]) + 1
        noise_band = np.delete(mag, [0, peak-2, peak-1, peak, peak+1, peak+2])
        sn_freqs[i] = mag[peak] / np.std(noise_band)
    # Ajuste en log-log
    coeffs = np.polyfit(np.log10(sn_times), np.log10(sn_freqs), 1)
    fit = 10**coeffs[1] * sn_times**coeffs[0]
    # Gráfica
    plt.figure(figsize=(8,5))
    plt.loglog(sn_times, sn_freqs, 'o', label='Datos')
    plt.loglog(sn_times, fit, '-', label=f"Ajuste: SNfreq ≈ {10**coeffs[1]:.3g}·SNtime^{coeffs[0]:.3f}")
    plt.xlabel('SNtime'); plt.ylabel('SNfreq'); plt.title('SNfreq vs SNtime (log-log)')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    return coeffs

coeffs = sn_to_no()


#Para pensar: ¿qué variables harían cambiar este comportamiento? ( A , tmax , freq , dt , …)
"""
Amplitud de la señal (A) y nivel de ruido (noise)
Directamente afectan SN_time = A/noise. En frecuencia, mayor amplitud eleva la altura del pico relativo al fondo, mejorando SN_freq.

Duración de la señal (tmax)
Más duración produce mejor resolución en frecuencia (ventana más larga → picos espectrales más estrechos) y reduce el ruido espectral.

Tasa de muestreo (dt)
Aumentar la frecuencia de muestreo y añadir un filtrado que limita el ancho de banda efectivo da como resultado menos ruido agregado.

Ancho de banda efectivo / Filtrado
Reducir el ancho de banda del sistema (con filtros) disminuye el ruido total en frecuencia, aumentando la SNR en frecuencia.

Profundidad de cuantización / Ruido del ADC
Mejorar el rango dinámico o reducir el ruido de cuantización (por sobre-muestreo y filtrado) eleva la SNR total.
"""

#1.c. Principio de indeterminación de las ondas
def peak_width(freqs, mag, level=0.5):
    peak = np.argmax(mag)
    target = mag[peak] * level
    left = freqs[np.where(mag[:peak] <= target)[0][-1]]
    right = freqs[peak + np.where(mag[peak:] <= target)[0][0]]
    return right - left

dt, A, freq, noise = 1e-3, 1.0, 150.0, 0.05
tmax_list, widths = [0.25, 0.5, 1.0, 2.0], []

for tmax in tmax_list:
    t, y = generate_data(tmax, dt, A, freq, noise)
    f = np.fft.rfftfreq(len(y), dt)
    mag = np.abs(np.fft.rfft(y))
    widths.append(peak_width(f, mag / mag.max()))

# Espectros normalizados
plt.figure()
for tmax in tmax_list:
    t, y = generate_data(tmax, dt, A, freq, noise)
    f = np.fft.rfftfreq(len(y), dt)
    mag = np.abs(np.fft.rfft(y)) / np.abs(np.fft.rfft(y)).max()
    plt.plot(f, mag, label=f"tmax={tmax}s")
plt.xlim(freq-50, freq+50)
plt.xlabel("Frecuencia (Hz)"); plt.ylabel("Espectro normalizado")
plt.title("Espectros normalizados vs tmax")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("1.c_spectra.pdf"); plt.close()

# Ancho del pico vs tmax
plt.figure()
plt.loglog(tmax_list, widths, 'o-')
plt.xlabel("tmax (s)"); plt.ylabel("FWHM (Hz)")
plt.title("Ancho de pico vs tmax")
plt.grid(True, which="both", ls=":")
plt.savefig("1.c_width_vs_tmax.pdf"); plt.close()



#1.d. (BONO) Más allá de Nyquist
def generate_data(tmax, dt, A, freq, noise, sampling_noise=0.0, rng=None):
    """ A*sin(2π f t) medida en tiempos con jitter: t' = t + N(0, sampling_noise·dt) + ruido de medición """
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(0.0, tmax + dt, dt)
    if sampling_noise > 0.0:
        t = t + rng.normal(0.0, sampling_noise * dt, size=t.shape)
    t = np.sort(t)                      # para que el eje x crezca
    y = A * np.sin(2*np.pi*freq*t)
    y = rng.normal(y, noise, size=t.shape)
    return t, y

def plot_beyond_nyquist_panels(
    sampling_noises=(0.002, 0.01, 0.10, 0.20, 0.30),   # 0.2%, 1%, 10%, 20%, 30% de dt
    tmax=2.2, dt=0.05, A=1.0, noise=0.08,              # Fs=20 Hz, Nyquist=10 Hz
    freq_true_factor=0.3,                               # f_real=3 Hz -> picos en 3, 17 (=20-3), 23 (=20+3)
    trials_time=25, trials_freq=40,                     # nº realizaciones para "borroso" y promedio
    out_pdf="1.d.pdf", seed=2025
):
    Fs  = 1.0/dt
    nyq = 0.5/dt
    f_real = freq_true_factor * nyq           # = 3 Hz con los parámetros arriba
    fmax  = 2.7 * nyq                         # límite 2.7×Nyquist = 27 Hz
    freqs = np.linspace(0.0, fmax, 2048)      # eje continuo hasta 2.7×Nyquist

    rng = np.random.default_rng(seed)
    nrows = len(sampling_noises)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 9))
    if nrows == 1: axes = np.array([axes])

    ref_level = None
    for r, sn in enumerate(sampling_noises):
        # --------- Izquierda: SOLO PUNTOS, "borrosos" con muchas realizaciones grises ---------
        axL = axes[r, 0]
        for _ in range(trials_time):
            tt, yy = generate_data(tmax, dt, A, f_real, noise, sampling_noise=sn, rng=rng)
            axL.scatter(tt, yy, s=14, color="gray", alpha=0.18)   # nubes grises (borroso)
        # Una realización "principal" en negro encima
        tt, yy = generate_data(tmax, dt, A, f_real, noise, sampling_noise=sn, rng=rng)
        axL.scatter(tt, yy, s=20, color="k", alpha=0.95)
        axL.set_ylabel(f"{sn*100:.1f}%")
        if r == 0: axL.set_title("Time domain")
        if r == nrows-1: axL.set_xlabel("t (s)")

        # --------- Derecha: muchas realizaciones grises + PROMEDIO negro (sin normalizar) ---------
        axR, mags = axes[r, 1], []
        for _ in range(trials_freq):
            t, y = generate_data(tmax, dt, A, f_real, noise, sampling_noise=sn, rng=rng)
            m = np.abs(Fourier_transform(t, y, freqs))
            mags.append(m)
            axR.plot(freqs, m, lw=0.8, alpha=0.12, color="gray")
        mean_mag = np.mean(mags, axis=0)
        if ref_level is None:                 # misma escala y línea roja para todas las filas
            ref_level = float(mean_mag.max())
        axR.axhline(ref_level, ls="--", lw=1.5, color="red")
        axR.plot(freqs, mean_mag, lw=2.2, color="black")
        if r == 0: axR.set_title("Frequency domain")
        if r == nrows-1: axR.set_xlabel("Frecuencia (Hz)")
        axR.set_xlim(0, fmax)
        axR.set_ylim(0, 1.05*ref_level)

    fig.text(0.02, 0.5, "Sampling noise std as percentage of mean sampling time",
             va="center", rotation="vertical")
    fig.tight_layout(h_pad=1.0, w_pad=2.0)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# Genera y guarda 1.d.pdf
plot_beyond_nyquist_panels()


#3Filtrando imágenes (FFT 2D)
# Cargar en [0,1] y en float
img = Image.open("miette.jpg").convert("RGB")
I = np.asarray(img).astype(np.float32) / 255.0   # (H, W, 3)
H, W = I.shape[:2]
print(I.shape, I.dtype)

def gaussian_lowpass_mask(shape_hw, sigma_px):
    H, W = shape_hw

    u = np.arange(-W//2, W//2)
    v = np.arange(-H//2, H//2)
    V, U = np.meshgrid(v, u, indexing="ij")
    D2 = U**2 + V**2
    Hmask = np.exp(-D2 / (2.0 * (sigma_px**2)))
    return Hmask

def lowpass_gaussian_fft2(channel_01, Hmask):
    F = np.fft.fft2(channel_01)
    F_shift = np.fft.fftshift(F)
    F_filt = F_shift * Hmask
    out = np.fft.ifft2(np.fft.ifftshift(F_filt)).real
    return out

# Elige el nivel de blur con una fracción del tamaño de la imagen
sigma_frac = 0.03
sigma_px = sigma_frac * min(H, W)

Hmask = gaussian_lowpass_mask((H, W), sigma_px)

out = np.empty_like(I)
for c in range(3):
    out[..., c] = lowpass_gaussian_fft2(I[..., c], Hmask)

# Normalizar y guardar
out = np.clip(out, 0.0, 1.0)
out_uint8 = (out * 255.0).round().astype(np.uint8)
Image.fromarray(out_uint8).save("3.a.jpg")
print("Guardado en 3.a.jpg")




# 3.b.a)
img = Image.open("p_a_t_o.jpg").convert("L")
I = np.asarray(img).astype(np.float32) / 255.0
H, W = I.shape
print("Tamaño:", I.shape)


# Bloque 2 — FFT 2D centrada y espectro (log |F|) con ejes

F   = np.fft.fft2(I)
Fsh = np.fft.fftshift(F)

mag = np.log1p(np.abs(Fsh))



# Bloque 3R — (reemplaza tu Bloque 3)

rects = [
     ((255, 125), (3,120 )),
     ((400, 255), (130, 3)),
]

def build_rect_notch_mask(shape_hw, rect_list):
    """
    Muescas rectangulares duras (cuadritos) en el espectro centrado (fftshift).
    - shape_hw: (H, W)
    - rect_list: [ ((cy, cx), (hy, hx)), ... ]
    """
    H, W = shape_hw
    mask = np.ones((H, W), dtype=np.float32)

    # Centro del espectro (por fftshift)
    cy0, cx0 = H // 2, W // 2

    def zero_rect(mask, cy, cx, hy, hx):
        y1 = max(0, cy - hy)
        y2 = min(H, cy + hy + 1)
        x1 = max(0, cx - hx)
        x2 = min(W, cx + hx + 1)
        mask[y1:y2, x1:x2] = 0.0

    for ((cy, cx), (hy, hx)) in rect_list:
        # Evitar tocar el centro
        if abs(cy - cy0) < 2 and abs(cx - cx0) < 2:
            continue

        zero_rect(mask, cy, cx, hy, hx)

        cy_sym = (2 * cy0 - cy) % H
        cx_sym = (2 * cx0 - cx) % W
        zero_rect(mask, cy_sym, cx_sym, hy, hx)

    return mask

Hmask_preview = build_rect_notch_mask((H, W), rects)


def build_wedge_notch_mask(shape_hw, angles_deg, width_deg=3.0, r_min=12, r_max=None):
    """
    Anula una o más direcciones (ángulos) con una cuña angosta alrededor de cada ángulo.
    Trabaja sobre el espectro centrado (fftshift).

    angles_deg : lista de ángulos en grados (como los ves en tu figura).
    width_deg  : ancho angular total de la cuña (p. ej. 3–6°).
    r_min      : radio mínimo para NO tocar el DC (centro).
    r_max      : radio máximo (si None, usa el máximo posible).
    """
    import numpy as np

    H, W = shape_hw
    if r_max is None:
        r_max = np.hypot(H//2, W//2)

    # Coordenadas centradas
    yy = np.arange(H) - H//2
    xx = np.arange(W) - W//2
    X, Y = np.meshgrid(xx, yy)
    R = np.hypot(X, Y)
    A = np.degrees(np.arctan2(Y, X))

    def angdiff(a, b):
        d = (a - b + 180.0) % 360.0 - 180.0
        return np.abs(d)

    mask = np.ones((H, W), dtype=np.float32)
    half_w = width_deg / 2.0

    for ang in angles_deg:
        d1 = angdiff(A, ang)
        d2 = angdiff(A, ang + 180.0)
        sector = ((d1 <= half_w) | (d2 <= half_w)) & (R >= r_min) & (R <= r_max)
        mask[sector] = 0.0

    return mask

angles_deg = [-20.0]
width_deg  = 4.0
r_min      = 10
r_max      = None

Hwedge = build_wedge_notch_mask((H, W), angles_deg, width_deg, r_min, r_max)


Hrect = build_rect_notch_mask((H, W), rects)

Hwedge = build_wedge_notch_mask((H, W), angles_deg, width_deg, r_min, r_max)

Hmask = Hrect * Hwedge

F_filt = Fsh * Hmask
If = np.fft.ifft2(np.fft.ifftshift(F_filt)).real
If = np.clip(If, 0.0, 1.0)

out_uint8 = (If * 255.0).round().astype(np.uint8)
Image.fromarray(out_uint8).save("3.b.a.jpg")



# 3.b.b)
# Bloque 1 — Importar librerías, cargar imagen y mostrarla con ejes
img = Image.open("g_a_t_o.png").convert("L")
I   = np.asarray(img).astype(np.float32) / 255.0

H, W = I.shape
print("Tamaño:", I.shape)


# Bloque 2 — FFT 2D centrada y espectro (log |F|) con ejes

F   = np.fft.fft2(I)
Fsh = np.fft.fftshift(F)

mag = np.log1p(np.abs(Fsh))


# Bloque a — calcular histograma angular y visualizar
Mag = np.abs(Fsh)
H, W = Mag.shape
cy, cx = H//2, W//2

Y, X = np.ogrid[:H, :W]
dx, dy = (X - cx), (Y - cy)
R  = np.sqrt(dx*dx + dy*dy)
th = (np.rad2deg(np.arctan2(dy, dx)) + 180.0) % 180.0

r0 = 20
valid = R > r0

bins = 360
hist, edges = np.histogram(th[valid], bins=bins, range=(0,180), weights=Mag[valid])
ang_centers = 0.5*(edges[1:] + edges[:-1])

plt.figure(figsize=(8,3), dpi=120)
plt.plot(ang_centers, hist)
plt.xlabel("Ángulo [°]"); plt.ylabel("Energía (ponderada)")
plt.title("Histograma angular del espectro (excluyendo DC)")
plt.tight_layout(); 

# Bloque b — Detectar automáticamente TODOS los picos angulares significativos

def detect_angle_peaks(hist, ang_centers, prominence_rel=0.25, min_sep_deg=3):
    """
    prominence_rel: fracción de (max - mediana) para considerar un pico (0.15–0.35 típico)
    min_sep_deg: separación mínima entre picos (para no contar casi el mismo dos veces)
    """
    # Suavizado ligero para evitar falsos picos muy estrechos
    k = 5
    ker = np.ones(k)/k
    hist_s = np.convolve(hist, ker, mode='same')

    med = np.median(hist_s)
    rng = (hist_s.max() - med)
    thr = med + prominence_rel * max(rng, 1e-9)

    # Candidatos: por encima del umbral
    cand = np.where(hist_s >= thr)[0]

    if cand.size == 0:
        return []

    # Agrupar candidatos contiguos; tomar el máximo de cada grupo
    groups = []
    g = [cand[0]]
    for i in cand[1:]:
        if i == g[-1] + 1:
            g.append(i)
        else:
            groups.append(g); g = [i]
    groups.append(g)

    peaks_idx = [g[np.argmax(hist_s[g])] for g in groups]

    # En grados:
    angles = [ang_centers[i] for i in peaks_idx]

    # Fusionar picos demasiado cercanos (< min_sep_deg)
    angles_sorted = sorted(angles)
    fused = []
    for a in angles_sorted:
        if not fused or abs(a - fused[-1]) >= min_sep_deg:
            fused.append(a)
        else:
            # si muy cerca, quedarse con el que tenga hist más alto
            prev = fused[-1]
            ia = np.argmin(np.abs(ang_centers - a))
            ip = np.argmin(np.abs(ang_centers - prev))
            if hist_s[ia] > hist_s[ip]:
                fused[-1] = a

    return fused

angles_detected = detect_angle_peaks(hist, ang_centers,
                                     prominence_rel=0.05,  # sube/baja sensibilidad
                                     min_sep_deg=3)

print(f"Ángulos detectados ({len(angles_detected)}):",
      [f"{a:.1f}°" for a in angles_detected])

# Bloque c — Construir máscara con TODAS las cuñas y aplicar

def build_wedge_mask(shape_hw, angles_deg, half_ap_deg=2.0, rmin=0, rmax=None, keep_dc=2):
    if not isinstance(angles_deg, (list, tuple, np.ndarray)):
        angles = [angles_deg]
    else:
        angles = list(angles_deg)

    H, W = shape_hw
    mask = np.ones((H, W), dtype=np.float32)
    cy, cx = H//2, W//2

    Y, X = np.ogrid[:H, :W]
    dx, dy = (X - cx), (Y - cy)
    R  = np.sqrt(dx*dx + dy*dy)
    th = (np.rad2deg(np.arctan2(dy, dx)) + 180.0) % 180.0

    def angdist(a, t):
        d     = np.abs((a - t + 90) % 180 - 90)
        d_sym = np.abs((a - ((t+180)%180) + 90) % 180 - 90)
        return np.minimum(d, d_sym)

    for t in angles:
        dth = angdist(th, t)
        band = dth <= half_ap_deg
        if rmin is not None: band &= (R >= rmin)
        if rmax is not None: band &= (R <= rmax)
        mask[band] = 0.0

    if keep_dc and keep_dc > 0:
        mask[R <= keep_dc] = 1.0

    return mask

# --- Parámetros de la cuña ---
half_ap = 2.5   # semiapertura en grados
rmin    = 0     # que llegue hasta el centro
rmax    = None
keep_dc = 2     # proteger del DC

Wmask_all = build_wedge_mask((H, W), angles_detected,
                             half_ap_deg=half_ap, rmin=rmin, rmax=rmax, keep_dc=keep_dc)



# Filtrar, reinyectar DC y volver a imagen
F_filt = Fsh * Wmask_all
cy, cx = H//2, W//2
F_filt[cy, cx] = Fsh[cy, cx]  # conservar brillo global

If = np.fft.ifft2(np.fft.ifftshift(F_filt)).real
If = np.clip(If, 0.0, 1.0)

from PIL import Image
out_uint8 = (If*255.0).round().astype(np.uint8)
Image.fromarray(out_uint8).save("3.b.b.png")
print("Guardado 3.b.b.png")




#4A - plicación real: datos con muestreo aleatorio
fn = "OGLE-LMC-CEP-0001.dat"
t, y, s = np.loadtxt(fn, unpack=True)

t = t - np.min(t)                 # trasladar tiempos
y0 = y - np.mean(y)               # quitar media (DC)

# Periodograma DFT para tiempos irregulares
T  = np.ptp(t)                    # <-- reemplaza t.ptp() por np.ptp(t)
df = 1.0 / (5.0 * T)              # resolución ~ 1/(5*T)
f  = np.arange(0.01, 2.0 + df/2, df)
E  = np.exp(-2j*np.pi * t[:,None] * f[None,:])
S  = (y0[:,None] * E).sum(0)
P  = (np.abs(S)**2) / max(len(y0),1)

# Refinar pico por parábola
k = int(np.argmax(P)); f_best = f[k]
if 0 < k < len(f)-1:
    df0 = f[1]-f[0]
    denom = (P[k-1] - 2*P[k] + P[k+1])
    dp = 0.5*(P[k-1] - P[k+1]) / (denom if denom!=0 else 1e-12)
    f_best = f[k] + dp*df0

period = 1.0 / f_best

# Fasear y graficar
phi = np.mod(f_best * t, 1.0)
idx = np.argsort(phi)
plt.figure(figsize=(6,4))
plt.scatter(phi[idx], y[idx], s=10, alpha=0.85)
plt.gca().invert_yaxis()
plt.xlabel("Fase ϕ = mod(f · t, 1)")
plt.ylabel("Brillo (magnitud)")
plt.title(f"f = {f_best:.6f} c/d  |  P = {period:.6f} d")
plt.tight_layout()
plt.savefig("4.pdf", dpi=200)
plt.close()




#5. Aplicación real: Reconstrucción tomográfica filtrada
# Cargar los datos
signals = np.load('5.npy')  # o usar 'taller_2/reference.npy'
rotation_angles = np.arange(0, 180, 0.5)

# Obtener dimensiones
rows = len(signals[0])
n_projections = len(signals)

# Inicializar la matriz de imagen sumada correctamente
imagen_sumada = np.zeros((rows, rows), dtype=np.float64)

print(f"Número de proyecciones: {n_projections}")
print(f"Número de ángulos: {len(rotation_angles)}")
print(f"Dimensiones de cada señal: {rows}")

# Función para aplicar filtro pasa-altas
def apply_highpass_filter(signal):
    """
    Aplica un filtro pasa-altas a la señal usando FFT
    """
    # Transformada de Fourier
    fft_signal = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal))
    
    # Crear filtro pasa-altas (suprimir frecuencias bajas)
    # Puedes ajustar este valor para controlar la fuerza del filtro
    cutoff_freq = 0.1  # Frecuencia de corte normalizada
    highpass_filter = np.abs(frequencies) > cutoff_freq
    
    # Aplicar filtro
    filtered_fft = fft_signal * highpass_filter
    
    # Transformada inversa
    filtered_signal = np.real(np.fft.ifft(filtered_fft))
    
    return filtered_signal

# Procesar cada proyección
for i in range(min(n_projections, len(rotation_angles))):
    # Obtener la señal actual
    signal = signals[i]
    
    # Aplicar filtro pasa-altas para mejorar el contraste
    filtered_signal = apply_highpass_filter(signal)
    
    # Crear imagen 2D repitiendo la señal filtrada
    imagen_2d = np.tile(filtered_signal[:, None], (1, rows))
    
    # Rotar la imagen
    imagen_rotada = ndi.rotate(
        imagen_2d,
        rotation_angles[i],
        reshape=False,
        mode="reflect"
    )
    
    # Sumar a la imagen total
    imagen_sumada += imagen_rotada

# Normalizar la imagen final
imagen_final = imagen_sumada / n_projections

# Ajustar contraste y convertir a uint8 para guardar
imagen_final_normalizada = np.clip(imagen_final, 0, np.max(imagen_final))
imagen_final_normalizada = (imagen_final_normalizada / np.max(imagen_final_normalizada) * 255).astype(np.uint8)

# Guardar la imagen
plt.figure(figsize=(10, 10))
plt.imshow(imagen_final_normalizada)
plt.axis('off')
plt.title('Reconstrucción Tomográfica con Filtro Pasa-Altas')
plt.savefig('4.png', bbox_inches='tight', dpi=150)  # <— guardado en carpeta actual
plt.close()

# También mostrar sin filtro para comparación
imagen_sin_filtro = np.zeros((rows, rows), dtype=np.float64)

for i in range(min(n_projections, len(rotation_angles))):
    signal = signals[i]
    
    # Sin filtro
    imagen_2d = np.tile(signal[:, None], (1, rows))
    imagen_rotada = ndi.rotate(
        imagen_2d,
        rotation_angles[i],
        reshape=False,
        mode="reflect"
    )
    imagen_sin_filtro += imagen_rotada

imagen_sin_filtro = imagen_sin_filtro / n_projections
imagen_sin_filtro_norm = np.clip(imagen_sin_filtro, 0, np.max(imagen_sin_filtro))
imagen_sin_filtro_norm = (imagen_sin_filtro_norm / np.max(imagen_sin_filtro_norm) * 255).astype(np.uint8)

# Crear figura comparativa
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

ax1.imshow(imagen_sin_filtro_norm)
ax1.set_title('Sin Filtro (Borrosa)')
ax1.axis('off')

ax2.imshow(imagen_final_normalizada)
ax2.set_title('Con Filtro Pasa-Altas (Mayor Contraste)')
ax2.axis('off')

plt.tight_layout()
plt.savefig('comparacion_tomografia.png', bbox_inches='tight', dpi=150)  # <— guardado en carpeta actual
plt.close()






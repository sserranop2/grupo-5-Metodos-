#Métodos Computacionales 2: Taller 2 - Fourier
#Grupo #5

#Librerias
import numpy as np
from typing import Iterable
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
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


#3
# =============== Utilidades básicas ===============
def base_dir():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path(os.getcwd())

def to_float01(a):  return np.asarray(a).astype(np.float32) / 255.0
def to_uint8(a01):  return (np.clip(a01,0,1) * 255.0 + 0.5).astype(np.uint8)
def save_img(p, a01): Image.fromarray(to_uint8(a01)).save(str(p))

def fft2_shift(x):    return np.fft.fftshift(np.fft.fft2(x))
def ifft2_unshift(X): return np.real(np.fft.ifft2(np.fft.ifftshift(X)))

def pad_reflect(img, pad):     return np.pad(img, ((pad,pad),(pad,pad)), mode="reflect")
def crop_center(img, H, W, p): return img[p:p+H, p:p+W]

# =============== 3.a: desenfoque gaussiano (FFT) ===============
def gaussian_lowpass(shape_hw, sigma_px):
    H, W = shape_hw
    cy, cx = H//2, W//2
    yy, xx = np.ogrid[:H, :W]
    r2 = (yy - cy)**2 + (xx - cx)**2
    return np.exp(-0.5 * r2 / float(sigma_px**2))

def run_3a(miette_path, out_path, sigma_frac=0.06):
    rgb = to_float01(np.array(Image.open(miette_path).convert("RGB")))
    H, W = rgb.shape[:2]
    Hmask = gaussian_lowpass((H, W), sigma_frac * min(H, W))
    out = np.empty_like(rgb)
    for c in range(3):
        out[..., c] = ifft2_unshift(fft2_shift(rgb[..., c]) * Hmask)
    save_img(out_path, np.clip(out, 0, 1))

# =============== 3.b: peineta de muescas (fundamental + armónicos) ===============
def _disk_mask(H, W, cy, cx, r):
    yy, xx = np.ogrid[:H, :W]
    return (yy - cy)**2 + (xx - cx)**2 <= r*r

def _find_fundamental_peak(mag, guard_r=18, num_candidates=80):
    """
    Devuelve (u_y, u_x) (dirección unitaria del patrón) y r0 (radio/frecuencia fundamental en píxeles)
    a partir de los picos más fuertes del espectro (excluyendo el centro).
    """
    H, W = mag.shape
    cy, cx = H//2, W//2

    # Ignora DC y bajas frecuencias
    work = mag.copy()
    work[_disk_mask(H, W, cy, cx, guard_r)] = 0.0

    peaks = []
    Wk = work.copy()
    for _ in range(num_candidates):
        idx = np.argmax(Wk)
        if Wk.flat[idx] <= 0:
            break
        y, x = np.unravel_index(idx, Wk.shape)
        peaks.append((y, x, Wk[y, x]))
        rr = 3
        yy, xx = np.ogrid[:H, :W]
        Wk[(yy - y)**2 + (xx - x)**2 <= rr*rr] = 0.0

    if not peaks:
        # Fallback raro: diagonal suave
        return (1/np.sqrt(2), 1/np.sqrt(2)), max(guard_r+8, 20)

    # Empareja con el simétrico y elige el par más fuerte
    best = None
    for y, x, val in peaks:
        ys, xs = (2*cy - y) % H, (2*cx - x) % W
        val_sym = work[ys, xs]
        score = val + val_sym
        r = np.hypot(y - cy, x - cx) + 1e-9
        if best is None or score > best[0]:
            dy, dx = (y - cy) / r, (x - cx) / r
            best = (score, (dy, dx), r)

    _, (uy, ux), r0 = best
    return (uy, ux), float(r0)

def _build_notch_comb_mask(shape, direction_uv, r0,
                           harm_max=None, guard_r=18,
                           notch_r0=6.0, notch_scale=0.012):
    """
    Máscara multiplicativa (1 entre muescas, 0 en muescas) que anula
    los armónicos ±m*r0 a lo largo de la recta definida por 'direction_uv'.
    Cada muesca es un disco (radio = notch_r0 + notch_scale * m * r0).
    """
    H, W = shape
    cy, cx = H//2, W//2
    yy, xx = np.ogrid[:H, :W]
    uy, ux = direction_uv
    rmax = int(np.hypot(cy, cx)) - 2
    if harm_max is None:
        harm_max = int(rmax // max(r0, 1.0))

    mask = np.ones((H, W), dtype=np.float32)

    def zero_disk(mask, cy0, cx0, r):
        rr2 = (yy - cy0)**2 + (xx - cx0)**2 <= r*r
        mask[rr2] = 0.0

    for m in range(1, harm_max + 1):
        r = m * r0
        if r < guard_r or r > rmax:
            continue
        y = int(round(cy + r * uy))
        x = int(round(cx + r * ux))
        ys = 2*cy - y
        xs = 2*cx - x

        rad = int(round(notch_r0 + notch_scale * r))  # radio crece con el orden
        zero_disk(mask, y, x, rad)
        zero_disk(mask, ys, xs, rad)

    # protege bajas frecuencias (detalle/iluminación global)
    mask[_disk_mask(H, W, cy, cx, guard_r)] = 1.0
    return mask

def _denoise_periodic_any(img01,
                          PAD_FRAC=0.22, GUARD_R=18,
                          NOTCH_R0=6.0, NOTCH_SCALE=0.012,
                          HARM_MAX=None):
    """
    1) pad reflect
    2) FFT -> detectar pico fundamental y dirección
    3) máscara peineta (fundamental + armónicos)
    4) iFFT y recorte
    """
    H, W = img01.shape
    pad = max(16, int(PAD_FRAC * max(H, W)))
    Ipad = pad_reflect(img01, pad)

    X = fft2_shift(Ipad)
    mag = np.log1p(np.abs(X)).astype(np.float32)

    (uy, ux), r0 = _find_fundamental_peak(mag, guard_r=GUARD_R, num_candidates=80)
    notch = _build_notch_comb_mask(Ipad.shape, (uy, ux), r0,
                                   harm_max=HARM_MAX, guard_r=GUARD_R,
                                   notch_r0=NOTCH_R0, notch_scale=NOTCH_SCALE)

    Xf = X * notch
    rec = ifft2_unshift(Xf)
    rec = crop_center(rec, H, W, pad)
    return np.clip(rec, 0.0, 1.0)

def run_3b_gray_auto(path_in, path_out,
                     PAD_FRAC=0.22, GUARD_R=20,
                     NOTCH_R0=7.0, NOTCH_SCALE=0.013,
                     HARM_MAX=None):
    I = to_float01(np.array(Image.open(path_in).convert("L")))
    out = _denoise_periodic_any(I, PAD_FRAC, GUARD_R, NOTCH_R0, NOTCH_SCALE, HARM_MAX)
    save_img(path_out, out)

def run_3b_rgb_auto(path_in, path_out,
                    PAD_FRAC=0.22, GUARD_R=20,
                    NOTCH_R0=7.0, NOTCH_SCALE=0.013,
                    HARM_MAX=None):
    """Detecta dirección y r0 en luminancia y aplica la MISMA peineta a R,G,B."""
    rgb = to_float01(np.array(Image.open(path_in).convert("RGB")))
    H, W, _ = rgb.shape
    pad = max(16, int(PAD_FRAC * max(H, W)))

    # detección en luminancia
    Y = 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]
    Ypad = pad_reflect(Y, pad)
    Xy = fft2_shift(Ypad)
    mag = np.log1p(np.abs(Xy)).astype(np.float32)

    (uy, ux), r0 = _find_fundamental_peak(mag, guard_r=GUARD_R, num_candidates=80)
    notch = _build_notch_comb_mask(Ypad.shape, (uy, ux), r0,
                                   harm_max=HARM_MAX, guard_r=GUARD_R,
                                   notch_r0=NOTCH_R0, notch_scale=NOTCH_SCALE)

    out = np.empty_like(rgb)
    for c in range(3):
        Ic = pad_reflect(rgb[...,c], pad)
        Xc = fft2_shift(Ic)
        rec = crop_center(ifft2_unshift(Xc * notch), H, W, pad)
        out[...,c] = np.clip(rec, 0.0, 1.0)
    save_img(path_out, out)

# --- versión iterativa para el gato (re-detecta y aplica otra peineta) ---
def run_3b_rgb_auto_iter(path_in, path_out,
                         iters=2, PAD_FRAC=0.24, GUARD_R=22,
                         NOTCH_R0_START=8.5, NOTCH_R0_END=11.0,
                         NOTCH_SCALE=0.013, HARM_MAX=48):
    """
    Quita el patrón periódico en varias iteraciones.
    En cada iteración re-detecta dirección y r0 en la luminancia
    y aplica una peineta con radio levemente mayor.
    """
    rgb = to_float01(np.array(Image.open(path_in).convert("RGB")))
    H, W, _ = rgb.shape
    out = rgb.copy()

    for t in range(iters):
        # detección sobre el estado actual (luminancia)
        Y = 0.2126*out[...,0] + 0.7152*out[...,1] + 0.0722*out[...,2]
        pad = max(16, int(PAD_FRAC * max(H, W)))
        Ypad = pad_reflect(Y, pad)
        Xy = fft2_shift(Ypad)
        mag = np.log1p(np.abs(Xy)).astype(np.float32)

        (uy, ux), r0 = _find_fundamental_peak(mag, guard_r=GUARD_R, num_candidates=120)

        # radio de muesca: crece un poco en cada pasada
        if iters > 1:
            notch_r0 = NOTCH_R0_START + (NOTCH_R0_END - NOTCH_R0_START) * (t / (iters - 1))
        else:
            notch_r0 = NOTCH_R0_START

        notch = _build_notch_comb_mask(Ypad.shape, (uy, ux), r0,
                                       harm_max=HARM_MAX, guard_r=GUARD_R,
                                       notch_r0=notch_r0, notch_scale=NOTCH_SCALE)

        # aplicar peineta a los 3 canales
        new = np.empty_like(out)
        for c in range(3):
            Ic = pad_reflect(out[..., c], pad)
            Xc = fft2_shift(Ic)
            rec = crop_center(ifft2_unshift(Xc * notch), H, W, pad)
            new[..., c] = np.clip(rec, 0.0, 1.0)
        out = new

    save_img(path_out, out)

# =================== EJECUCIÓN por puntos ===================
ROOT = base_dir()

# nombres según tu carpeta
MIETTE_PATH = ROOT / "miette.jpg"
DUCK_PATH   = ROOT / "p_a_t_o.jpg"   # B/N con bandas verticales
BLINDS_PATH = ROOT / "g_a_t_o.png"   # bandas diagonales

# 3.a
run_3a(MIETTE_PATH, ROOT / "3.a.jpg", sigma_frac=0.06)

# 3.b.a (pato, gris). Si queda residuo, sube NOTCH_R0 a 8–10 o fija HARM_MAX=30
run_3b_gray_auto(DUCK_PATH, ROOT / "3.b.a.jpg",
                 PAD_FRAC=0.24, GUARD_R=22,
                 NOTCH_R0=8.0, NOTCH_SCALE=0.013, HARM_MAX=None)

# 3.b.b (gato) — dos pasadas, muescas algo más anchas y más armónicos
run_3b_rgb_auto_iter(BLINDS_PATH, ROOT / "3.b.b.png",
                     iters=2, PAD_FRAC=0.24, GUARD_R=22,
                     NOTCH_R0_START=8.5, NOTCH_R0_END=11.0,
                     NOTCH_SCALE=0.013, HARM_MAX=48)


#4




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






from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos
signals = np.load('taller_2/5.npy')  # o usar 'taller_2/reference.npy'
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
plt.savefig('taller_2/4.png', bbox_inches='tight', dpi=150)
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
plt.savefig('taller_2/comparacion_tomografia.png', bbox_inches='tight', dpi=150)
plt.close()

print("Reconstrucción tomográfica completada!")
print("Imagen guardada como '4.png'")
print("Comparación guardada como 'comparacion_tomografia.png'")
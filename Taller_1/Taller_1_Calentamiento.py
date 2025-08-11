# Librerías 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import UnivariateSpline



#Introduccion al experimento:
#Cuando una fuente de rayos X emite radiación, no todos los rayos tienen la misma energía. En vez de eso, se emiten 
#muchos fotones (partículas de luz) con diferentes niveles de energía.

#Para organizar los datos del experimento, la idea es la siguiente:

#Un diccionario general con 3 llaves ('Mo_unfiltered_10kV-50kV', 'Rh_unfiltered_10kV-50kV', 'W_unfiltered_10kV-50kV'). 

#Cada llave clasifica espectros de rayos X segun el elemento que se uso como anodo para la fuente de rayos X; 
#en este caso un tubo de rayos X.

#En cada llave, tres diccionario anidados para tres datos particulares del experimento y uno que representa un dataframe;
#este contiene 40 espectros de rayos X, cada uno corresponden a un voltaje particular.
#El voltaje oscila entre 10(kV) y 50(kV). 

#Recordemos que el espectro de rayos X es la gráfica o representación que muestra cuántos fotones hay (o la intensidad) 
#para cada nivel de energía.

#De esta manera el espectro de rayos X tendra coordenadas:
#(energy = energía, fluence = intensidad)
#Eje X: Nivel de energía de los fotones 
#Eje Y: Cantidad de fotones o intensidad para esa energía específica.

#De esta manera la estrucctura de datos que manejaremos será:
#datos = {
#    "Mo_unfiltered_10kV-50kV": {"dataframe": dataframe,
#                                "anode": anode,
#                                "anode_angle": anode_angle,
#                                "inherent_filtration": inherent_filtration},

#    "Rh_unfiltered_10kV-50kV": {"dataframe": dataframe,
#                                "anode": anode,
#                                "anode_angle": anode_angle,
#                                "inherent_filtration": inherent_filtration},

#    "W_unfiltered_10kV-50kV": {"dataframe": dataframe,
#                                "anode": anode,
#                                "anode_angle": anode_angle,
#                                "inherent_filtration": inherent_filtration}
#        }

#Los dataframe van a tener una estructura asi:
#df = pd.DataFrame({'energy ': energy,
#                    'fluence': fluence
#                 })


# --- Configuración ---
# Ruta base donde está la carpeta Taller_1
data_dir = os.path.join(os.getcwd(), 'Taller_1')

# Carpeta de datos por elemento
folders = [
    os.path.join(data_dir, "Mo_unfiltered_10kV-50kV"),
    os.path.join(data_dir, "Rh_unfiltered_10kV-50kV"),
    os.path.join(data_dir, "W_unfiltered_10kV-50kV")
]

# --- Interfaz: pedir niveles ---
# Esto facilita el analisis de datos al no tener que elegir unicamente 3 conjuntos de archivos
print("Selecciona 3 niveles de energía (kV) con los que quieres trabajar para cada elemento.")
print("Ejemplo: 10 20 30\n")
selected_kv = {}
for folder in folders:
    element_key = os.path.basename(folder)  # Ej: Mo_unfiltered_10kV-50kV
    kvs = input(f"Ingrese 3 niveles de kV para {element_key.split('_')[0]} separados por espacio o coma: ")
    selected_kv[element_key] = [f"{int(k.strip())}kV" for k in kvs.replace(",", " ").split()]

# --- Construcción de la base de datos ---
datos = {}

for folder in folders:
    element_key = os.path.basename(folder)
    element_name = element_key.split("_")[0]

    combined_df = pd.DataFrame()
    anode = None
    anode_angle = None
    inherent_filtration = None

    files_of_i = sorted(os.listdir(folder))

    for filename in files_of_i:
        if filename.endswith(".dat"):
            # Solo procesar si el archivo coincide con uno de los kV seleccionados
            if not any(kv in filename for kv in selected_kv[element_key]):
                continue

            ruta = os.path.join(folder, filename)

            with open(ruta, "r", encoding="latin1") as f:
                lines = f.readlines()

            if anode is None:
                anode = lines[4].split(":")[-1].strip()
                anode_angle = lines[6].split(":")[-1].strip()
                inherent_filtration = lines[7].split(":")[-1].strip()

            df_temp = pd.read_csv(
                ruta,
                sep=r"\s+",
                comment="#",
                header=None,
                encoding="latin1"
            )
            df_temp.columns = ["energy", "fluence"]
            df_temp["kv"] = filename.split("_")[1].replace(".dat", "")
            df_temp["filename"] = filename

            combined_df = pd.concat([combined_df, df_temp], ignore_index=True)

    datos[element_key] = {
        "dataframe": combined_df,
        "anode": anode,
        "anode_angle": anode_angle,
        "inherent_filtration": inherent_filtration
    }

# Esto lo puse para confirmar los datos que estuvieran bien, se puede quitar despues
for element_key, content in datos.items():
    print(f"\n=== {element_key} ===")
    print(content["dataframe"])
    print(f"\nMetadatos:")
    print(f"Anode: {content['anode']}")
    print(f"Anode Angle: {content['anode_angle']}")
    print(f"Inherent Filtration: {content['inherent_filtration']}")
    print(f"Total filas: {len(content['dataframe'])}")



#Punto 1


#Punto 2 - Comportamiento del continuo (Bremsstrahlung)

#2.a. Remover los picos
def remover_picos(df, altura_min=2.0, distancia_min=3, rel_height=0.8, ancho_max=10):
    fluencia = df["fluence"].values
    picos, props = find_peaks(fluencia, height=altura_min, distance=distancia_min)
    results_half = peak_widths(fluencia, picos, rel_height=rel_height)
    indices_a_eliminar = set()
    for i, pico in enumerate(picos):
        left = int(np.floor(results_half[2][i]))
        right = int(np.ceil(results_half[3][i]))
        # Solo eliminar si el ancho no es demasiado grande
        if (right - left) <= ancho_max:
            indices_a_eliminar.update(range(left, right + 1))
    indices_a_eliminar = [i for i in indices_a_eliminar if 0 <= i < len(df)]
    df_picos = df.iloc[indices_a_eliminar].copy()
    df_sin_picos = df.drop(index=df.index[indices_a_eliminar]).copy()
    return df_sin_picos, df_picos



# Crear pdf en la carpeta Taller_1
# Al usar el codigo plt.savefig("2.a.pdf", bbox_inches="tight", pad_inches=0.1) la imagen s egudraba por fuera de la carpeta, por lo cual primero creo el la direccion del archivo .pdf
dir_pdf = os.path.join(data_dir, "2.a.pdf")
fig, axs = plt.subplots(3, 3, figsize=(15, 10))  # 3 filas (elementos) x 3 columnas (kV)

with PdfPages(dir_pdf) as pdf:
    for element_key, content in datos.items():
        df_elemento = content["dataframe"].copy()

        # Obtener la lista de 3 kV en el orden deseado:
        # Si existe selected_kv en el entorno lo uso para mantener el orden de selección.
        kvs = None
        if 'selected_kv' in globals():
            # selected_kv puede tener claves con element_key o con el nombre corto (Mo). Probar ambas.
            if element_key in selected_kv:
                kvs = selected_kv[element_key]
            else:
                short = element_key.split('_')[0]
                if short in selected_kv:
                    kvs = selected_kv[short]
        # fallback: tomar los 3 kV únicos que aparezcan en el dataframe
        if kvs is None:
            kvs = list(df_elemento['kv'].unique())[:3]
        # asegurar longitud 3 (rellenar con None si faltan)
        while len(kvs) < 3:
            kvs.append(None)

        # Crear figura 3 filas x 3 columnas
        fig, axs = plt.subplots(3, 3, figsize=(15, 10))
        fig.suptitle(f"{element_key}", fontsize=16)

        for col_idx, kv in enumerate(kvs):
            # Si no hay kv (faltaron datos), apagamos esas columnas
            if kv is None:
                for r in range(3):
                    axs[r, col_idx].axis('off')
                continue

            # Extraer solo el espectro de ese kv
            df_kv = df_elemento[df_elemento["kv"] == kv].copy().reset_index(drop=True)
            if df_kv.empty:
                for r in range(3):
                    axs[r, col_idx].text(0.5, 0.5, f"No hay datos para {element_key.split('_')[0]}_{kv}",
                                         ha='center', va='center')
                    axs[r, col_idx].set_xticks([])
                    axs[r, col_idx].set_yticks([])
                continue

            # Obtener copia sin picos y DataFrame de picos (usando la función)
            df_sin, df_picos = remover_picos(df_kv, altura_min=2.0, distancia_min=3, rel_height=0.5)

            # FILA 1: original con picos marcados
            ax = axs[0, col_idx]
            ax.plot(df_kv['energy'], df_kv['fluence'], color='black', linewidth=0.8)
            if not df_picos.empty:
                ax.scatter(df_picos['energy'], df_picos['fluence'], color='red', s=20, zorder=5)
            ax.set_title(f"{element_key.split('_')[0]}_{kv}")
            if col_idx == 0:
                ax.set_ylabel("Original\nIntensidad")
            ax.set_xlabel("Energía (keV)")

            # FILA 2: solo sin picos
            ax = axs[1, col_idx]
            ax.plot(df_sin['energy'], df_sin['fluence'], color='blue', linewidth=0.9, label='Sin picos')
            if col_idx == 0:
                ax.set_ylabel("Sin picos\nIntensidad")
            ax.set_xlabel("Energía (keV)")
            if col_idx == 2:
                ax.legend(loc='upper right', fontsize=8)

            # Fila 3: ambas superpuestas (compa
            ax = axs[2, col_idx]
            ax.plot(df_kv['energy'], df_kv['fluence'], color='black', alpha=0.6, label='Original')
            ax.plot(df_sin['energy'], df_sin['fluence'], color='blue', label='Sin picos')
            if not df_picos.empty:
                ax.scatter(df_picos['energy'], df_picos['fluence'], color='red', s=12, label='Picos')
            if col_idx == 2:  # solo poner leyenda en la última columna para no sobrecargar
                ax.legend(loc='upper right')
            if col_idx == 0:
                ax.set_ylabel("Comparación\nIntensidad")
            ax.set_xlabel("Energía (keV)")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

print(f"PDF guardado en: {dir_pdf}")


#2.b. Aproximar el continuo
# 2.b. Aproximar el continuo y guardar los datos del ajuste spline
from scipy.interpolate import UnivariateSpline

def ajustar_continuo_spline(df_sin_picos, s=2):
    x = df_sin_picos["energy"].values
    y = df_sin_picos["fluence"].values
    spline = UnivariateSpline(x, y, s=s)
    y_fit = spline(x)
    return x, y_fit

dir_pdf_b = os.path.join(data_dir, "2.b.pdf")
ajustes_spline = {}  # <--- Aquí se guardarán los resultados

with PdfPages(dir_pdf_b) as pdf:
    for element_key, content in datos.items():
        df_elemento = content["dataframe"].copy()
        kvs = None
        if 'selected_kv' in globals():
            if element_key in selected_kv:
                kvs = selected_kv[element_key]
            else:
                short = element_key.split('_')[0]
                if short in selected_kv:
                    kvs = selected_kv[short]
        if kvs is None:
            kvs = list(df_elemento['kv'].unique())[:3]
        while len(kvs) < 3:
            kvs.append(None)

        ajustes_spline[element_key] = {}  # Inicializa para este elemento

        fig, axs = plt.subplots(3, 3, figsize=(15, 10))
        fig.suptitle(f"Ajuste del continuo - {element_key.split('_')[0]}", fontsize=16)

        for col_idx, kv in enumerate(kvs):
            if kv is None:
                for r in range(3):
                    axs[r, col_idx].axis('off')
                continue

            df_kv = df_elemento[df_elemento["kv"] == kv].copy().reset_index(drop=True)
            if df_kv.empty:
                for r in range(3):
                    axs[r, col_idx].text(0.5, 0.5, f"No hay datos para {element_key.split('_')[0]}_{kv}",
                                         ha='center', va='center')
                    axs[r, col_idx].set_xticks([])
                    axs[r, col_idx].set_yticks([])
                continue

            df_sin, _ = remover_picos(df_kv, altura_min=2.0, distancia_min=3, rel_height=0.5, ancho_max=10)
            x_fit, y_fit = ajustar_continuo_spline(df_sin, s=3)

            # Guarda los datos del ajuste spline para este elemento y kV
            ajustes_spline[element_key][kv] = {
                "energy": x_fit,
                "spline": y_fit,
                "df_sin": df_sin
            }

            # Fila 1: solo datos sin picos
            axs[0, col_idx].plot(df_sin["energy"], df_sin["fluence"], color="blue")
            axs[0, col_idx].set_title(f"{element_key.split('_')[0]}_{kv}\nSin picos")
            axs[0, col_idx].set_xlabel("Energía (keV)")
            axs[0, col_idx].set_ylabel("Intensidad")

            # Fila 2: solo ajuste
            axs[1, col_idx].plot(x_fit, y_fit, color="orange")
            axs[1, col_idx].set_title("Ajuste spline")
            axs[1, col_idx].set_xlabel("Energía (keV)")
            axs[1, col_idx].set_ylabel("Intensidad")

            # Fila 3: comparación
            axs[2, col_idx].plot(df_sin["energy"], df_sin["fluence"], color="blue", label="Sin picos")
            axs[2, col_idx].plot(x_fit, y_fit, color="orange", label="Ajuste")
            axs[2, col_idx].set_title("Comparación")
            axs[2, col_idx].set_xlabel("Energía (keV)")
            axs[2, col_idx].set_ylabel("Intensidad")
            axs[2, col_idx].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

print(f"PDF guardado en {dir_pdf_b}")


#2.c. Analizar el continuo 
def calcular_fwhm(x, y):
    y_max = np.max(y)
    half_max = y_max / 2
    # Encuentra los cruces con la media altura
    indices = np.where(y >= half_max)[0]
    if len(indices) < 2:
        return np.nan  # No se puede calcular FWHM
    left_idx = indices[0]
    right_idx = indices[-1]
    # Interpolación lineal para mayor precisión
    def interp(idx1, idx2):
        return x[idx1] + (half_max - y[idx1]) * (x[idx2] - x[idx1]) / (y[idx2] - y[idx1])
    # Buscar el cruce a la izquierda
    if left_idx > 0:
        x_left = interp(left_idx-1, left_idx)
    else:
        x_left = x[left_idx]
    # Buscar el cruce a la derecha
    if right_idx < len(x)-1:
        x_right = interp(right_idx, right_idx+1)
    else:
        x_right = x[right_idx]
    return x_right - x_left

# Diccionario para guardar resultados
resultados = {
    "elemento": [],
    "kv": [],
    "voltaje": [],
    "maximo": [],
    "energia_max": [],
    "fwhm": []
}

for element_key in ajustes_spline:
    elemento = element_key.split('_')[0]
    for kv in ajustes_spline[element_key]:
        ajuste = ajustes_spline[element_key][kv]
        x = ajuste["energy"]
        y = ajuste["spline"]
        if len(x) == 0 or len(y) == 0:
            continue
        idx_max = np.argmax(y)
        maximo = y[idx_max]
        energia_max = x[idx_max]
        fwhm = calcular_fwhm(x, y)
        resultados["elemento"].append(elemento)
        resultados["kv"].append(kv)
        resultados["voltaje"].append(float(kv.replace("kV", "")))
        resultados["maximo"].append(maximo)
        resultados["energia_max"].append(energia_max)
        resultados["fwhm"].append(fwhm)

# Convertir a DataFrame para graficar
df_res = pd.DataFrame(resultados)

# Graficar
dir_pdf_c = os.path.join(data_dir, "2.c.pdf")
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
colores = {"Mo": "tab:blue", "Rh": "tab:orange", "W": "tab:green"}

for elemento in df_res["elemento"].unique():
    df_e = df_res[df_res["elemento"] == elemento]
    axs[0,0].plot(df_e["voltaje"], df_e["maximo"], marker='o', label=elemento, color=colores[elemento])
    axs[0,1].plot(df_e["voltaje"], df_e["energia_max"], marker='o', label=elemento, color=colores[elemento])
    axs[1,0].plot(df_e["voltaje"], df_e["fwhm"], marker='o', label=elemento, color=colores[elemento])
    axs[1,1].plot(df_e["energia_max"], df_e["maximo"], marker='o', label=elemento, color=colores[elemento])

axs[0,0].set_title("Máximo del continuo vs Voltaje")
axs[0,0].set_xlabel("Voltaje (kV)")
axs[0,0].set_ylabel("Máximo del continuo")
axs[0,0].legend()

axs[0,1].set_title("Energía del máximo vs Voltaje")
axs[0,1].set_xlabel("Voltaje (kV)")
axs[0,1].set_ylabel("Energía del máximo (keV)")
axs[0,1].legend()

axs[1,0].set_title("FWHM vs Voltaje")
axs[1,0].set_xlabel("Voltaje (kV)")
axs[1,0].set_ylabel("FWHM (keV)")
axs[1,0].legend()

axs[1,1].set_title("Máximo del continuo vs Energía del máximo")
axs[1,1].set_xlabel("Energía del máximo (keV)")
axs[1,1].set_ylabel("Máximo del continuo")
axs[1,1].legend()

plt.tight_layout()
plt.savefig(dir_pdf_c, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
print(f"PDF guardado en {dir_pdf_c}")


#Mañana optimizo el punto 2 y coninuo con el bono




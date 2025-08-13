# Librerías 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import Rbf
from scipy.signal import savgol_filter



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

# Intefaz de selección de kV
# Esto facilita el uso del programa dado que se puedene stuidar varios niveles de energía (kV) para cada elemento.
print("Selecciona 3 niveles de energía (kV) con los que quieres trabajar para las gráficas.")
print("Ejemplo: 10 20 30\n")
selected_kv = {}
for folder in folders:
    element_key = os.path.basename(folder)  # Ej: Mo_unfiltered_10kV-50kV
    kvs = input(f"Ingrese 3 niveles de kV para {element_key.split('_')[0]} separados por espacio o coma: ")
    selected_kv[element_key] = [f"{int(k.strip())}kV" for k in kvs.replace(",", " ").split()]

# --- Construcción de la base de datos con TODOS los archivos ---
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


#Punto 1


#Punto 2 - Comportamiento del continuo (Bremsstrahlung)

# 2.a. Remover los picos
datos_sin_picos = [] # Creamos un Dataframe de los datos sin picos para ser usados en el punto B y posteriores

def remover_picos(df, altura_min=2.0, distancia_min=3, rel_height=0.8, ancho_max=10):
    fluencia = df["fluence"].values
    picos, _ = find_peaks(fluencia, height=altura_min, distance=distancia_min)
    results_half = peak_widths(fluencia, picos, rel_height=rel_height)
    indices_a_eliminar = set()
    for i, picos in enumerate(picos):
        left = int(np.floor(results_half[2][i]))
        right = int(np.ceil(results_half[3][i]))
        if (right - left) <= ancho_max:  # eliminar solo si ancho <= ancho_max
            indices_a_eliminar.update(range(left, right + 1))
    indices_a_eliminar = [i for i in indices_a_eliminar if 0 <= i < len(df)]
    df_picos = df.iloc[indices_a_eliminar].copy()
    df_sin_picos = df.drop(index=df.index[indices_a_eliminar]).copy()
    return df_sin_picos, df_picos


dir_pdf = os.path.join(data_dir, "2.a.pdf")

#La funcion PDFPages permite guardar varias graficas en un solo archivo PDF.
with PdfPages(dir_pdf) as pdf:
    for element_key, content in datos.items():
        df_elemento = content["dataframe"].copy()
        kvs = selected_kv.get(element_key, list(df_elemento['kv'].unique())[:3])
        while len(kvs) < 3:
            kvs.append(None)

        fig, axs = plt.subplots(3, 3, figsize=(15, 10))
        fig.suptitle(f"{element_key}", fontsize=16)

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

            df_sin, df_picos = remover_picos(df_kv, altura_min=2.0, distancia_min=3, rel_height=0.5, ancho_max=10)

            # Guardar en registros_sin_picos
            for i, row in df_sin.iterrows():
                datos_sin_picos.append({
                    "elemento": element_key.split('_')[0],
                    "kv": kv,
                    "energy": row["energy"],
                    "fluence_sin_picos": row["fluence"]
                })

            # FILA 1: original con picos
            ax = axs[0, col_idx]
            ax.plot(df_kv['energy'], df_kv['fluence'], color='black', linewidth=0.8)
            if not df_picos.empty:
                ax.scatter(df_picos['energy'], df_picos['fluence'], color='red', s=20, zorder=5)
            ax.set_title(f"{element_key.split('_')[0]}_{kv}")
            if col_idx == 0:
                ax.set_ylabel("Original\nIntensidad")
            ax.set_xlabel("Energía (keV)")

            # FILA 2: Datos sin picos
            ax = axs[1, col_idx]
            ax.plot(df_sin['energy'], df_sin['fluence'], color='blue', linewidth=0.9, label='Sin picos')
            if col_idx == 0:
                ax.set_ylabel("Sin picos\nIntensidad")
            ax.set_xlabel("Energía (keV)")
            if col_idx == 2:
                ax.legend(fontsize=8)

            # FILA 3: comparación de datos
            ax = axs[2, col_idx]
            ax.plot(df_kv['energy'], df_kv['fluence'], color='black', alpha=0.6, label='Original')
            ax.plot(df_sin['energy'], df_sin['fluence'], color='blue', label='Sin picos')
            if not df_picos.empty:
                ax.scatter(df_picos['energy'], df_picos['fluence'], color='red', s=12, label='Picos')
            if col_idx == 2:
                ax.legend()
            if col_idx == 0:
                ax.set_ylabel("Comparación\nIntensidad")
            ax.set_xlabel("Energía (keV)")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

# DataFrame con todos los datos sin picos
df_sin_picos = pd.DataFrame(datos_sin_picos)


#2.b. Aproximar el continuo
# -------- 2.b. Aproximar el continuo (mismo FORMATO que 2.a) --------
def aproximacion_rbf(df_sin_picos, function='quintic', smooth=5, smooth_window=9, polyorder=1):
    x = df_sin_picos["energy"].values
    y = df_sin_picos["fluence_sin_picos"].values

    # Suavizado previo para evitar que el ruido distorsione el RBF
    if len(y) >= smooth_window:
        y_suave = savgol_filter(y, smooth_window, polyorder)
    else:
        y_suave = y

    rbf = Rbf(x, y_suave, function=function, smooth=smooth)
    y_fit = rbf(x)
    return x, y_fit

datos_continuo = []  # aquí sí guardamos el continuo calculado
dir_pdf_b = os.path.join(data_dir, "2.b.pdf")

with PdfPages(dir_pdf_b) as pdf:
    # Recorremos con la MISMA clave que en 2.a para que coincidan columnas/orden
    for element_key, content in datos.items():
        elemento_corto = element_key.split('_')[0]
        # usar selected_kv con la MISMA clave 'element_key' (no con 'Mo/Rh/W')
        kvs = selected_kv.get(
            element_key,
            list(df_sin_picos[df_sin_picos["elemento"] == elemento_corto]["kv"].unique())[:3]
        )
        while len(kvs) < 3:
            kvs.append(None)

        fig, axs = plt.subplots(3, 3, figsize=(15, 10))
        # Supertítulo con el mismo tamaño que 2.a (puedes dejar el texto que prefieras)
        fig.suptitle(f"Ajuste del continuo (RBF) - {elemento_corto}", fontsize=16)

        for col_idx, kv in enumerate(kvs):
            if kv is None:
                for r in range(3):
                    axs[r, col_idx].axis('off')
                continue

            # Filtramos del DF "2.a" ya sin picos
            df_sin = df_sin_picos[
                (df_sin_picos["elemento"] == elemento_corto) & (df_sin_picos["kv"] == kv)
            ]
            if df_sin.empty:
                for r in range(3):
                    axs[r, col_idx].text(0.5, 0.5, f"No hay datos para {elemento_corto}_{kv}",
                                         ha='center', va='center')
                    axs[r, col_idx].set_xticks([])
                    axs[r, col_idx].set_yticks([])
                continue

            x_fit, y_fit = aproximacion_rbf(df_sin)

            # --- Guardar continuo para análisis posteriores ---
            for xi, yi, yorig in zip(df_sin["energy"].values, y_fit, df_sin["fluence_sin_picos"].values):
                datos_continuo.append({
                    "elemento": elemento_corto,
                    "kv": kv,
                    "energy": xi,
                    "fluence_sin_picos": yorig,
                    "fluence_continuo": yi
                })

            # --- FILA 1: datos sin picos ---
            ax = axs[0, col_idx]
            ax.plot(df_sin["energy"], df_sin["fluence_sin_picos"], color="blue")
            ax.set_title(f"{elemento_corto}_{kv}")  # igual que 2.a: título solo en fila 1
            if col_idx == 0:
                ax.set_ylabel("Sin picos\nIntensidad")
            ax.set_xlabel("Energía (keV)")

            # --- FILA 2: solo continuo ---
            ax = axs[1, col_idx]
            ax.plot(df_sin["energy"], y_fit, color="orange", label="Continuo")
            if col_idx == 0:
                ax.set_ylabel("Continuo (RBF)\nIntensidad")
            ax.set_xlabel("Energía (keV)")
            if col_idx == 2:
                ax.legend(fontsize=8)

            # --- FILA 3: comparación (dos curvas sobrepuestas) ---
            ax = axs[2, col_idx]
            ax.plot(df_sin["energy"], df_sin["fluence_sin_picos"], color="blue", label="Sin picos")
            ax.plot(df_sin["energy"], y_fit, color="orange", label="Continuo")
            if col_idx == 2:
                ax.legend()
            if col_idx == 0:
                ax.set_ylabel("Comparación\nIntensidad")
            ax.set_xlabel("Energía (keV)")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

# DataFrame con los datos de la aproximacion continua
datos_continuo = pd.DataFrame(datos_continuo)



#2.c. Analizar el continuo 
def calcular_fwhm(x, y):
    y_max = np.max(y)
    half_max = y_max / 2
    indices = np.where(y >= half_max)[0]
    if len(indices) < 2:
        return np.nan
    left_idx = indices[0]
    right_idx = indices[-1]

    def interp(idx1, idx2):
        return x[idx1] + (half_max - y[idx1]) * (x[idx2] - x[idx1]) / (y[idx2] - y[idx1])

    x_left = interp(left_idx - 1, left_idx) if left_idx > 0 else x[left_idx]
    x_right = interp(right_idx, right_idx + 1) if right_idx < len(x) - 1 else x[right_idx]

    return x_right - x_left

#Creamos un diccionario para los datos a calcular
resultados = {
    "elemento": [],
    "kv": [],
    "voltaje": [],
    "maximo": [],
    "energia_max": [],
    "fwhm": []
}

for (elemento, kv), df_kv in datos_continuo.groupby(["elemento", "kv"]):
    x = df_kv["energy"].values
    y = df_kv["fluence_continuo"].values  # ajuste RBF
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

# Convertir a DataFrame con todos los cálculos
df_res = pd.DataFrame(resultados)

# Crear un mapeo de claves cortas a kV seleccionados en la interfaz
selected_kv_short = {}
for full_key, kv_list in selected_kv.items():
    short = full_key.split("_")[0]  
    selected_kv_short[short] = kv_list

# Graficar SOLO selected_kv 
dir_pdf_c = os.path.join(data_dir, "2.c.pdf")
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
colores = {"Mo": "tab:blue", "Rh": "tab:orange", "W": "tab:green"}

for elemento in df_res["elemento"].unique():
    # Filtrar solo los kV seleccionados para las gráficas
    kvs_graf = selected_kv_short.get(elemento, [])
    df_e = df_res[(df_res["elemento"] == elemento) & (df_res["kv"].isin(kvs_graf))]

    axs[0, 0].plot(df_e["voltaje"], df_e["maximo"], marker='o', label=elemento, color=colores[elemento])
    axs[0, 1].plot(df_e["voltaje"], df_e["energia_max"], marker='o', label=elemento, color=colores[elemento])
    axs[1, 0].plot(df_e["voltaje"], df_e["fwhm"], marker='o', label=elemento, color=colores[elemento])
    axs[1, 1].plot(df_e["energia_max"], df_e["maximo"], marker='o', label=elemento, color=colores[elemento])

# Etiquetas y leyendas
axs[0, 0].set_title("Máximo del continuo vs Voltaje")
axs[0, 0].set_xlabel("Voltaje (kV)")
axs[0, 0].set_ylabel("Máximo del continuo")
axs[0, 0].legend()

axs[0, 1].set_title("Energía del máximo vs Voltaje")
axs[0, 1].set_xlabel("Voltaje (kV)")
axs[0, 1].set_ylabel("Energía del máximo (keV)")
axs[0, 1].legend()

axs[1, 0].set_title("FWHM vs Voltaje")
axs[1, 0].set_xlabel("Voltaje (kV)")
axs[1, 0].set_ylabel("FWHM (keV)")
axs[1, 0].legend()

axs[1, 1].set_title("Máximo del continuo vs Energía del máximo")
axs[1, 1].set_xlabel("Energía del máximo (keV)")
axs[1, 1].set_ylabel("Máximo del continuo")
axs[1, 1].legend()

plt.tight_layout()
plt.savefig(dir_pdf_c, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)



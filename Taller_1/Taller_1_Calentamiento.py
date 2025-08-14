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
    for i, pico in enumerate(picos):
        left = int(np.floor(results_half[2][i]))
        right = int(np.ceil(results_half[3][i]))
        if (right - left) <= ancho_max:  # eliminar solo si ancho <= ancho_max
            indices_a_eliminar.update(range(left, right + 1))
    indices_a_eliminar = [i for i in indices_a_eliminar if 0 <= i < len(df)]
    df_picos = df.iloc[indices_a_eliminar].copy()
    df_sin_picos = df.drop(index=df.index[indices_a_eliminar]).copy()
    return df_sin_picos, df_picos

dir_pdf = os.path.join(data_dir, "2.a.pdf")

with PdfPages(dir_pdf) as pdf:
    for element_key, content in datos.items():
        df_elemento = content["dataframe"].copy()
        elemento_corto = element_key.split('_')[0]

        # --- Guardar TODOS los kV en datos_sin_picos ---
        for kv_total in df_elemento['kv'].unique():
            df_kv_total = df_elemento[df_elemento["kv"] == kv_total].copy().reset_index(drop=True)
            if not df_kv_total.empty:
                df_sin_total, _ = remover_picos(df_kv_total, altura_min=2.0, distancia_min=3, rel_height=0.5, ancho_max=10)
                for i, row in df_sin_total.iterrows():
                    datos_sin_picos.append({
                        "elemento": elemento_corto,
                        "kv": kv_total,
                        "energy": row["energy"],
                        "fluence_sin_picos": row["fluence"]
                    })

        # --- Solo graficar los seleccionados ---
        kvs = selected_kv.get(element_key, list(df_elemento['kv'].unique())[:3])
        while len(kvs) < 3:
            kvs.append(None)

        fig, axs = plt.subplots(3, 3, figsize=(15, 10))
        fig.suptitle(f"{elemento_corto}", fontsize=16)

        for col_idx, kv in enumerate(kvs):
            if kv is None:
                for r in range(3):
                    axs[r, col_idx].axis('off')
                continue

            df_kv = df_elemento[df_elemento["kv"] == kv].copy().reset_index(drop=True)
            if df_kv.empty:
                for r in range(3):
                    axs[r, col_idx].text(0.5, 0.5, f"No hay datos para {elemento_corto}_{kv}",
                                         ha='center', va='center')
                    axs[r, col_idx].set_xticks([])
                    axs[r, col_idx].set_yticks([])
                continue

            df_sin, df_picos = remover_picos(df_kv, altura_min=2.0, distancia_min=3, rel_height=0.5, ancho_max=10)

            # Fila 1: original con picos
            ax = axs[0, col_idx]
            ax.plot(df_kv['energy'], df_kv['fluence'], color='black', linewidth=0.8)
            if not df_picos.empty:
                ax.scatter(df_picos['energy'], df_picos['fluence'], color='red', s=20, zorder=5)
            ax.set_title(f"{elemento_corto}_{kv}")
            if col_idx == 0:
                ax.set_ylabel("Original\nIntensidad")
            ax.set_xlabel("Energía (keV)")

            # Fila 2: datos sin picos
            ax = axs[1, col_idx]
            ax.plot(df_sin['energy'], df_sin['fluence'], color='blue', linewidth=0.9, label='Sin picos')
            if col_idx == 0:
                ax.set_ylabel("Sin picos\nIntensidad")
            ax.set_xlabel("Energía (keV)")
            if col_idx == 2:
                ax.legend(fontsize=8)

            # Fila 3: comparación
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

# DataFrame con todos los datos sin picos (TODOS los kV)
df_sin_picos = pd.DataFrame(datos_sin_picos)

#2.b. Aproximar el continuo
def aproximacion_rbf(df_sin_picos, function='quintic', smooth=5, smooth_window=9, polyorder=1):
    x = df_sin_picos["energy"].values
    y = df_sin_picos["fluence_sin_picos"].values

    if len(y) >= smooth_window:
        y_suave = savgol_filter(y, smooth_window, polyorder)
    else:
        y_suave = y

    rbf = Rbf(x, y_suave, function=function, smooth=smooth)
    y_fit = rbf(x)
    return x, y_fit

datos_continuo = []  # Guardará continuo de TODOS los kV
dir_pdf_b = os.path.join(data_dir, "2.b.pdf")

with PdfPages(dir_pdf_b) as pdf:
    for element_key, content in datos.items():
        elemento_corto = element_key.split('_')[0]

        # --- Guardar TODOS los kV en datos_continuo ---
        for kv_total in df_sin_picos[df_sin_picos["elemento"] == elemento_corto]["kv"].unique():
            df_sin_total = df_sin_picos[(df_sin_picos["elemento"] == elemento_corto) & (df_sin_picos["kv"] == kv_total)]
            if not df_sin_total.empty:
                x_fit, y_fit = aproximacion_rbf(df_sin_total)
                for xi, yi, yorig in zip(df_sin_total["energy"].values, y_fit, df_sin_total["fluence_sin_picos"].values):
                    datos_continuo.append({
                        "elemento": elemento_corto,
                        "kv": kv_total,
                        "energy": xi,
                        "fluence_sin_picos": yorig,
                        "fluence_continuo": yi
                    })

        # --- Solo graficar los seleccionados ---
        kvs = selected_kv.get(
            element_key,
            list(df_sin_picos[df_sin_picos["elemento"] == elemento_corto]["kv"].unique())[:3]
        )
        while len(kvs) < 3:
            kvs.append(None)

        fig, axs = plt.subplots(3, 3, figsize=(15, 10))
        fig.suptitle(f"Ajuste del continuo (RBF) - {elemento_corto}", fontsize=16)

        for col_idx, kv in enumerate(kvs):
            if kv is None:
                for r in range(3):
                    axs[r, col_idx].axis('off')
                continue

            df_sin = df_sin_picos[(df_sin_picos["elemento"] == elemento_corto) & (df_sin_picos["kv"] == kv)]
            if df_sin.empty:
                for r in range(3):
                    axs[r, col_idx].text(0.5, 0.5, f"No hay datos para {elemento_corto}_{kv}",
                                         ha='center', va='center')
                    axs[r, col_idx].set_xticks([])
                    axs[r, col_idx].set_yticks([])
                continue

            x_fit, y_fit = aproximacion_rbf(df_sin)

            # Fila 1: datos sin picos
            ax = axs[0, col_idx]
            ax.plot(df_sin["energy"], df_sin["fluence_sin_picos"], color="blue")
            ax.set_title(f"{elemento_corto}_{kv}")
            if col_idx == 0:
                ax.set_ylabel("Sin picos\nIntensidad")
            ax.set_xlabel("Energía (keV)")

            # Fila 2: solo continuo
            ax = axs[1, col_idx]
            ax.plot(df_sin["energy"], y_fit, color="orange", label="Continuo")
            if col_idx == 0:
                ax.set_ylabel("Continuo (RBF)\nIntensidad")
            ax.set_xlabel("Energía (keV)")
            if col_idx == 2:
                ax.legend(fontsize=8)

            # Fila 3: comparación
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

# DataFrame con todos los datos de la aproximación continua (TODOS los kV)
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

# 
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
    y = df_kv["fluence_continuo"].values
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

# Convertir resultados a DataFrame
df_res = pd.DataFrame(resultados)

# Graficar
dir_pdf_c = os.path.join(data_dir, "2.c.pdf")
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
colores = {"Mo": "tab:blue", "Rh": "tab:orange", "W": "tab:green"}

for elemento in df_res["elemento"].unique():
    df_e = df_res[df_res["elemento"] == elemento].sort_values("voltaje")

    axs[0, 0].plot(df_e["voltaje"], df_e["maximo"], marker='o', label=elemento, color=colores[elemento])
    axs[0, 1].plot(df_e["voltaje"], df_e["energia_max"], marker='o', label=elemento, color=colores[elemento])
    axs[1, 0].plot(df_e["voltaje"], df_e["fwhm"], marker='o', label=elemento, color=colores[elemento])
    axs[1, 1].plot(df_e["energia_max"], df_e["maximo"], marker='o', label=elemento, color=colores[elemento])

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


#Punto 3 - Picos (Rayos-X característicos)

# 3.a. Aislar los picos
def aislar_picos(datos_originales, datos_continuo):
    """
    Resta el continuo del espectro original para obtener solo los picos
    """
    datos_picos_aislados = []
    
    for elemento in datos_continuo["elemento"].unique():
        for kv in datos_continuo[datos_continuo["elemento"] == elemento]["kv"].unique():
            # Obtener datos originales
            element_key = f"{elemento}_unfiltered_10kV-50kV"
            df_original = datos_originales[element_key]["dataframe"]
            df_orig_kv = df_original[df_original["kv"] == kv].copy().reset_index(drop=True)
            
            # Obtener continuo calculado
            df_cont = datos_continuo[
                (datos_continuo["elemento"] == elemento) & 
                (datos_continuo["kv"] == kv)
            ].copy()
            
            if df_orig_kv.empty or df_cont.empty:
                continue
                
            # Hacer merge por energía para restar continuo
            df_merged = pd.merge(df_orig_kv, df_cont, on="energy", how="inner")
            df_merged["fluence_picos"] = df_merged["fluence"] - df_merged["fluence_continuo"]
            
            # Filtrar solo valores positivos (picos reales)
            df_merged["fluence_picos"] = np.maximum(df_merged["fluence_picos"], 0)
            
            for _, row in df_merged.iterrows():
                datos_picos_aislados.append({
                    "elemento": elemento,
                    "kv": kv,
                    "energy": row["energy"],
                    "fluence_picos": row["fluence_picos"]
                })
    
    return pd.DataFrame(datos_picos_aislados)

# Obtener picos aislados
df_picos_aislados = aislar_picos(datos, datos_continuo)

# Graficar 3.a - Picos aislados con zoom
# Graficar 3.a - Picos aislados con zoom SOLO para los kV seleccionados
dir_pdf_3a = os.path.join(data_dir, "3.a.pdf")

with PdfPages(dir_pdf_3a) as pdf:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    colores_kv = plt.cm.viridis(np.linspace(0, 1, 10))  # Paleta de colores para diferentes kV

    for col_idx, elemento in enumerate(["Mo", "Rh", "W"]):
        ax = axs[col_idx]
        element_key = f"{elemento}_unfiltered_10kV-50kV"
        # Solo los kV seleccionados en la interfaz
        kvs_seleccionados = selected_kv[element_key]
        df_elemento = df_picos_aislados[df_picos_aislados["elemento"] == elemento]
        
        # Encontrar el rango de energías donde hay picos significativos SOLO para los kV seleccionados
        energias_con_picos = []
        for kv in kvs_seleccionados:
            df_kv = df_elemento[df_elemento["kv"] == kv]
            if not df_kv.empty:
                max_fluence = df_kv["fluence_picos"].max()
                if max_fluence > 0:
                    indices_significativos = df_kv[df_kv["fluence_picos"] > 0.05 * max_fluence]
                    if not indices_significativos.empty:
                        energias_con_picos.extend(indices_significativos["energy"].tolist())

        if energias_con_picos:
            min_energy = max(0, min(energias_con_picos) - 5)
            max_energy = min(60, max(energias_con_picos) + 5)
        else:
            min_energy, max_energy = 0, 30

        # Graficar solo los kV seleccionados
        for i, kv in enumerate(kvs_seleccionados):
            df_kv = df_elemento[df_elemento["kv"] == kv]
            if not df_kv.empty:
                df_zoom = df_kv[(df_kv["energy"] >= min_energy) & (df_kv["energy"] <= max_energy)]
                color_idx = min(i, len(colores_kv) - 1)
                ax.plot(df_zoom["energy"], df_zoom["fluence_picos"],
                        label=kv, color=colores_kv[color_idx], linewidth=1.2)

        ax.set_title(f"Picos aislados - {elemento}")
        ax.set_xlabel("Energía (keV)")
        ax.set_ylabel("Intensidad (picos)")
        ax.set_xlim(min_energy, max_energy)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

# 3.b. Ajustar función gaussiana a los picos principales
from scipy.optimize import curve_fit

def gaussiana(x, amplitud, centro, sigma):
    return amplitud * np.exp(-((x - centro) ** 2) / (2 * sigma ** 2))

def encontrar_y_ajustar_pico_principal(df_kv, altura_min=0.1, distancia_min=5):
    x = df_kv["energy"].values
    y = df_kv["fluence_picos"].values

    if len(x) < 5 or np.max(y) < altura_min:
        return None

    picos, propiedades = find_peaks(y, height=altura_min, distance=distancia_min)
    if len(picos) == 0:
        return None

    alturas = propiedades["peak_heights"]
    idx_pico_principal = picos[np.argmax(alturas)]
    altura_principal = y[idx_pico_principal]
    energia_principal = x[idx_pico_principal]

    ventana = 10  # keV
    mask = (x >= energia_principal - ventana) & (x <= energia_principal + ventana)
    x_ajuste = x[mask]
    y_ajuste = y[mask]

    if len(x_ajuste) < 3:
        return None

    try:
        amplitud_inicial = altura_principal
        centro_inicial = energia_principal
        sigma_inicial = 2.0

        popt, pcov = curve_fit(
            gaussiana,
            x_ajuste,
            y_ajuste,
            p0=[amplitud_inicial, centro_inicial, sigma_inicial],
            maxfev=5000
        )

        amplitud, centro, sigma = popt
        fwhm = 2.355 * abs(sigma)

        # FILTRO: amplitud no debe ser mayor que 5x el máximo real del espectro
        # Se desarollo este filtro dado que en el elemento de Rh se estaban presnetando Alturas de Pico de 1e6 lo que apalstaba los valores de los demas elementos
        if abs(centro - energia_principal) > 5 or fwhm > 20 or amplitud <= 0 or amplitud > 5 * np.max(y):
            return None

        return {
            "amplitud": amplitud,
            "centro": centro,
            "sigma": abs(sigma),
            "fwhm": fwhm,
            "x_ajuste": x_ajuste,
            "y_ajuste": y_ajuste,
            "y_fit": gaussiana(x_ajuste, *popt)
        }

    except Exception:
        return None

resultados_ajustes = {
    "elemento": [],
    "kv": [],
    "voltaje": [],
    "altura_pico": [],
    "fwhm_pico": [],
    "posicion_pico": []
}

ajustes_exitosos = {}

for elemento in df_picos_aislados["elemento"].unique():
    df_elemento = df_picos_aislados[df_picos_aislados["elemento"] == elemento]
    for kv in df_elemento["kv"].unique():
        df_kv = df_elemento[df_elemento["kv"] == kv]
        if df_kv.empty:
            continue
        resultado = encontrar_y_ajustar_pico_principal(df_kv)
        if resultado is not None:
            voltaje_num = float(kv.replace("kV", ""))
            if voltaje_num >= 15:
                resultados_ajustes["elemento"].append(elemento)
                resultados_ajustes["kv"].append(kv)
                resultados_ajustes["voltaje"].append(voltaje_num)
                resultados_ajustes["altura_pico"].append(resultado["amplitud"])
                resultados_ajustes["fwhm_pico"].append(resultado["fwhm"])
                resultados_ajustes["posicion_pico"].append(resultado["centro"])
                ajustes_exitosos[f"{elemento}_{kv}"] = resultado

df_ajustes = pd.DataFrame(resultados_ajustes)

# Graficar 3.b - Resultados de los ajustes
dir_pdf_3b = os.path.join(data_dir, "3.b.pdf")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
colores = {"Mo": "tab:blue", "Rh": "tab:orange", "W": "tab:green"}

for elemento in df_ajustes["elemento"].unique():
    df_e = df_ajustes[df_ajustes["elemento"] == elemento]
    axs[0].plot(df_e["voltaje"], df_e["altura_pico"],
                marker='o', label=elemento, color=colores[elemento], linewidth=2)
    axs[1].plot(df_e["voltaje"], df_e["fwhm_pico"],
                marker='o', label=elemento, color=colores[elemento], linewidth=2)

axs[0].set_title("Altura del pico principal vs Voltaje del tubo")
axs[0].set_xlabel("Voltaje (kV)")
axs[0].set_ylabel("Altura del pico")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

axs[1].set_title("FWHM del pico principal vs Voltaje del tubo")
axs[1].set_xlabel("Voltaje (kV)")
axs[1].set_ylabel("FWHM (keV)")
axs[1].legend()
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(dir_pdf_3b, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)



import os as os
import pandas as pd

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


data_dictionary = {}
data_dir = os.path.dirname(os.path.realpath('__file__')) + '/Taller_1/'
folders = [data_dir + "Mo_unfiltered_10kV-50kV",data_dir + "Rh_unfiltered_10kV-50kV", data_dir + "W_unfiltered_10kV-50kV"]
for i in folders:
    data_dictionary[i[90:]] = {}
    files_of_i = os.listdir(i)  # Devuelve una lista con los nombres de todos los archivos ["Elemento_1", "Elemento_2",...,"Elemento_n"]
                # files_of_i ["Mo_10kV.dat", "Mo_11kV.dat",..., "Mo_50kV.dat"]
    files_of_i.sort() #Organizar la lista anterior

    for j in files_of_i:
        if j.endswith(".dat"):  # Mira la cadena de str j y pregunta si termina en ".dat"
            ruta = os.path.join(i, j) # une varias partes de una ruta de archivo en una sola cadena, utilizando el separador correcto del sistema operativo (para Windows es \). Esto con el fin de poder tener las coordenadas exactas de cada uno de los archivos
            #i = "Mo_unfiltered_10kV-50kV"
            #j = "Mo_10kV.dat"
            #ruta = os.path.join(i, j) = "Mo_unfiltered_10kV-50kV\Mo_25kV.dat"

            # Abrir archivo para extraer datos de líneas específicas
            with open(ruta, "r") as f:
                lines = f.readlines()

            # Extraer datos de líneas específicas (considerando que la primera línea es lines[0])
            # Por ejemplo:
            # Línea 4 (índice 4) es la 5ta línea
            anode_line = lines[4].strip()
            anode = anode_line.split(":")[-1].strip()  

            anode_angle_line = lines[6].strip()
            anode_angle = anode_angle_line.split(":")[-1].strip()

            inherent_filtration_line = lines[7].strip()
            inherent_filtration = inherent_filtration_line.split(":")[-1].strip()

            dataframe = pd.read_csv (ruta, 
                        sep=r"\s+",        # separador es uno o más espacios en blanco
                        comment="#",       # ignora líneas que empiezan con #
                        header=None,       # no hay encabezado en los datos
                        encoding="latin1")  # para que no dé error con el símbolo °
        
            dataframe.columns = ["energy", "fluence"]

            data_dictionary[i[90:]][j] = {"dataframe": dataframe,
                                    "anode": anode,
                                    "anode_angle": anode_angle,
                                    "inherent_filtration": inherent_filtration
                                    }

#Prints para tener mas claro que es lo que esta arrojando el codigo


#print(data_dictionary["W_unfiltered_10kV-50kV"]["W_17kV.dat"])
print(data_dictionary["W_unfiltered_10kV-50kV"]["W_17kV.dat"]["inherent_filtration"])






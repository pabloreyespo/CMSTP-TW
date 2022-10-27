from statistics import mean
import pandas as pd

def rescatar(input, output):
    with open(input, "r") as file:
        lineas = file.readlines()
        lineas = [i.rstrip() for i in lineas]

    resultados = []

    for i in range(56 * 5):
        instancia_carga = lineas[i*11].replace(".txt", "")
        instancia_carga = instancia_carga.replace("Q", "")
        instancia, Q = instancia_carga.split("-")

        vals = []
        times= []
        best_its = []

        for j in range(10):
            datos  = lineas[i * 11 + j + 1]
            datos = datos.split()

            vals.append(float(datos[4]))
            times.append(float(datos[6]))
            best_its.append(float(datos[9]))

        diccionario = {
            "name": instancia_carga,
            "instancia": instancia,
            "Q" : Q,
            "min": min(vals),
            "avg": mean(vals),
            "t_avg": mean(times),
            "avg_its": mean(best_its),
            "least_its": min(best_its)
        }

        resultados.append(diccionario)

    df = pd.DataFrame(resultados)
    print(df)
    df.to_excel(output, index = False)

rescatar("salida_ILS8.txt", "Experimento 8.xlsx")
rescatar("salida_ILS7.txt", "Experimento 7.xlsx")

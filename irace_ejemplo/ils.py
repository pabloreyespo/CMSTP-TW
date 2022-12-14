#!/usr/bin/env python3
import tsplib95
import matplotlib.pyplot as plt
import random
import time
import animated_visualizer as animacion
import sys, getopt

graficar_ruta = False
coord_x = []
coord_y = []
problem = dict()
semilla = 0
Prob1 = 0.0
Prob2 = 0.0
Prob3 = 0.0

# distancia entre la ciudad i y j
def distancia(i, j):
    u = i, j
    return problem.get_weight(*u)

# Costo de la ruta
def costoTotal(ciudad):
    suma = 0
    i = 0
    while i < len(ciudad) - 1:
        # print(ciudad[i], ciudad[i +1])
        suma += distancia(ciudad[i], ciudad[i + 1])
        i += 1
    suma += distancia(ciudad[-1], ciudad[0])
    return suma

# heurística del vecino más cercano
def vecinoMasCercano(n, desde):
    actual = desde
    ciudad = []
    ciudad.append(desde)
    seleccionada = [False] * n
    seleccionada[actual] = True
    # print(seleccionada)
    while len(ciudad) < n:
        min = 9999999
        for candidata in range(n):
            if seleccionada[candidata] == False and candidata != actual:
                costo = distancia(actual, candidata)
                if costo < min:
                    min = costo
                    siguiente = candidata

        ciudad.append(siguiente)
        seleccionada[siguiente] = True
        actual = siguiente
    # print(ciudad)
    # print(costoTotal(ciudad))
    return ciudad

def DosOpt(ciudad):
    actual = 0
    n = len(ciudad)
    flag = True
    contar = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            nuevoCosto = distancia(ciudad[i], ciudad[j]) + distancia(ciudad[i + 1], ciudad[j + 1]) - distancia(ciudad[i], ciudad[i + 1]) - distancia(ciudad[j], ciudad[j + 1])
            if nuevoCosto < actual:
                actual = nuevoCosto
                min_i, min_j = i, j
                # Al primer cambio se sale
                contar += 1
                if contar == 1 :
                    flag = False

        if flag == False:
            break

    # Actualiza la subruta se encontró
    if actual < 0:
        ciudad[min_i + 1 : min_j + 1] = ciudad[min_i + 1 : min_j + 1][::-1]

def perturbation(ciudad):
    i = 0
    j = 0
    n = len(ciudad)
    while i == j:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)

    # intercambio
    temp = ciudad[i]
    ciudad[i] = ciudad[j]
    ciudad[j] = temp

def perturbation3(ciudad):
    i = 0
    j = 0
    n = len(ciudad)
    while i == j:
        i = random.randint(0, n - 2)
        # j = random.randint(0, n - 1)
    j = i + 1
        # intercambio
    temp = ciudad[i]
    ciudad[i] = ciudad[j]
    ciudad[j] = temp


def perturbation2(ciudad):
    i = 0
    j = 0
    n = len(ciudad)
    while i >= j:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
    ciudad[i : j] = ciudad[i : j][::-1]

def ILS(ciudad):
    inicioTiempo = time.time()
    random.seed(10)
    n = len(ciudad)
    s = vecinoMasCercano(n, 0)
    s_mejor = s[:]
    costoMejor = costoTotal(s_mejor)
    lista_soluciones = []
    lista_costos = []
    lista_costosMejores = []
    lista_costos.append(costoMejor)
    lista_costosMejores.append(costoMejor)
    lista_soluciones.append(s_mejor)
    print("inicial %d" % costoMejor)
    iterMax = 5
    for iter in range(iterMax):
        # Perturbación
        if random.uniform(0, 1) < Prob1:
            perturbation(s)
        elif random.uniform(0, 1) < Prob1+Prob2:
            perturbation2(s)
        else:
            perturbation3(s)
        nuevo2 = costoTotal(s)
        #print(nuevo2)
        # Búsqueda local
        DosOpt(s)
        # DosOpt(s)
        nuevo = costoTotal(s)
        #print(nuevo2, nuevo)
        # Mejor solución
        if costoMejor > nuevo:
            s_mejor = s[:]
            costoMejor = nuevo
            print("%d\t%d" % (iter, costoMejor))
            # graficar(coord_x, coord_y, s_mejor)
        lista_costos.append(nuevo)
        lista_costosMejores.append(costoMejor)
        lista_soluciones.append(s)
        # criterio de aceptación de la solución actual
        if abs(costoMejor - nuevo) / costoMejor > Prob3:
            s = s_mejor[:]
        # print(costoTotal(s_mejor))
    finTiempo = time.time()
    tiempo = finTiempo - inicioTiempo
    print("tiempo: ", tiempo, " segundos")
    print(s_mejor)
    print("Mejor", costoTotal(s_mejor))

    lista_soluciones.append(s_mejor)
    lista_costos.append(costoMejor)

    # if graficar_ruta:
    #     animacion.animateTSP(lista_soluciones, coord_x, coord_y, lista_costos)
    #     graficar_soluciones(lista_costosMejores)

def graficar_soluciones(soluciones):
    plt.plot([i for i in range(len(soluciones))], soluciones)
    plt.ylabel("Costo")
    plt.xlabel("Iteraciones")
    plt.title("Iteraciones vs Costo - TSP")
    plt.xlim((0, len(soluciones)))
    plt.show()

def graficar(coord_x, coord_y, solucion):
    plt.figure(figsize = (20,20))
    plt.scatter(coord_x, coord_y, color = 'blue')
    s = []
    for n in range(len(coord_x)):
        s_temp = []
        s_temp.append("%.1f" % coord_x[n])
        s_temp.append("%.1f" % coord_y[n])
        s.append(s_temp)

        plt.xlabel("Distancia X")
        plt.ylabel("Distancia Y")
        plt.title("Ubicacion de las ciudades - TSP")

    ruta = list(solucion)
    if len(ruta) != 0:
        for i in range(len(ruta))[:-1]:
            plt.plot([coord_x[ruta[i]], coord_x[ruta[i+1]]],[coord_y[ruta[i]], coord_y[ruta[i+1]]], color='b', alpha=0.4, zorder=0)
            plt.scatter(x = coord_x, y = coord_y, color='blue', zorder=1)

    for n in range(len(coord_x)):
        plt.annotate(str(n), xy=(coord_x[n], coord_y[n] ), xytext=(coord_x[n]+0.5, coord_y[n]+1),color='red')

def main():
    G = problem.get_graph()
    ciudad = list(problem.get_nodes())

    info = problem.as_keyword_dict()
    if info['EDGE_WEIGHT_TYPE'] == 'EUC_2D': # se puede graficar la ruta
        global graficar_ruta
        graficar_ruta = True
        for i in range(len(ciudad)):
            x, y = info['NODE_COORD_SECTION'][i]
            coord_x.append(x)
            coord_y.append(y)

    graficar(coord_x, coord_y, [])
    ILS(ciudad)

if __name__ == "__main__":
    argv = sys.argv[1:]
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"a:b:c:i:s:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('test.py -a prob1 -b prob2 -c prob3')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-a':
            Prob1 = float(arg)
        elif opt in ("-b", "--bfile"):
            Prob2 = float(arg)
        elif opt in ("-c", "--cfile"):
            Prob3 = float(arg)
        elif opt in ("-i", "--ifile"):
            instancia = arg
            problem = tsplib95.load(instancia)
        elif opt in ("-s", "--sfile"):
            semilla = arg

    print ('Prob1:', Prob1)
    print ('Prob2:', Prob2)
    print ('Prob3:', Prob3)
    print ('Instancia:', instancia)
    print ('Semilla:', semilla)
    main()

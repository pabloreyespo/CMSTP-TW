import numpy as np
np.random.seed(42)

with open("coordenadas.txt", "r") as file:
    lineas = file.readlines()
    lineas = [tuple(map(float, l.rstrip().split())) for l in lineas]

nodos = []
xcoords = []
ycoords = []

for i, x, y in lineas:
    nodos.append(int(i))
    xcoords.append(x)
    ycoords.append(y)

parents = [-1, 0, 1, 4, 0, 0, 5, 4, 5, 6]
gates   = [ 0, 1, 1, 4, 4, 5, 5, 4, 5, 5]
loads   = [ 0, 2, 0, 0, 3, 4, 0, 0, 0, 0]
arrival = [0] * 10
xcoords = np.array(xcoords) * 10 
ycoords = np.array(ycoords) * 10 
#earliest = np.random.random(10) * 10
#earliest = earliest.round(1) * 10
earliest = np.array([0,  95, 73, 50, 55, 6, 16, 87, 60, 40])
latest   = np.array([200, 114, 152, 72, 60, 47, 73, 97, 82, 50])
earliest[0] = 0

print(xcoords)
print(ycoords)

for i in range(10):
    for j in nodos:
        if j > 0:
            if parents[j] == i:
                arrival[j] = arrival[i] + int(np.sqrt((xcoords[i] - xcoords[j])** 2 + (ycoords[i] - ycoords[j])** 2))
                if arrival[j] <= earliest[j]:
                    arrival[j] = earliest[j]
                    
suma = np.random.random(10) * 20
latest = arrival + suma.round()
latest[0] = 120
print(parents)
print(list(arrival))
print(earliest)
print(latest)
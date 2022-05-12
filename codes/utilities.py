import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

def read_salomon(location):
    nodes = []
    with open(location,"r") as inst:
        for i, line in enumerate(inst):
            if i in [1,2,3,5,6,7,8]:
                pass
            elif i == 0:
                name = line.strip()
            elif i == 4:
                capacity = line.strip().split()[-1]
            else:
                nodes.append(line.strip().split()[0:-1])
    return name, capacity, nodes

def read_R7(location):
    data = pd.read_csv(location)
    name = "R7"
    capacity = 10000
    nodes = data.to_numpy()
    return name, capacity, nodes

def extract_data(nodes):
    index, xcoords, ycoords, demands, earliest_times, latest_times = list(zip(*nodes))
    index = [int(i) for i in index]
    xcoords = [float(i) for i in xcoords]
    ycoords = [float(i) for i in ycoords]
    demands = [float(i) for i in demands]
    earliest_times = [float(i) for i in earliest_times]
    latest_times = [float(i) for i in latest_times]
    return index, xcoords, ycoords, demands, earliest_times, latest_times

def extract_data_R7(nodes):
    index, xcoords, ycoords, earliest_times, latest_times = list(zip(*nodes))
    index = [int(i) for i in index]
    if index[0] == 1:
        index = [i-1 for i in index]
    xcoords = [float(i) for i in xcoords]
    ycoords = [float(i) for i in ycoords]
    demands = [1 for i in index]
    earliest_times = [float(i) for i in earliest_times]
    latest_times = [float(i) for i in latest_times]
    return index, xcoords, ycoords, demands, earliest_times, latest_times

def visualizar(xcoords, ycoords, edges, weights = None):
    fig, ax = plt.subplots(1,1)
    
    ax.scatter(xcoords[0],ycoords[0], color ='green',marker = 'o',s = 275,zorder=2)
    ax.scatter(xcoords[1:],ycoords[1:], color ='indianred',marker = 'o',s = 275,zorder=2)

    for i,j in  edges: 
        ax.plot([xcoords[i],xcoords[j]],[ycoords[i],ycoords[j]], color = 'black',linestyle = ':',zorder=1)

    for i in range(len(xcoords)): 
        plt.annotate(str(i) ,xy = (xcoords[i],ycoords[i]), xytext = (xcoords[i]-0.6,ycoords[i]-0.6), color = 'black',zorder=4)

    if weights is not None:
        for i in range(len(xcoords)):
            plt.annotate(weights[i] ,xy = (xcoords[i],ycoords[i]), xytext = (xcoords[i]-0.6,ycoords[i]-0.6), color = 'black',zorder=4)

    plt.show()

# name, capacity, nodes = read_salomon("instances/c101.txt")
# index, xcoords, ycoords, demands, earliest_times, latest_times = extract_data(nodes)
# lim = 100
# N = [i for i in range(lim)]
# xcoords = xcoords[0:lim]
# ycoords = ycoords[0:lim]

# edges = [(i,j) for i in N for j in N if i != j]

# visualizar(xcoords, ycoords, edges)


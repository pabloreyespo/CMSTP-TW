import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
import numpy as np

class instance():
    def __init__(self, name, capacity, node_data, num, reset_demand = True):
        self.name = name
        self.n = num + 1
        self.capacity = int(capacity)
        self.index, self.xcoords, self.ycoords, self.demands, self.earliest, self.latest\
            = extract_data(node_data[:num+1])

        self.nodes = [i for i in range(num+1)]
        self.edges = [(i,j) for i in self.nodes for j in self.nodes[1:] if i != j]

        #demands = 1 for all nodes 
        if reset_demand:
            self.demands = {i:1 for i in self.nodes}

        # cost = time = distance for simplicity

        global D
        D = np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(i+1,self.n):
                D[i,j] = self.dist(i,j)
                D[j,i] = D[i,j]

        self.cost = D

    def dist(self,i,j):
        x = self.xcoords[i] - self.xcoords[j]
        y = self.ycoords[i] - self.ycoords[j]
        return sqrt(x**2 + y**2)

def read_instance(location, extension = "txt"):
    if extension == "txt":
        node_data = []
        with open(location,"r") as inst:
            for i, line in enumerate(inst):
                if i in [1,2,3,5,6,7,8]:
                    pass
                elif i == 0:
                    name = line.strip()
                elif i == 4:
                    capacity = line.strip().split()[-1]
                else:
                    node_data.append(line.strip().split()[0:-1])
    elif extension == "csv":
        data = pd.read_csv(location)
        name = "R7"
        capacity = 10000000
        node_data = data.to_numpy()
    else:
        print(f"extension '{extension}' not recognized")
        name, capacity, node_data = [None]*3
    return name, capacity, node_data

def extract_data(nodes):
    try:
        # Read txt solutions
        index, xcoords, ycoords, demands, earliest_times, latest_times = list(zip(*nodes))
    except:
        # read R7 data
        index, xcoords, ycoords, earliest_times, latest_times = list(zip(*nodes))
        demands = [1 for _ in index]
        
    index = [int(i) for i in index]
    xcoords = [float(i) for i in xcoords]
    ycoords = [float(i) for i in ycoords]
    demands = [float(i) for i in demands]
    earliest_times = [float(i) for i in earliest_times]
    latest_times = [float(i) for i in latest_times]
    return index, xcoords, ycoords, demands, earliest_times, latest_times

def visualize(xcoords, ycoords, F):
    fig, ax = plt.subplots(1,1)

    # root node
    ax.scatter(xcoords[0],ycoords[0], color ='green',marker = 'o',s = 275,zorder=2)
    # other nodes
    ax.scatter(xcoords[1:],ycoords[1:], color ='indianred',marker = 'o',s = 275,zorder=2)

    # edges activated

    for i in  F: 
        if i != 0:
            j = F[i]
            ax.plot([xcoords[i],xcoords[j]],[ycoords[i],ycoords[j]], color = 'black',linestyle = ':',zorder=1)

    # node label
    for i in range(len(xcoords)): 
        plt.annotate(str(i) ,xy = (xcoords[i],ycoords[i]), xytext = (xcoords[i]-0.6,ycoords[i]-0.6), color = 'black',zorder=4)

    plt.show()

if __name__ == "__main__":
    pass

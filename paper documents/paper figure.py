from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
from math import inf, sqrt
from random import choice, seed, sample, random, randint
from sortedcollections import SortedDict
from time import perf_counter
from disjoint_set import DisjointSet

env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

PENALIZATION = 9.378
PERTURBATION1 = 0.1
PERTURBATION2 = 0.1 + 0.9 * 0.392
LOCAL_SEARCH_PARAM1 = 0.1 # best_father
LOCAL_SEARCH_PARAM2 = 0.1 + 0.9 * 0.402 # best_father
BRANCH_TIME = 2
INITIAL_TRIGGER = 40

class tree:
    def __init__(self, n, Q, D):
        self.parent = np.ones(n, dtype = int) * -1
        self.gate = np.zeros(n, dtype = int)
        self.load = np.zeros(n, dtype = int)
        self.arrival = np.zeros(n)
        self.capacity = Q
        self.distance = D

    def connect(self, k, j):
        """k: parent, j: children"""
        if self.gate[k] == 0:
            self.gate[j] = gate = j
        else:
            self.gate[j] = gate = self.gate[k]
        self.load[gate] += 1
        self.parent[j] = k
        self.arrival[j] = self.arrival[k] + distance(k,j)
        if not self.arrival[j] >= earliest[j]:
            self.arrival[j] = earliest[j]
            
    def fitness(self):
        cost = 0 
        feasible = True
        for j,k in enumerate(self.parent):
            if j != 0:
                cost += distance(k,j)
                arr = self.arrival[j]
                lat = latest[j]
                if  arr > lat:
                    feasible = False
                    cost += (arr - lat) * PENALIZATION

        return cost, feasible
    
    def __repr__(self):
        
        parent = f'parent: [{", ".join([str(elem) for elem in self.parent])}]\n'
        gate = f'gate: [{", ".join([str(elem) for elem in self.gate])}]\n'
        load = f'load: [{", ".join([str(elem) for elem in self.load])}]\n'
        arrival = f'arrival: [{", ".join([str(elem) for elem in self.arrival])}]\n'
        return parent + gate + load + arrival

class instance():
    def __init__(self, name, capacity, node_data, num, reset_demand = True):
        self.name = name
        self.n = n = num + 1
        self.capacity = int(capacity)
        self.index, self.xcoords, self.ycoords, self.demands, self.earliest, self.latest\
            = extract_data(node_data[:n])

        self.nodes = np.array(range(n), dtype = int)
        self.edges = [(i,j) for i in self.nodes for j in self.nodes[1:] if i != j]

        #demands = 1 for all nodes 
        if reset_demand:
            self.demands = {i:1 for i in self.nodes}

        # cost = time = distance for simplicity

        global D
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                if i != j:
                    D[i,j] = D[j,i]= self.dist(i,j)

        self.cost = D

    def dist(self,i,j):
        x = self.xcoords[i] - self.xcoords[j]
        y = self.ycoords[i] - self.ycoords[j]
        return sqrt(x**2 + y**2)

def distance(i,j):
    return D[i,j]

def initialize(ins):
    global D, Q, earliest, latest, xcoords, ycoords
    D = ins.cost
    Q = ins.capacity
    earliest = ins.earliest
    latest = ins.latest
    xcoords = ins.xcoords    
    ycoords = ins.ycoords

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
    # Read txt solutions
    index, xcoords, ycoords, demands, earliest, latest = list(zip(*nodes))
        
    index = np.array([int(i) for i in index], dtype = int)
    xcoords = np.array([float(i) for i in xcoords])
    ycoords = np.array([float(i) for i in ycoords])
    demands = np.array([float(i) for i in demands])
    earliest = np.array([float(i) for i in earliest])
    latest = np.array([float(i) for i in latest])

    return index, xcoords, ycoords, demands, earliest, latest

def visualize(xcoords, ycoords, s):
    fig, ax = plt.subplots(1,1)

    # root node
    ax.scatter(xcoords[0],ycoords[0], color ='green',marker = 'o',s = 275, zorder=2)
    # other nodes
    ax.scatter(xcoords[1:],ycoords[1:], color ='indianred',marker = 'o',s = 275, zorder=2)

    # edges activated
    for j,k in  enumerate(s.parent): 
        if j != 0:
            ax.plot([xcoords[k],xcoords[j]],[ycoords[k],ycoords[j]], color = 'black',linestyle = ':',zorder=1)

    # node label
    for i in range(len(xcoords)): 
        plt.annotate(str(i) ,xy = (xcoords[i],ycoords[i]), xytext = (xcoords[i]-0.04,ycoords[i]-0.05), color = 'black', zorder=4)
    plt.show()

def gurobi_solution(ins, vis = False, time_limit = 1800, verbose = False, initial = False, start = None):

    n = ins.n
    initialize(ins)

    edges, cost = gp.multidict({(i,j): D[i,j] for (i,j) in ins.edges})
    nodes, earliest, latest, demands = gp.multidict({i: (ins.earliest[i], ins.latest[i], ins.demands[i]) for i in ins.nodes })
    nodesv = nodes[1:]

    M = max(latest.values()) + max(cost.values())

    # model and variables
    if verbose:
        mdl = gp.Model(ins.name)
    else:
        mdl = gp.Model(ins.name, env = env)
    x = mdl.addVars(edges, vtype = GRB.BINARY, name = "x") #
    y = mdl.addVars(edges, vtype = GRB.CONTINUOUS, name = "y", lb = 0)
    d = mdl.addVars(nodes, vtype = GRB.CONTINUOUS, name = "d", lb = 0)

    mdl.setObjective(x.prod(cost))
    # par = [-1,3,6,0,12,6,0,0,7,3,11,7,0]
    R1 = mdl.addConstrs((gp.quicksum(x[(i,j)] for i in nodes if i!=j) == 1 for j in nodesv),name = "R1")
    R2 = mdl.addConstrs((gp.quicksum(y[(i,j)] for i in nodes if i!=j) - gp.quicksum(y[(j,i)] for i in nodesv if i!=j) == demands[j] for j in nodesv), name = "R2") 
    R3 = mdl.addConstrs((x[(i,j)] <= y[(i,j)] for i,j in edges),name = "R3") 
    R4 = mdl.addConstrs((y[(i,j)] <= Q * x[(i,j)] for i,j in edges), name = "R4") 
    R5 = mdl.addConstrs((d[i] + cost[(i,j)] - d[j] <= M * (1 - x[(i,j)]) for i,j in edges), name = "R5") 
    R6 = mdl.addConstrs((d[i] >= earliest[i] for i in nodes), name = "R6") 
    R7 = mdl.addConstrs((d[i] <= latest[i] for i in nodes), name = "R7")
    # R8 = mdl.addConstrs((x[(par[j],j)] == 1 for j in nodesv), name = "R8")

    mdl.Params.TimeLimit = time_limit
    mdl.Params.Threads = 1

    solution = mdl.optimize()
    obj = mdl.getObjective()
    objective_value = obj.getValue()
    s = tree(n, Q, D)

    if not initial:

        time = mdl.Runtime
        best_bound = mdl.ObjBound
        gap = mdl.MIPGap

        if vis:
            for i,j in edges:
                if x[i,j].X > 0.9:
                    s.parent[j] = i
            visualize(ins.xcoords, ins.ycoords, s)

        return objective_value, time, best_bound, gap

    else: 
        departure = np.zeros(n)
        for i,j in edges:
            if x[(i,j)].X > 0.9:
                s.parent[j] = i
                departure[j] = d[j].X

        for j in sorted(nodes, key = lambda x: departure[x]):
            if j != 0:
                k = s.parent[j]
                s.connect(k,j)

        if vis:
            visualize(ins.xcoords, ins.ycoords, s)

        return s, objective_value
    
def main():
    name, capacity, node_data = read_instance("instances/r102.txt")
    ins = instance(name, capacity, node_data, 8)
    ins.name = "Prueba Paper"
    # np.random.seed(12)
    np.random.seed(0)
    # ins.xcoords = np.random.randint(0,5,8)
    # ins.ycoords = np.random.randint(0,5,8)  
    ins.xcoords = np.array([0,0.5,1,1,4,2,2.5,4,1.5])
    ins.ycoords = np.array([3,1.5,2,4,2,1,2.0,4,2.5])
    #ins.xcoords[4], ins.xcoords[0] = ins.xcoords[0], ins.xcoords[4]
    #ins.ycoords[4], ins.ycoords[0] = ins.ycoords[0], ins.ycoords[4]
    ins.earliest = np.random.randint(0,10,9) * 10
    ins.latest = ins.earliest + np.random.randint(0,20,9) * 10
    ins.earliest[0] = 0
    ins.latest[0] = 1000
    # cost = time = distance for simplicity
    global D
    D = np.zeros((ins.n,ins.n))
    for i in range(ins.n):
        for j in range(i+1,ins.n):
            if i != j:
                D[i,j] = D[j,i]= ins.dist(i,j)
    ins.cost = D
    ins.capacity = 4

    global xcoords, ycoords, latest, earliest, Q
    xcoords, ycoords = ins.xcoords, ins.ycoords
    earliest, latest = ins.earliest, ins.latest
    D = ins.cost
    Q = ins.capacity
    
    print(ins.xcoords, ins.ycoords)
    print(ins.nodes)
    print(ins.n)
    print(ins.earliest)
    print(ins.latest)
    print(ins.cost)
    s, cost = gurobi_solution(ins, vis = True, time_limit= 30, verbose = True, initial=True)
    print("gurobi:", cost)
    print(s.parent)
    print(s.gate)
    print(s.load)
    print(s.arrival)

if __name__ == "__main__":
    main()
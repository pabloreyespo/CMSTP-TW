import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
from math import inf, sqrt
from random import choice, seed, sample, random
from sortedcollections import SortedDict
from time import perf_counter
from disjoint_set import DisjointSet

env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

PENALIZATION = 0.5
PERTURBATION_A = 1
PERTURBATION_B = 0
LOCAL_SEARCH_PARAM = 1 # best_father
BRANCH_TIME = 1
INITIAL_TRIGGER = 20
LOCAL_SEARCH_PROPORTION = 0
MERGE_BEST = False

def distance(i,j):
    return D[i,j]

def copy(s):
    return (s[0].copy(), s[1].copy(), s[2].copy(), s[3].copy())

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

def visualize(xcoords, ycoords, F):
    fig, ax = plt.subplots(1,1)

    # root node
    ax.scatter(xcoords[0],ycoords[0], color ='green',marker = 'o',s = 275, zorder=2)
    # other nodes
    ax.scatter(xcoords[1:],ycoords[1:], color ='indianred',marker = 'o',s = 275, zorder=2)

    # edges activated

    for j,k in  enumerate(F): 
        ax.plot([xcoords[k],xcoords[j]],[ycoords[k],ycoords[j]], color = 'black',linestyle = ':',zorder=1)

    # node label
    for i in range(len(xcoords)): 
        plt.annotate(str(i) ,xy = (xcoords[i],ycoords[i]), xytext = (xcoords[i]-0.6,ycoords[i]-0.6), color = 'black', zorder=4)

    plt.show()

def algorithm(ins, b, alpha, s, mu, vis  = False, Q = None, initial = False):
    nodes, n = ins.nodes, ins.n

    global D
    D = ins.cost

    if Q is None:
        Q = ins.capacity

    earliest = ins.earliest
    latest = ins.latest

    M = 100**5
    tol = 1**-7

    LPDH = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            LPDH[i,j] = 0 if distance(i,j) <= s else 1
            LPDH[j,i] = LPDH[i,j]

    ESGC = lambda j, sj, gsj : b[0]*D[sj,j] + b[1] * D[j,gsj] + b[2] * D[sj,gsj] + b[3] / abs(D[0,sj] - D[0,j] + tol) + b[4] / (D[0,j] - alpha * D[sj,j] + tol) + b[5]* abs(latest[j] - (latest[sj] + D[sj,j]))
    criterion = lambda j,sj,gsj: ESGC(j,sj,gsj) + mu * M *LPDH[sj,j]

    start = perf_counter()

    parent = np.ones(n, dtype = int) * -1
    gate = np.zeros(n, dtype = int)
    load = np.zeros(n, dtype = int)
    arrival = np.zeros(n)
    waiting = np.zeros(n)
        
    itree = set() # muestra que es lo ultimo que se ha añadido
    nodes_left = set(nodes)

    d = inf
    for j in nodes[1:]:
        di = distance(0,j)
        if di < d:
            d = di
            v = j

    itree.add(0) #orden en que son nombrados
    itree.add(v)
    nodes_left.remove(0)
    nodes_left.remove(v)

    gate[v] = v
    load[gate[v]] += 1
    parent[v] = 0
    arrival[v] = cost = d
    waiting[v] = 0

    if not  arrival[v] >= earliest[v]:
        waiting[v] = earliest[v]- arrival[v]
        arrival[v] = earliest[v]

    while len(nodes_left) > 0:
        min_tree = inf
        for j in nodes_left:
            min_node = inf
            for ki in itree:# k: parent, j: offspring
                # calcula si alcanza a llegar desde alguno de los nodos que ya estan colocados
                dkj = distance(ki,j)
                # criterion = dkj
                tj = arrival[ki] + dkj
                Qj = load[gate[ki]]

                if tj <= latest[j] and Qj < Q: # isFeasible() # reescribir

                    if tj < earliest[j]:
                        tj = earliest[j]

                    crit_node = dkj
                    if crit_node < min_node:
                        min_node = crit_node
                        k = ki
                
            ### best of the node
            crit_tree = criterion(j,k, gate[k])

            if crit_tree < min_tree:
                kk = k
                jj = j
                min_tree = crit_tree

        itree.add(jj)
        nodes_left.remove(jj)
        parent[jj] = kk
        # visualize(ins.xcoords, ins.ycoords, parent)
        cost += distance(kk,jj)
        if gate[kk] == 0:
            gate[jj] = jj
        else:
            gate[jj] = gate[kk]
        load[gate[jj]] += 1
        
        arrival[jj] = arrival[kk] + distance(kk,jj)
        if not arrival[jj] >= earliest[jj]:
            waiting[jj] = earliest[jj] - arrival[jj]
            arrival[jj] = earliest[jj]

    time = perf_counter() - start
        
    if vis:
        visualize(ins.xcoords, ins.ycoords, parent)
    
    if initial:
        return (parent, gate, load, arrival) , cost
        
    else:
        best_bound = None
        gap = None
        return cost, time, best_bound, gap

def prim(ins,  vis  = False, initial = False):
    return algorithm(ins,  b = np.array([1,0,0,0,0,0]), alpha = 0, s = 0, mu = 0, vis  = vis, initial = initial)

def gurobi_solution(ins, vis = False, time_limit = 1800, verbose = False, initial = False, start = None):

    n = ins.n

    Q = ins.capacity
    earliest = ins.earliest
    latest = ins.latest
    demands = ins.demands
    global D
    D = ins.cost

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

    if start is not None:
        parent = start[0][0]
        arrival = start[0][3]
        for j in nodes:
            if j != 0:
                i = parent[j]
                x[(i,j)].Start = 1
                d[j].Start = arrival[j]

    mdl.setObjective(x.prod(cost))

    R1 = mdl.addConstrs((gp.quicksum(x[(i,j)] for i in nodes if i!=j) == 1 for j in nodesv),name = "R1")
    R2 = mdl.addConstrs((gp.quicksum(y[(i,j)] for i in nodes if i!=j) - gp.quicksum(y[(j,i)] for i in nodesv if i!=j) == demands[j] for j in nodesv), name = "R2") 
    R3 = mdl.addConstrs((x[(i,j)] <= y[(i,j)] for i,j in edges),name = "R3") 
    R4 = mdl.addConstrs((y[(i,j)] <= Q * x[(i,j)] for i,j in edges), name = "R4") 
    R5 = mdl.addConstrs((d[i] + cost[(i,j)] - d[j] <= M * (1 - x[(i,j)]) for i,j in edges), name = "R5") 
    R6 = mdl.addConstrs((d[i] >= earliest[i] for i in nodes), name = "R6") 
    R7 = mdl.addConstrs((d[i] <= latest[i] for i in nodes), name = "R7")


    mdl.Params.TimeLimit = time_limit
    mdl.Params.Threads = 1

    solution = mdl.optimize()

    obj = mdl.getObjective()
    objective_value = obj.getValue()

    if not initial:

        time = mdl.Runtime
        best_bound = mdl.ObjBound
        gap = mdl.MIPGap

        if vis:
            parent = np.ones(n, dtype = int) * -1
            for i,j in edges:
                if x[i,j].X > 0.9:
                    parent[j] = i
            visualize(ins.xcoords, ins.ycoords, parent)

        return objective_value, time, best_bound, gap

    else: 
        parent = np.ones(n, dtype = int) * -1
        departure = np.zeros(n)
        for i,j in edges:
            if x[(i,j)].X > 0.9:
                parent[j] = i
                departure[j] = d[j].X

        gate = np.zeros(n, dtype = int)
        load = np.zeros(n, dtype = int)
        arrival = np.zeros(n)

        for j in sorted(nodes, key = lambda x: departure[x]):
            if j != 0:
                i = parent[j]
                if i == 0:
                    gate[j] = j
                else:
                    gate[j] = gate[i]
                load[gate[j]] += 1
                arrival[j] = arrival[i] + distance(i,j)
                if arrival[j] < earliest[j]:
                    arrival[j] = earliest[j]
        return (parent, gate, load, arrival), objective_value

def fitness(s):
    P = s[0]
    arrival = s[3]
    cost = 0 
    feasible = True
    for j,k in enumerate(P):
        if j != 0:
            cost += distance(k,j)
            if arrival[j] > latest[j]:
                feasible = False
                cost += (arrival[j] - latest[j]) * PENALIZATION
    return cost, feasible

def main():
    name, capacity, node_data = read_instance("instances/c207.txt")
    ins = instance(name, capacity, node_data, 100)

    global xcoords, ycoords, latest, earliest, D, Q
    xcoords, ycoords = ins.xcoords, ins.ycoords
    earliest, latest = ins.earliest, ins.latest
    D = ins.cost
    Q = ins.capacity
    
    initial_solution = prim(ins, vis = False, initial = True)
    obj, time, best_bound, gap = gurobi_solution(ins, vis = False, time_limit= 30, verbose = False, start =initial_solution)

def test(cap, nnodes, nombre, time_limit, instances_folder):
    instances = os.listdir(instances_folder)
    results = list()
    for inst in instances:
        print(f"{inst}-Q{cap}")
        name, capacity, node_data = read_instance(f"{instances_folder}/{inst}")
        ins = instance(name, capacity, node_data, nnodes)
        ins.capacity = cap

        global xcoords, ycoords, latest, earliest, D, Q
        xcoords, ycoords = ins.xcoords, ins.ycoords
        earliest, latest = ins.earliest, ins.latest
        D = ins.cost
        Q = ins.capacity

        initial_solution = prim(ins, vis = False, initial = True)
        obj, time, best_bound, gap = gurobi_solution(ins, vis = False, time_limit= time_limit, verbose = True, start =initial_solution) 
        dic = {"name": name, "LB": best_bound, "UB": obj, "gap": gap, "t": time}
        results.append(dic)

    df = pd.DataFrame(results)
    df.to_excel(f"{nombre}.xlsx", index= False)

if __name__ == "__main__":
    instances_folder = "instances" # "gehring instances/200" para instancias más grandes
    caps = [10000,20,15,10,5]
    nnodes = 100
    time_limit = 60
    for cap in caps:
        nombre = f"GUROBI n{nnodes} Q-{cap} {time_limit}s"
        test(cap, nnodes, nombre, time_limit, instances_folder)
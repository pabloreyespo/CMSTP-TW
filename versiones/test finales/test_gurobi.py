import gurobipy as gp
from gurobipy import GRB

import pandas as pd
import numpy as np
from math import inf, sqrt
from time import perf_counter
from scipy.spatial.distance import cdist
import os, sys, getopt

env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()
rho = 7.001
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
        cost, feasible = 0, True 
        for j,k in enumerate(self.parent):
            if j != 0:
                cost += distance(k,j)
                arr, lat = self.arrival[j], latest[j]
                if  arr > lat:
                    feasible = False
                    cost += (arr - lat) * rho

        return cost, feasible
    
def read_instance(location):
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

class instance():
    def __init__(self, name, capacity, node_data, num, reset_demand = True):
        self.name = name
        self.n = n = num + 1
        self.capacity = int(capacity)
        self.index, self.xcoords, self.ycoords, self.demands, self.earliest, self.latest\
            = extract_data(node_data[:n])

        self.nodes = np.array(range(n), dtype = int)
        self.edges = [(i,j) for i in self.nodes for j in self.nodes[1:] if i != j]
        self.edges_index = {(i,j): ind for ind, (i,j) in enumerate(self.edges)}

        #demands = 1 for all nodes 
        if reset_demand:
            self.demands = {i:1 for i in self.nodes}

        # cost = time = distance for simplicity
        global D
        aux = np.vstack((self.xcoords, self.ycoords)).T
        D  = cdist(aux,aux, metric='euclidean')

        self.cost = D
        self.maxcost = self.cost.mean()

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
    earliest, latest = ins.earliest, ins.latest
    xcoords, ycoords = ins.xcoords, ins.ycoords


def distance(i,j):
    return D[(i,j)]

def prim(ins, vis  = False, initial = False):
    
    initialize(ins)

    nodes, n = ins.nodes, ins.n
    start = perf_counter()

    s = tree(n, Q, D)
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
    
    s.connect(0,v)
    cost = distance(0,v)

    while len(nodes_left) > 0:
        min_tree = inf
        for j in nodes_left:
            min_node = inf
            for ki in itree:# k: parent, j: offspring
                # calcula si alcanza a llegar desde alguno de los nodos que ya estan colocados
                dkj = distance(ki,j)
                # criterion = dkj
                tj = s.arrival[ki] + dkj
                Qj = s.load[s.gate[ki]]

                if tj <= latest[j] and Qj < Q: # isFeasible() # reescribir

                    if tj < earliest[j]:
                        tj = earliest[j]

                    crit_node = dkj
                    if crit_node < min_node:
                        min_node = crit_node
                        k = ki
                
            ### best of the node
            crit_tree = crit_node

            if crit_tree < min_tree:
                kk = k
                jj = j
                min_tree = crit_tree

        itree.add(jj)
        nodes_left.remove(jj)
        s.connect(kk,jj)
        cost += distance(kk,jj)

    time = perf_counter() - start
    
    if initial:
        return s , cost
        
    else:
        best_bound = None
        gap = None
        return cost, time, best_bound, gap

def gurobi_solution(ins, vis = False, time_limit = 1800, verbose = False, initial = False, start = None, rando = False):
    n = ins.n
    initialize(ins)

    edges, cost = gp.multidict({(i,j): D[i,j] for (i,j) in ins.edges})
    nodes, earliest, latest, demands = gp.multidict({i: (ins.earliest[i], ins.latest[i], ins.demands[i]) for i in ins.nodes })
    nodesv = nodes[1:]

    M = max(latest.values()) + max(cost.values())

    # model and variables
    mdl = gp.Model(ins.name, env = env)

    x = mdl.addVars(edges, vtype = GRB.BINARY, name = "x") #
    y = mdl.addVars(edges, vtype = GRB.CONTINUOUS, name = "y", lb = 0)
    d = mdl.addVars(nodes, vtype = GRB.CONTINUOUS, name = "d", lb = 0)

    if start:   
        for j in ins.nodes[1:]:
            i = start.parent[j] 
            x[(i,j)].Start = 1 # fijar solucion inicial
            # d[j].Start = start.arrival[j] # fijar salida en solución inicial
            if rando:
                x[i,j] = mdl.addVar(vtype = GRB.BINARY, lb=1, ub=1, name = "x[%d,%d]" % (i,j)) # rando contreras
                x[i,j].VarHintVal = 1
                x[i,j].VarHintPri = int(ins.maxcost * (1 + 1/ins.cost[i,j]))


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
    time = mdl.Runtime
    best_bound = mdl.ObjBound
    gap = mdl.MIPGap

    return objective_value, time, best_bound, gap

def test(cap, nnodes, nombre, time_limit):
    
    folder = "gehring instances/200" if nnodes > 100 else "instances"
    instances = os.listdir(folder)
    results = list()
    for inst in instances:
        print(f"{inst}-Q{cap}")
        name, capacity, node_data = read_instance(f"{folder}/{inst}")
        ins = instance(name, capacity, node_data, nnodes)
        ins.capacity = cap

        s, cost = prim(ins, vis = False, initial = True)
        obj, time, best_bound, gap = gurobi_solution(ins, time_limit= time_limit, start=s) 
        dic = {"name": name, "LB": best_bound, "UB": obj, "gap": gap, "t": time}
        results.append(dic)

    df = pd.DataFrame(results)
    df.to_excel(f"{nombre}.xlsx", index= False)

if __name__ == "__main__":
    caps = [10000,20,15,10,5]
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'n:t:')
        print("Leido")
    except getopt.GetoptError:
        print ('test.py -n nnodes -t time')
        exit(0)

    for opt, arg in opts:
        if opt == '-n':
            nnodes = int(arg)
        elif opt == "-t":
            time_limit = int(arg)

    for cap in caps:
        test(cap, nnodes, f"GUROBI_{time_limit}_n{nnodes}_Q{cap}", time_limit)
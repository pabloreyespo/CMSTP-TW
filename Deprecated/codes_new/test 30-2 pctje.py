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
LOCAL_SEARCH_PROPORTION = 0.1
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

class branch_bound:
    def __init__(self, s):
        self.nodes = s
        self.best_solution = None
        self.best_cost = inf

        P = SortedDict()
        arrival = SortedDict()
        gate = SortedDict()
        load = SortedDict()

        P[0] = -1
        arrival[0] = gate[0] = load[0] = 0
        for i in self.nodes:
            arrival[i] = gate[i] = load[i] = 0

        nodes_left = set(self.nodes)
        
        s = (P, gate, load, arrival)
        self.explore(s, 0, nodes_left)

    def explore(self, s, cost, nodes_left):
        if len(nodes_left) == 0:
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = copy(s)

        else:
            for j in nodes_left:
                for i in s[0].keys():
                    if cost + distance(i, j) < self.best_cost:
                        P, gate, load, arrival = copy(s)
                        P[j] = i
                        if i == 0:
                            gate[j] = j
                        else:
                            gate[j] = gate[i]
    
                        if arrival[i] + distance(i,j) <= latest[j] and load[gate[j]] < Q: # isFeasible() # reescribir
                            
                            load[gate[j]] += 1
                            arrival[j] = arrival[i] + distance(i,j)
                            if arrival[j] < earliest[j]:
                                arrival[j] = earliest[j]
                            
                            self.explore((P, gate, load, arrival), cost + distance(i, j), nodes_left - {j})
                            load[gate[j]] -= 1

def merge_branches(s):

    P = s[0]
    gate = s[1]
    load = s[2]
    arrival = s[3]
    n = len(P)

    ds = DisjointSet() 
    
    for u in range(1,n): ds.find(u) 

    for v in range(1,n): 
        u = P[v]
        if u != 0:
            if u != -1:
                if ds.find(u) != ds.find(v):
                    ds.union(u,v)

    ds = [list(i) for i in ds.itersets()]
    nds = len(ds)

    xc = xcoords - xcoords[0]
    yc = ycoords - ycoords[0]

    x_sets = np.zeros(nds)
    y_sets = np.zeros(nds)

    for i, st in enumerate(ds):
        m = len(st)
        if m > 1:
            x,y = 0,0
            for j in st:
                x += xc[j]
                y += yc[j]
            x,y = x/m, y/m
        else:
            j = st[0]
            x,y = xc[j], yc[j]
        x_sets[i], y_sets[i] = x,y

    r = np.sqrt(x_sets ** 2 + y_sets ** 2)
    theta = np.arctan2(y_sets, x_sets) + (random() * (2 * np.pi))
    branches = list(range(nds))
    branches = sorted(branches, key = lambda x: theta[x])
    
    for i in range(nds//2):
        s1, s2 = i*2, i*2+1
        branch = ds[s1] + ds[s2]
        lo = len(branch)

        if lo < 5 and lo >= 2:
            bb = branch_bound(branch)
            aux = bb.best_solution
        elif lo <= INITIAL_TRIGGER:
            aux = branch_gurobi(branch, P)
        else:
            aux = branch_gurobi(branch, P, initial = True)

        for j in branch:
            P[j] = aux[0][j]
            gate[j] = aux[1][j]
            load[j] = aux[2][j]
            arrival[j] = aux[3][j]

    cost, feasible = fitness((P, gate, load, arrival))
    return (P, gate, load, arrival), cost, feasible      

def optimal_branch(s):

    P = s[0]
    gate = s[1]
    load = s[2]
    arrival = s[3]

    for i in set(gate):
        if i != 0:
            lo = load[i]
            if lo <= 20 and lo >= 2:
                branch = [j for j in range(1, len(P)) if gate[j] == i]
                if lo < 5 and lo >= 2:
                    bb = branch_bound(branch)
                    aux = bb.best_solution
                else:
                    aux = branch_gurobi(branch, P)

                for j in branch:
                    P[j] = aux[0][j]
                    gate[j] = aux[1][j]
                    load[j] = aux[2][j]
                    arrival[j] = aux[3][j]

    cost, feasible = fitness((P, gate, load, arrival))
    return (P, gate, load, arrival), cost, feasible            

def branch_gurobi(branch, P, initial = False):

    nodes = [0] + branch
    nodesv = branch
    edges =  [(i,j) for i in nodes for j in nodesv if i != j]

    edges, cost = gp.multidict({(i,j): D[i,j] for (i,j) in edges})
    nodes, earliests, latests, demands = gp.multidict({i: (earliest[i], latest[i], 1) for i in nodes })
    nodesv = nodes[1:]

    M = max(latests.values()) + max(cost.values())

    # model and variables
    mdl = gp.Model(env = env)
    x = mdl.addVars(edges, vtype = GRB.BINARY, name = "x") #
    y = mdl.addVars(edges, vtype = GRB.CONTINUOUS, name = "y", lb = 0)
    d = mdl.addVars(nodes, vtype = GRB.CONTINUOUS, name = "d", lb = 0)

    mdl.setObjective(x.prod(cost))

    R1 = mdl.addConstrs((gp.quicksum(x[(i,j)] for i in nodes if i!=j) == 1 for j in nodesv),name = "R1")
    R2 = mdl.addConstrs((gp.quicksum(y[(i,j)] for i in nodes if i!=j) - gp.quicksum(y[(j,i)] for i in nodesv if i!=j) == demands[j] for j in nodesv), name = "R2")
    R3 = mdl.addConstrs((x[(i,j)] <= y[(i,j)] for i,j in edges),name = "R3")
    R4 = mdl.addConstrs((y[(i,j)] <= Q * x[(i,j)] for i,j in edges), name = "R4")
    R5 = mdl.addConstrs((d[i] + cost[(i,j)] - d[j] <= M * (1 - x[(i,j)]) for i,j in edges), name = "R5")
    R6 = mdl.addConstrs((d[i] >= earliests[i] for i in nodes), name = "R6")
    R7 = mdl.addConstrs((d[i] <= latests[i] for i in nodes), name = "R7")

    mdl.Params.TimeLimit = BRANCH_TIME
    mdl.Params.Threads = 1
    # ajustar un unicio para las variables
    if initial:
        for j in nodes:
            if j != 0:
                i = P[j]
                x[(i,j)].Start = 1

    solution = mdl.optimize() 

    P = SortedDict()
    departure = SortedDict()
    for i,j in edges:
        if x[(i,j)].X > 0.9:
            P[j] = i
            departure[j] = d[j].X

    gate= SortedDict()
    load = { j : 0 for j in P.keys()}
    arrival = SortedDict()
    arrival[0] = 0
    for j in sorted(P.keys(), key = lambda x: departure[x]):
        if j != 0:
            i = P[j]
            if i == 0:
                gate[j] = j
            else:
                gate[j] = gate[i]
            load[gate[j]] += 1
            arrival[j] = arrival[i] + distance(i,j)
            if arrival[j] < earliest[j]:
                arrival[j] = earliest[j]

    return (P, gate, load, arrival)

def perturbation(s):
    x = random()
    if x <= PERTURBATION_A:
        return random_father(s) 
    elif x <= PERTURBATION_B:
        return branch_to_root(s)
    else:
        return branch_to_branch(s)

def random_father(s): 

    P = s[0] 
    gate = s[1] 
    load = s[2] 
    arrival = s[3] 

    n = len(P) 
    j = np.random.randint(1,n) 
    P[j] = -1 

    ds = DisjointSet() 
    for u in range(n): 
        ds.find(u) 

    for v in range(n): 
        u = P[v]
        if u != -1:
            if ds.find(u) != ds.find(v):
                ds.union(u,v)

    sets = list(ds.itersets())
    con, disc = sets[0], sets[1]

    l = len(disc)
    load[gate[j]] -= l 

    for i in sorted(con, key = lambda x: random()):
        if load[gate[i]] + l <= Q: 
            
            P[j] = i
            subroot = gate[i] if i != 0 else j
            for jj in sorted(disc, key = lambda x: arrival[x]):
                gate[jj] = subroot
                arrival[jj] = max(arrival[P[jj]] + distance(jj, P[jj]), earliest[jj]) # hacer esto por orde de llegada, asi se me hace mas facil
            load[gate[j]] += l
            return (P, gate, load, arrival)

def branch_to_root(s):
    
    P = s[0]
    gate = s[1]
    load = s[2]
    arrival = s[3]
    n = len(P)
    
    j  = np.random.randint(1,n) 
    parents = [j]
    lo = 1
    arrival[j] = max(distance(0,j), earliest[j])
    
    old_gate = gate[j]
    gate[j] = j
    P[j] = 0
    arrival[j] = max(distance(0,j), earliest[j])
    
    for i in sorted(range(n), key = lambda x: arrival[x]):
        if P[i] in parents:
            parents.append(i) 
            lo += 1
            k = P[i]
            gate[i] = j
            arrival[i] = arrival[k] + distance(k,i)
            if arrival[i] < earliest[i]:
                arrival[i] = earliest[i]

    load[old_gate] -= lo
    load[j] += lo

    return (P, gate, load, arrival)

def branch_to_branch(s):
    P = s[0]
    gate = s[1]
    load = s[2]
    arrival = s[3]
    n = len(P)

    j = np.random.randint(1,n) 
    parents = [j]
    nodes = [j]

    # discover which nodes are part of the branch
    for i in sorted(range(n), key = lambda x: arrival[x]):
        if P[i] in parents:
            parents.append(i) 
            nodes.append(i)

    lo = len(nodes)
    load[gate[j]] -= lo

    for i in sorted(range(n), key = lambda x: random()):
        if i not in nodes:
            if load[gate[i]] + lo <= Q:

                P[j] = i
                arrival[j] = max(arrival[i] + distance(i,j), earliest[j])

                if i != 0:
                    gate[j] = gate[i]
                else:
                    gate[j] = j
                load[gate[j]] += lo
                break
      
    for i in sorted(nodes, key = lambda x: arrival[x]): #this should avoid using j
        k = P[i]
        gate[i] = gate[j]
        arrival[i] = arrival[k] + distance(k,i)
        if arrival[i] < earliest[i]:
            arrival[i] = earliest[i]

    return (P, gate, load, arrival)

def best_father(s): #perturbation 1
    P = s[0]
    gate = s[1]
    load = s[2]
    arrival = s[3]
    n = len(P)
    
    if random() <= LOCAL_SEARCH_PARAM:
        nodes = [np.random.randint(1,n)]
    else:
        nodes = sample(range(1,n), 5)

    for j in nodes:
    
        P[j] = -1
        ds = DisjointSet()
        for u in range(n):
            ds.find(u)

        for v in range(n):
            u = P[v]
            if u != -1:
                if ds.find(u) != ds.find(v):
                    ds.union(u,v)

        sets = list(ds.itersets())
        con, disc = sets[0], sets[1]

        l = len(disc)
        load[gate[j]] -= l # descontar lo que està de sobra
        # print(j)
        minim = inf
        for i in con:
            if load[gate[i]] + l <= Q and D[(i,j)] < minim:
                minim = D[(i,j)]
                k = i

        P[j] = k
        subroot = gate[k] if k != 0 else j
        for jj in sorted(disc, key = lambda x: arrival[x]):
            arrival[jj] = max(arrival[P[jj]] + distance(jj, P[jj]), earliest[jj])
            gate[jj] = subroot
        load[gate[j]] += l

    cost, feasible = fitness(s)

    return (P, gate, load, arrival), cost, feasible

def local_search(s):
    if random() < LOCAL_SEARCH_PROPORTION:
        return merge_branches(s)
    else:
        return best_father(s)

def ILS_solution(ins, semilla = None, acceptance = 0.05,
                feasibility_param = 100, elite_param = 250, elite_size = 20, iterMax = 15000, p = PENALIZATION,
                pa = PERTURBATION_A, pb = PERTURBATION_B, lsp = LOCAL_SEARCH_PARAM, initial_solution = None,
                elite_revision_param = 1500, vis  = False, verbose = False, time_limit = 60, limit_type = "t"):
    
    global PENALIZATION, Q, PERTURBATION_A, PERTURBATION_B, LOCAL_SEARCH_PARAM, D
    PENALIZATION = p
    PERTURBATION_A = pa
    PERTURBATION_B = pb
    LOCAL_SEARCH_PARAM = lsp
    
    if semilla is not None:
        np.random.seed(semilla)
        seed(semilla)

    start = perf_counter()

    if initial_solution is None:
        s, cost_best = prim(ins,  vis = False)
    else:
        s, cost_best = initial_solution(ins)

    s, candidate_cost = copy(s), cost_best
    cost_best_unfeasible = inf
    feasible = True

    s_best = copy(s)
    s_best_unfeasible = None

    
    # feasible_count = 1
    # unfeasible_count = 0
    # mejoras = 0

    # costs_list = [cost_best]
    # bestCosts_list = [cost_best]
    # solutions_list = [s_best]
    # feasibility_list = [feasible]

    elite = SortedDict()
    elite[cost_best] = [copy(s),False]

    it = 0
    best_it = 0
 
    get_counter = lambda : it if limit_type != 't' else perf_counter() - start
    limit = iterMax if limit_type != 't' else time_limit

    while get_counter() < limit:
        s = perturbation(s)
        s, candidate_cost, feasible = local_search(s)
        if feasible:
            if cost_best > candidate_cost:
                if MERGE_BEST:
                    s, candidate_cost, feasible = merge_branches(s)
                s_best = copy(s)
                cost_best = candidate_cost
                best_it = it + 1
                elite[candidate_cost] = [copy(s), False]
                if len(elite) > elite_size: 
                    elite.popitem()
        else:
            if cost_best_unfeasible > candidate_cost:
                s_best_unfeasible = copy(s)
                cost_best_unfeasible = candidate_cost

        if verbose: print(it, candidate_cost)
        else:
            # count = get_counter()
            # text = f'{count:^10.2f}/{limit} [{"#"*int(count*50//limit):<50}] cost: {candidate_cost:^10.3f} best: {cost_best:^10.3f} it: {it+1}'
            # print(text, end = "\r")
            pass
        
        #if feasible: feasible_count += 1
        #else: unfeasible_count += 1

        # costs_list.append(candidate_cost)
        # bestCosts_list.append(cost_best)
        # solutions_list.append(s)
        # feasibility_list.append(feasible)

        if abs(cost_best - candidate_cost) / cost_best > acceptance or not feasible:
            s = copy(s_best)
        
        if (it + 1) % feasibility_param == 0:
            try: s = copy(s_best_unfeasible)
            except: s = s

        if (it + 1) % elite_param == 0:
            x = choice(elite.values())
            x = x[0]
            s = copy(x)

        if (it + 1) % elite_revision_param == 0:
            for cost in elite:
                ss, rev = elite[cost]
                if not rev:
                    ss, cost_after, feasible = optimal_branch(copy(ss))
                    # print(cost, "->", cost_after)
                    elite[cost][1] = True
                    if feasible and cost_after < cost:
                        elite[cost_after] = [copy(ss), True]

                        if cost_after < cost_best:
                            s_best = copy(ss)
                            cost_best = cost_after
                            
                            if MERGE_BEST:
                                ss, cost_after, feasible = merge_branches(copy(ss))
                                if cost < cost_best:
                                    s_best = copy(ss)
                                    cost_best = cost_after

                            best_it = it + 1


                    while len(elite) > elite_size:
                        elite.popitem()

        it += 1
    
    time = perf_counter() - start
    best_bound = None
    gap = None

    if vis: visualize(ins.xcoords, ins.ycoords, s_best[0])
    if not verbose:
        count = get_counter()
        text = f'{count:^10.2f}/{limit} [{"#"*int(count*50//limit):<50}] cost: {candidate_cost:^10.3f} best: {cost_best:^10.3f} best_it: {best_it}'
        print(text)

    return cost_best, time, best_bound, gap

def main():
    name, capacity, node_data = read_instance("instances/c207.txt")
    ins = instance(name, capacity, node_data, 100)

    global xcoords, ycoords, latest, earliest, D, Q
    xcoords, ycoords = ins.xcoords, ins.ycoords
    earliest, latest = ins.earliest, ins.latest
    D = ins.cost
    Q = ins.capacity
    
    initial_solution = prim(ins, vis = False, initial = True)
    s, objective_value= gurobi_solution(ins, vis = False, time_limit= 30, verbose = False, initial=True, start =initial_solution)
    print(f"sol inicial: {objective_value}")
    initial_solution = lambda x: (copy(s), objective_value)

    obj, time, best_bound, gap = ILS_solution(
            ins, semilla = 0, acceptance = 0.05,
            feasibility_param = 1000, elite_param = 2500, elite_size = 20, p = 0.5,
            pa = 0.4, pb = 0.6, lsp = 0.8, initial_solution = initial_solution, #   initial_solution,
            elite_revision_param = 1500 , vis  = True, verbose = False, time_limit = 30 )

def test(q, a, f, e, s, n, x, y, z, r, l, t, g, nnodes, ins_folder, nombre):

    instances = os.listdir(ins_folder)
    results = list()
    for p in instances:
        print(p)
        best_obj = inf
        time_sum = 0
        solution_sum = 0
        pa = x
        pb = x + y

        name, capacity, node_data = read_instance(ins_folder + "/"+  p)
        ins = instance(name, capacity, node_data, nnodes)
        ins.capacity = q
        global xcoords, ycoords, latest, earliest, D, Q
        xcoords, ycoords = ins.xcoords, ins.ycoords
        earliest, latest = ins.earliest, ins.latest
        D = ins.cost
        Q = ins.capacity

        global BRANCH_TIME
        BRANCH_TIME = t

        initial_solution = prim(ins, vis = False, initial = True)
        generate_solution = lambda x: gurobi_solution(x, vis = False, time_limit= g, verbose = False, initial=True)
        (parent, gate, load, arrival), objective_value= generate_solution(ins)
        initial_solution = lambda x: ((parent.copy(), gate.copy(), load.copy(), arrival.copy()), objective_value)
        solution_sum = 0
        for i in range(10):
            obj, time, best_bound, gap = ILS_solution(
                ins, semilla = i, acceptance = a,
                feasibility_param = f, elite_param = e, elite_size = s, p = n,
                pa = pa, pb = pb, lsp = l, initial_solution = initial_solution,
                elite_revision_param = r, vis  = False, verbose = False, time_limit = 60 - g)
            if obj < best_obj:
                best_obj = obj
            time_sum += time
            solution_sum += obj
            
        dic = {"name": f"{name}","min": best_obj, "avg": solution_sum/10,  "t_avg": time_sum/10 }
        results.append(dic)

    df = pd.DataFrame(results)
    df.to_excel(f"{nombre}.xlsx", index= False)

if __name__ == "__main__":
    experimento = "% Local Search"
    capacities = [10000, 20, 15, 10, 5]
    g = 30
    a, f, e, s, n, x, y, z, r, l, t = 0.033 ,9500 ,10000 ,40 ,6.326 ,0.182 ,0.476 ,0.342 ,4000 ,0.18 ,4
    configuracion = f"conf40-1 PRIM {experimento}"
    for q in capacities:
        ins_folder = "Instances"
        test(q, a, f, e, s, n, x, y, z, r, l, t, g, 100, ins_folder, f"{configuracion} Q-{q} n-100")


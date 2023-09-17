from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from operators_cython_elite import perturbation, best_father, fitness

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
PERTURBATION2 = 0.1 + 0.7 * 0.392
PERTURBATION3 = PERTURBATION2 + 0.7 * (1 - 0.392)    
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
        self.maximum = D.max() * n

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

def verificar(s, texto):
    print(texto)
    if s.gate[0] != 0:
        print("la puerta de cero está mal asignada")
        exit(2)
    if s.load[0] > 0:
        print("la carga de cero está mal")
        exit(3)
    for i in range(len(s.parent)):
        if s.parent[i] == i:
            print("hay un nodo cuyo padre es si mismo:", i)
            exit(1)
        if s.load[i] > Q:
            print("Carga sobrepasada:", i)
            exit(1)
        if s.load[i] < 0:
            print("Carga negativa:", i)
            exit(1)

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

def get_disjoint(parent):
    n = len(parent)
    ds = DisjointSet() 
    for u in range(n): 
        ds.find(u) 

    for v in range(n): 
        u = parent[v]
        if u != -1:
            if ds.find(u) != ds.find(v):
                ds.union(u,v)
    return ds

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
        plt.annotate(str(i) ,xy = (xcoords[i],ycoords[i]), xytext = (xcoords[i]-0.6,ycoords[i]-0.6), color = 'black', zorder=4)
    plt.show()

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
        
    if vis:
        visualize(ins.xcoords, ins.ycoords, s)
    
    if initial:
        return s , cost
        
    else:
        best_bound = None
        gap = None
        return cost, time, best_bound, gap

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

    if start is not None:
        for j in nodesv:
            i = start.parent[j]
            x[(i,j)].Start = 1
            d[j].Start = start.arrival[j]

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

class branch_bound:
    def __init__(self, branch):
        self.nodes = [0] + branch
        self.best_solution = None
        self.best_cost = inf
        self.t = {num:j for j,num in enumerate(self.nodes)}
        self.decode = {j:num for j,num in enumerate(self.nodes)}
        self.decode[-1] = -1
        self.n = n = len(self.nodes)

        parent = np.zeros(n, dtype = int)
        gate = np.zeros(n, dtype = int)
        load = np.zeros(n, dtype = int)
        arrival = np.zeros(n)
        parent[0] = -1

        nodes_left = set(self.nodes[1:])
        itree = set()
        itree.add(0)
        
        s = (parent, gate, load, arrival)
        self.explore(s, 0, nodes_left, itree)

    def explore(self, s, cost, nodes_left, itree):
        if len(nodes_left) == 0:
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = deepcopy(s)
        else:
            for j in nodes_left:
                kj = self.t[j]
                for i in itree:
                    ki = self.t[i]
                    
                    if cost + distance(i, j) < self.best_cost:
                        
                        parent, gate, load, arrival = deepcopy(s)
                        parent[kj] = ki
                        if i == 0:
                            gate[kj] = kj
                        else:
                            gate[kj] = gate[ki]
    
                        if arrival[ki] + distance(i,j) <= latest[j] and load[gate[kj]] < Q: # isFeasible() # reescribir
                            
                            load[gate[kj]] += 1
                            arrival[kj] = arrival[ki] + distance(i,j)
                            if arrival[kj] < earliest[j]:
                                arrival[kj] = earliest[j]
                            
                            self.explore((parent, gate, load, arrival), cost + distance(i, j), nodes_left - {j}, itree | {j})
                            load[gate[kj]] -= 1
    def give_solution(self):
        parent = SortedDict()
        arrival = SortedDict()
        gate = SortedDict()
        load = SortedDict()
        for ki in range(self.n):
            i = self.decode[ki]
            parent[i] = self.decode[self.best_solution[0][ki]]
            gate[i] = self.decode[self.best_solution[1][ki]]
            load[i] = self.best_solution[2][ki]
            arrival[i] = self.best_solution[3][ki]
        return(parent, gate, load, arrival)

def branch_gurobi(branch, parent, initial = False):

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
                i = parent[j]
                x[(i,j)].Start = 1

    solution = mdl.optimize() 

    parent = SortedDict()
    departure = SortedDict()
    for i,j in edges:
        if x[(i,j)].X > 0.9:
            parent[j] = i
            departure[j] = d[j].X

    gate= SortedDict()
    load = { j : 0 for j in parent.keys()}
    arrival = SortedDict()
    arrival[0] = 0
    for j in sorted(parent.keys(), key = lambda x: departure[x]):
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

    return (parent, gate, load, arrival)   

def merge_branches(s):

    n = len(s.parent)

    ds = get_disjoint(s.parent)
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
    for i,j in enumerate(theta):
        if j > 2 * np.pi:
            theta[i] -= 2 * np.pi
        
    branches = list(range(nds))
    branches = sorted(branches, key = lambda x: theta[x])
    
    for i in range(nds//2):
        s1, s2 = i*2, i*2+1
        branch = ds[s1] + ds[s2]
        lo = len(branch)

        if lo < 5 and lo >= 2:
            bb = branch_bound(branch)
            aux = bb.give_solution()
        elif lo <= INITIAL_TRIGGER:
            aux = branch_gurobi(branch, s.parent)
        else:
            aux = branch_gurobi(branch, s.parent, initial = True)

        for j in aux[0].keys():
            s.parent[j] = aux[0][j]
            s.gate[j] = aux[1][j]
            s.load[j] = aux[2][j]
            s.arrival[j] = aux[3][j]

    cost, feasible = fitness(s, latest, PENALIZATION)
    return s, cost, feasible     

def optimal_branch(s):

    for i in set(s.gate):
        if i != 0:
            lo = s.load[i]
            if lo <= 20 and lo >= 2:
                branch = [j for j in range(1, len(s.parent)) if s.gate[j] == i]
                if lo < 5 and lo >= 2:
                    bb = branch_bound(branch)
                    aux = bb.give_solution()
                else:
                    aux = branch_gurobi(branch, s.parent)
                for j in branch:
                    s.parent[j] = aux[0][j]
                    s.gate[j] = aux[1][j]
                    s.load[j] = aux[2][j]
                    s.arrival[j] = aux[3][j]

    cost, feasible = fitness(s, latest, PENALIZATION)
    return s, cost, feasible         

def local_search(s):
    x = random()
    if x < LOCAL_SEARCH_PARAM1:
        # print("LA")
        return merge_branches(s)
    elif x < LOCAL_SEARCH_PARAM2:
        # print("LB")
        return best_father(s,1, latest, PENALIZATION)
    else:
        # print("LC")
        return best_father(s,5, latest, PENALIZATION)

def ILS_solution(ins, semilla = None, acceptance = 0.026,
                feasibility_param = 6000, elite_size = 30, iterMax = 15000, p = PENALIZATION,
                pp1 = PERTURBATION1, pp2 = PERTURBATION2, pp3 = PERTURBATION3,  lsp1 = LOCAL_SEARCH_PARAM1, lsp2 = LOCAL_SEARCH_PARAM2,  initial_solution = None,
                elite_revision_param = 1000, vis  = False, verbose = False, time_limit = 60, limit_type = "t"):
    global PENALIZATION, Q, PERTURBATION1, PERTURBATION2,PERTURBATION3, LOCAL_SEARCH_PARAM1, LOCAL_SEARCH_PARAM2, D
    PENALIZATION = p
    PERTURBATION1 = pp1
    PERTURBATION2 = pp2
    PERTURBATION3 = pp3
    LOCAL_SEARCH_PARAM1 = lsp1
    LOCAL_SEARCH_PARAM2 = lsp2
    
    if semilla is not None:
        np.random.seed(semilla)
        seed(semilla)

    start = perf_counter()

    if initial_solution is None:
        s, cost_best = prim(ins,  vis = False)
    else:
        s, cost_best = initial_solution(ins)

    s, candidate_cost = deepcopy(s), cost_best
    cost_best_unfeasible = inf
    feasible = True

    s_best = deepcopy(s)
    s_best_unfeasible = None

    elite = SortedDict()
    elite[cost_best] = [deepcopy(s),False]

    it = 0
    best_it = 0
 
    get_counter = lambda : it if limit_type != 't' else perf_counter() - start
    limit = iterMax if limit_type != 't' else time_limit

    while get_counter() < limit:

        s = perturbation(s, PERTURBATION1, PERTURBATION2, PERTURBATION3)
        s, candidate_cost, feasible = local_search(s)
        if feasible:
            if cost_best > candidate_cost:
                s_best = deepcopy(s)
                cost_best = candidate_cost
                best_it = it + 1
                elite[candidate_cost] = [deepcopy(s), False]
                if len(elite) > elite_size: 
                    elite.popitem()
        else:
            if cost_best_unfeasible > candidate_cost:
                s_best_unfeasible = deepcopy(s)
                cost_best_unfeasible = candidate_cost

        if verbose: print(it, candidate_cost)
        else:
            # count = get_counter()
            # text = f'{count:^10.2f}/{limit} [{"#"*int(count*50//limit):<50}] cost: {candidate_cost:^10.3f} best: {cost_best:^10.3f} it: {it+1}'
            # print(text, end = "\r")
            pass

        if candidate_cost > cost_best * (1 + acceptance) or not feasible:
            x = choice(elite.values())
            x = x[0]
            s = deepcopy(x)
        
        if (it + 1) % feasibility_param == 0:
            if s_best_unfeasible is not None:
                s = deepcopy(s_best_unfeasible)

        if (it + 1) % elite_revision_param == 0:
            for cost in elite:
                ss, rev = elite[cost]
                if not rev:
                    ss, cost_after, feasible = optimal_branch(deepcopy(ss))
                    # print(cost, "->", cost_after)
                    elite[cost][1] = True
                    if feasible and cost_after < cost:
                        elite[cost_after] = [deepcopy(ss), True]

                        if cost_after < cost_best:
                            s_best = deepcopy(ss)
                            cost_best = cost_after
                            best_it = it + 1

                    while len(elite) > elite_size:
                        elite.popitem()
        it += 1
    
    time = perf_counter() - start
    best_bound = None
    gap = None
    count = get_counter()
  
    if not verbose:     
        text = f'{count:^10.2f}/{limit} [{"#"*int(count*50//limit):<50}] cost: {candidate_cost:^10.3f} best: {cost_best:^10.3f} best_it: {best_it}/{it}'
        print(text)
    if vis: visualize(ins.xcoords, ins.ycoords, s_best)
    return cost_best, time, best_bound, gap

def main():
    name, capacity, node_data = read_instance("instances/r102.txt")
    ins = instance(name, capacity, node_data, 100)

    global xcoords, ycoords, latest, earliest, D, Q
    xcoords, ycoords = ins.xcoords, ins.ycoords
    earliest, latest = ins.earliest, ins.latest
    D = ins.cost
    Q = ins.capacity
    
    s, cost = prim(ins, vis = False, initial = True)
    print("prim:", cost)
    s, cost = gurobi_solution(ins, vis = False, time_limit= 30, verbose = False, initial=True, start = s)
    print("gurobi:", cost)
    initial_solution = lambda x: (deepcopy(s), cost)

    obj, time, best_bound, gap = ILS_solution(
            ins, semilla = 42, initial_solution = initial_solution, #   initial_solution,
            vis  = True, verbose = False, time_limit = 30 )
        
    print("ILS:", obj)

def test(gurobi_time, q, nnodes, ins_folder):
    global PENALIZATION ,PERTURBATION1, PERTURBATION2,PERTURBATION3 ,LOCAL_SEARCH_PARAM1 ,LOCAL_SEARCH_PARAM2 ,BRANCH_TIME ,INITIAL_TRIGGER
    instances = os.listdir(ins_folder)
    results = list()
    for p in instances:
        print(p)
        best_obj = inf
        time_sum = 0
        solution_sum = 0

        name, capacity, node_data = read_instance(ins_folder + "/"+  p)
        ins = instance(name, capacity, node_data, nnodes)
        ins.capacity = q
        initialize(ins)

        initial_solution, cost = prim(ins, vis = False, initial = True)
        print("prim:", cost)
        generate_solution = lambda x: gurobi_solution(x, vis = False, time_limit= gurobi_time, verbose = False, initial=True, start = initial_solution)
        sol, objective_value = generate_solution(ins)
        print("gurobi:", objective_value)
        initial_solution = lambda x: (deepcopy(sol), objective_value)
        solution_sum = 0
        
        for i in range(10):
            obj, time, best_bound, gap = ILS_solution(
                ins, semilla = i, initial_solution = initial_solution, vis  = False, 
                verbose = False, time_limit = 60 - gurobi_time)
            if obj < best_obj:
                best_obj = obj
            time_sum += time
            solution_sum += obj
            
        dic = {"name": f"{name}","min": best_obj, "avg": solution_sum/10,  "t_avg": time_sum/10 }
        results.append(dic)

    df = pd.DataFrame(results)
    nombre = f"conf{gurobi_time} PRIM% Q-{q} n-{nnodes}"
    df.to_excel(f"{nombre}.xlsx", index= False)

if __name__ == "__main__":
    main()
    exit(1)
    capacities = [10000, 20, 15, 10, 5]
    gurobi_time = 30
    nnodes = 100
    for q in capacities:
        ins_folder = "Instances"
        test(gurobi_time, q, nnodes, ins_folder)

    # import getopt,sys
    # argv = sys.argv[1:]
    # try:
    #     opts, args = getopt.getopt(argv, 'g:q:n:f')
    # except getopt.GetoptError:
    #     print ('matheuristic.py -g gurobi_time -q capacity -n nnodes -f instance_folder')
    #     sys.exit(2)

    # for opt, arg in opts:
    #     if opt == "-g":
    #         gurobi_time = int(arg)
    #     elif opt == "-q":
    #         q = int(arg)
    #     elif opt == "-n":
    #         nnodes = int(arg)
    #     elif opt == "-f":
    #         ins_folder = arg
    #     else: pass
    # ins_folder = "Instances"
    # print(gurobi_time, q, nnodes, ins_folder)
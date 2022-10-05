from msilib.schema import IniFile
from docplex.mp.model import Model
import numpy as np
from math import sqrt, inf

from utilities import extract_data, read_instance,  visualize
from time import perf_counter
from sortedcollections import SortedDict

class instance():
    def __init__(self, name, capacity, node_data, num, reset_demand = True, Q = None):
        self.name = name
        self.n = num+1
        self.capacity = int(capacity)
        self.index, self.xcoords, self.ycoords, self.demands, self.earliest, self.latest\
            = extract_data(node_data[:num+1])

        self.nodes = [i for i in range(num+1)]
        self.nodesv = self.nodes[1:]

        # problem edges: {(i,j): i,j \in nodes ^ }
        self.edges = [(i,j) for i in self.nodes for j in self.nodesv if i != j]

        #demands = 1 for all nodes 
        if reset_demand:
            self.demands = {i:1 for i in self.nodes}

        # cost = time = distance for simplicity
        nnodes = num + 1

        global D
        D = np.zeros((nnodes,nnodes))
        for i in range(nnodes):
            for j in range(i+1,nnodes):
                D[i,j] = self.dist(i,j)
                D[j,i] = D[i,j]

        self.D = D

    def dist(self,i,j):
        x = self.xcoords[i] - self.xcoords[j]
        y = self.ycoords[i] - self.ycoords[j]
        return sqrt(x**2 + y**2)

def distance(i,j):
    return D[(i,j)]

def cplex_solution(ins, vis = False, time_limit = 1800, verbose = False):

    nodes = ins.nodes
    nnodes = ins.n
    edges = ins.edges
    nodesv = nodes[1:]
    Q = ins.capacity
    earliest = ins.earliest
    latest = ins.latest

    demands = [1 for i in nodes]

    # model and variables
    mdl = Model(ins.name)
    x = mdl.binary_var_dict(edges, name = "x") #
    y = mdl.integer_var_dict(edges, name = "y", lb = 0)
    d = mdl.continuous_var_dict(nodes, name = "d", lb = 0)

    # objective function
    mdl.minimize(mdl.sum(distance(i,j) * x[(i,j)] for i,j in edges))

    # restrictions
    for j in nodesv:
        mdl.add_constraint(mdl.sum(x[(i,j)] for i in nodes if i!=j) == 1)

    for j in nodesv:
        mdl.add_constraint(mdl.sum(y[(i,j)] for i in nodes if i!=j) - mdl.sum(y[(j,i)] for i in nodesv if i!=j) == demands[j])
    
    for i,j in edges:
        mdl.add_constraint(x[(i,j)] <= y[(i,j)])
    
    for i,j in edges:
        mdl.add_constraint(y[(i,j)] <= (Q - demands[i]) * x[(i,j)])

    #  M = max(latest) + max(cost.values()) + 1
    for i,j in edges:
        mdl.add_indicator(x[(i,j)], d[i] + distance(i,j) <= d[j])

    for i in nodes:
        mdl.add_constraint(d[i] >= earliest[i])
    
    for i in nodes:
        mdl.add_constraint(d[i]  <= latest[i])

    mdl.parameters.timelimit = time_limit # timelimit = 30 minutes
    mdl.parameters.threads = 1 # only one cpu thread in use
    solution = mdl.solve(log_output = False)

    solution_edges = np.ones(nnodes, dtype = int)*-1
    for i,j in edges:
        if x[(i,j)].solution_value > 0.9:
            solution_edges[j] = i

    objective_value = mdl.objective_value
    time = mdl.solve_details.time
    best_bound = mdl.solve_details.best_bound
    gap = mdl.solve_details.mip_relative_gap

    # to display the solution given by cplex
    if verbose == True: 
        solution.display()
    # to visualize the graph 
    if vis: 
        visualize(ins.xcoords, ins.ycoords, solution_edges) 

    return objective_value, time, best_bound, gap

def algorithm(ins, b, alpha, s, mu, vis  = False, Q = None, initial = False):
    nodes = ins.nodes
    nnodes = ins.n
    global D
    D = ins.D

    if Q is None:
        Q = ins.capacity

    earliest = ins.earliest
    latest = ins.latest

    start = perf_counter()

    M = 100**5
    tol = 1**-7

    LPDH = np.zeros((nnodes,nnodes))
    for i in range(nnodes):
        for j in range(i+1,nnodes):
            LPDH[i,j] = 0 if distance(i,j) <= s else 1
            LPDH[j,i] = LPDH[i,j]

    ESGC = lambda j, sj, gsj : b[0]*D[(sj,j)] + b[1] * D[(j,gsj)] + b[2] * D[(sj,gsj)] + b[3] / abs(D[(0,sj)] - D[(0,j)] + tol) + b[4] / (D[(0,j)] - alpha * D[(sj,j)] + tol) + b[5]* abs(latest[j] - (latest[sj] + D[(sj,j)]))
    criterion = lambda j,sj,gsj: ESGC(j,sj,gsj) + mu * M *LPDH[(sj,j)]

    start = perf_counter()

    pred = SortedDict()
    arrival_time = SortedDict()
    waiting_time = SortedDict()
    gate = SortedDict()
    load = SortedDict()
    for i in range(nnodes):
        pred[i] = -1
        arrival_time[i] = 0
        waiting_time[i] = 0
        gate[i] = 0
        load[i] = 0
        
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
    pred[v] = 0
    arrival_time[v] = cost = d
    waiting_time[v] = 0

    if not  arrival_time[v] >= earliest[v]:
        waiting_time[v] = earliest[v]- arrival_time[v]
        arrival_time[v] = earliest[v]

    while len(nodes_left) > 0:
        min_tree = inf
        for j in nodes_left:
            min_node = inf
            for ki in itree:# k: parent, j: offspring
                # calcula si alcanza a llegar desde alguno de los nodos que ya estan colocados
                dkj = distance(ki,j)
                # criterion = dkj
                tj = arrival_time[ki] + dkj
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
        pred[jj] = kk
        # visualize(ins.xcoords, ins.ycoords, pred)
        cost += distance(kk,jj)
        if gate[kk] == 0:
            gate[jj] = jj
        else:
            gate[jj] = gate[kk]
        load[gate[jj]] += 1
        
        arrival_time[jj] = arrival_time[kk] + distance(kk,jj)

        
        if not arrival_time[jj] >= earliest[jj]:
            waiting_time[jj] = earliest[jj] - arrival_time[jj]
            arrival_time[jj] = earliest[jj]

    time = perf_counter() - start
    solution_edges = [(i,j) for j,i in enumerate(pred) if i is not None]    
    if vis:
        visualize(ins.xcoords, ins.ycoords, solution_edges)
    best_bound = None
    gap = None

    if initial:
        return (pred, gate, load, arrival_time) , cost
    return pred, cost, time, best_bound, gap

def SGH_solution(ins, vis  = False, initial = False):
    return algorithm(ins, b = np.array([1,0,0,0,0,0]), alpha = 0, s = 0, mu = 0, vis  = vis, Q = 10000000, initial = initial)

def SGHC_solution(ins, vis  = False, initial = False): # 
    return algorithm(ins,  b = np.array([1,0,0,0,0,0]), alpha = 0, s = 0, mu = 0,  vis  = vis, initial = initial)

def ESGH_solution(ins, vis  = False, initial = False, b = np.array([1,0,1,0.2,0.4,0]), alpha = 1):
    return algorithm(ins,  b = b, alpha = alpha, s = 0, mu = 0, vis  = vis, initial = initial)

def LPDH_solution(ins,  vis  = False, initial = False, b = np.array([1,0,1,0.2,0.4,0]), alpha = 1, s = 7, mu = 1):
    return algorithm(ins,  b = b, alpha = alpha, s = s, mu = mu, vis  = vis, initial = initial)

def main():
    name, capacity, node_data = read_instance("instances/c101.txt")
    ins = instance(name, capacity, node_data, 100)

    s, obj, time, best_bound, gap = SGH_solution(ins,vis = False)
    # print(solution_edges)
    print("--------")
    print(f"objective value {obj}")
    print(f"time: {time}")
    print(f"best bound: {best_bound}")
    print(f"optimality gap: {gap}")
    return None

def test():
    name, capacity, node_data = read_instance("instances/c101.txt")
    ins = instance(name, capacity, node_data, 100)
    ini = perf_counter()
    mejor = inf
    bb = [0,0.5,1]

    values = np.array(np.meshgrid(1,bb,bb,bb,bb,bb)).T.reshape(-1,6)

    for i in range(len(values)):
        b = values[i]

        pred, obj, time, best_bound, gap = LPDH_solution(ins, b = b, vis = False)

        if obj < mejor:
            mejor = obj
            b_mejor  = b.copy()
            print(b_mejor, obj)
    print(perf_counter() - ini)
    return None

if __name__ == "__main__":
    test()

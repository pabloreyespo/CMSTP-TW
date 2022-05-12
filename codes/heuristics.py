from docplex.mp.model import Model
import numpy as np
from math import sqrt, inf

from sympy import O
from utilities import extract_data, extract_data_R7, extract_data_R7, read_salomon,  visualizar, read_R7

import networkx as nx
import matplotlib.pyplot as plt

from time import perf_counter

class CMSTP_ATW():
    def __init__(self, name, capacity, nodes):
        self.name = name
        self.capacity = int(capacity)
        self.nodes = nodes
        self.index, self.xcoords, self.ycoords, self.demands, self.earliest_times, self.latest_times\
            = extract_data(nodes)

    def dist(self,i,j):
        x = self.xcoords[i] - self.xcoords[j]
        y = self.ycoords[i] - self.ycoords[j]
        return round(sqrt(x**2 + y**2),4)

    def cplex_solution(self, num = 25, vis = False, Q = None, verbose = False):

        if Q is None:
            Q = self.capacity

        # nodes coordinates
        xcoords, ycoords = self.xcoords[:num+1],self.ycoords[:num+1]

        # nodes indexes, nodesv = nodes - {0}
        nodes = [i for i in range(num+1)]
        nodesv = nodes[1:]
        # problem edges: {(i,j): i,j \in nodes ^ }
        edges = [(i,j) for i in nodes for j in nodesv if i != j]

        earliest = self.earliest_times
        latest = self.latest_times

        demands = {i:1 for i in nodes}

        cost = {(i,j): self.dist(i,j) for i,j in edges}

        mdl = Model(self.name)
        

        x = mdl.binary_var_dict(edges, name = "x")
        y = mdl.integer_var_dict(edges, name = "y", lb = 0)
        d = mdl.continuous_var_dict(nodes, name = "s", lb = 0)

        # Cálculos de big-M

        # M = max(latest.values()) + max(cost.values()) - min(latest.values()) + 1
        M1 = max(latest) + max(cost.values()) + 1
        M = M1
        print(f"m: {M}")
        mdl.minimize(mdl.sum(cost[(i,j)] * x[(i,j)] for i,j in edges))

        for j in nodesv:
            mdl.add_constraint(mdl.sum(x[(i,j)] for i in nodes if i!=j) == 1)

        for j in nodesv:
            mdl.add_constraint(mdl.sum(y[(i,j)] for i in nodes if i!=j) - mdl.sum(y[(j,i)] for i in nodesv if i!=j) == demands[j])

        for i,j in edges:
            mdl.add_constraint(x[(i,j)] <= y[(i,j)])

        for i,j in edges:
            mdl.add_constraint(y[(i,j)] <= (Q - demands[i]) * x[(i,j)])

            # si uso edgesv en vez de edges la restricción de Q desaparece, sin embargo genera la misma topologia

        for i,j in edges:
            mdl.add_indicator(x[(i,j)], d[i] + cost[(i,j)] <= d[j])

        for i in nodes:
            mdl.add_constraint(d[i] >= earliest[i])

        for i in nodes:
            mdl.add_constraint(d[i]  <= latest[i])
            # mdl.add_indicator(x[(i,j)], s[(i,j)] + cost[(i,j)] <= latest[(i,j)])

        # msg = mdl.export_to_string()
        mdl.parameters.timelimit = 1800
        mdl.parameters.threads = 1
        solution = mdl.solve(log_output = True)
        print(mdl.get_solve_status())
        solution.display() #estas lienas se encuentran como comentario porque preferí simplificar la información que despliega el programa
    
        solution_edges = [i for i in edges if x[i].solution_value > 0.9]
        objective_value = mdl.objective_value

        if vis:
            visualizar(xcoords, ycoords, solution_edges)

        return solution_edges, objective_value 

    def heuristic_solution(self, num = 25, vis = False, Q = None, verbose = False):
        
        if Q is None:
            Q = self.capacity

        xcoords = self.xcoords[:num+1]
        ycoords = self.ycoords[:num+1]

        nodes = [i for i in range(num+1)]
        nodesv = nodes[1:]

        edges = [(i,j) for i in nodes for j in nodesv if i != j]
        edgesv = [(i,j) for i in nodesv for j in nodesv if i != j]

        earliest = self.earliest_times
        latest = self.latest_times

        demands = {i:1 for i in nodes}

        cost = {(i,j): self.dist(i,j) for i,j in edges}

        solution_edges = None

        if vis:
            visualizar(xcoords, ycoords, solution_edges)

    def prim_solution(self, num = 25, vis  = False):

        xcoords = self.xcoords[:num+1]
        ycoords = self.ycoords[:num+1]

        nodes = [i for i in range(num+1)]
        nodesv = nodes[1:]

        edges = [(i,j) for i in nodes for j in nodesv if i != j]
        cost = {(i,j): self.dist(i,j) for i,j in edges}

        coordinates = np.column_stack((xcoords, ycoords))
        positions = dict(zip(nodes, coordinates))

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from([(i,j,cost[(i,j)]) for (i,j) in cost])

        for p in positions:
            G.nodes[p]['pos'] = positions[p]

        pi, key = prim(G, cost)

        solution_edges = [(i,j) for j,i in enumerate(pi) if i is not None]
        obj = sum(key)

        if vis:
            visualizar(xcoords, ycoords, solution_edges)
        return solution_edges, obj

    def SGH(self,num = 25, vis  = False):
        xcoords = self.xcoords[:num+1]
        ycoords = self.ycoords[:num+1]

        nodes = [i for i in range(num+1)]
        nodesv = nodes[1:]

        edges = [(i,j) for i in nodes for j in nodesv if i != j]
        cost = {(i,j): self.dist(i,j) for i,j in edges}

        coordinates = np.column_stack((xcoords, ycoords))
        positions = dict(zip(nodes, coordinates))

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from([(i,j,cost[(i,j)]) for (i,j) in cost])

        for p in positions:
            G.nodes[p]['pos'] = positions[p]

        etime = self.earliest_times[:num+1]
        timel = self.latest_times[:num+1]
        pi, cost = solomon(G, cost, etime, timel)

        solution_edges = [(i,j) for j,i in enumerate(pi) if i is not None]
        obj = cost

        if vis:
            visualizar(xcoords, ycoords, solution_edges)
        return solution_edges, obj


def solomon(G,D,etime, timel):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    n = len(G.nodes())
    key = [inf]*n # el mínimo desde i a j entre los disponibles
    pi = [None]*n # predecesor
    time = [0]*n #tiempo de visita
    wti = [0]*n #tiempo de espera, del nodo
    D1 = {j: D[(0,j)] for j in range(1,n)}
    it = [0]*n #esta actuando como una lista de salida
    itree = [0]*n

    itree[0] = 0 #orden en que son nombrados
    P = list(G.nodes())[1:]
    P.sort(key = (lambda x: D1[x]), reverse = True) 
    # hace un primer paso fuera del ciclo
    v = P.pop()
    itree[1] = v
    pi[v] = 0
    time[v] = D[(0,v)]
    wti[v] = 0
    if not  time[v] >= etime[v]:
        wti[v] = etime[v]- time[v]
        time[v] = etime[v]

    numnod = 2
    cost = D[(0,v)]
    while numnod < n:
        min1 = 10**7
        for j in range(n):
            minim = 10**7
            it[j] = 0
            index = 0
            #verifica que solo tenga una salida
            for l in range(numnod):
                if itree[l] == j:
                    index = 1
            if index == 1:
                continue
            for k in range(numnod):
                # calcula si alcanza a llegar desde alguno de los nodos que ya estan colocados
                pos = itree[k]
                dkj = D[(pos,j)]
                tj = time[pos] + dkj
                if tj > timel[j]:
                    continue
                if tj < etime[j]:
                    tj = etime[j]
                if dkj > minim:
                    continue
                minim = dkj
                it[j] = k
            crit2 = minim
            if crit2 > min1:
                continue
            itj = it[j]
            jj = j
            min1 = crit2

        numnod += 1
        itree[numnod-1] = jj
        pi[jj] = itree[itj]
        cost += D[(pi[jj],jj)]
        time[jj] = time[pi[jj]] + D[(pi[jj],jj)]
        if not time[jj] >= etime[jj]:
            wti[jj] = etime[jj]-time[jj]
            time[jj] = etime[jj]
    return pi, cost

def prim(G,w):
    u = 0
    n = len(G.nodes())
    key = [inf]*n
    pi = [None]*n
    key[0] = 0
    
    Q = sorted(G.nodes(), reverse = True)
    while len(Q)!=0:
        u = Q.pop()
        for v in G.neighbors(u):
            if v in Q and w[(u,v)] < key[v]:
                # añadir chequeo de restriccion de tiempo
                pi[v] = u
                key[v] = w[(u,v)]
    return pi, key

def main():
    start = perf_counter()
    name, capacity, nodes = read_salomon("instances/r103.txt")
    ins = CMSTP_ATW(name, capacity, nodes)
    start = perf_counter()
    solution_edges, obj = ins.SGH(100,vis = False)
    print(f"time: {perf_counter()-start}")
    # print(solution_edges)
    print(obj)
    
    return None

if __name__ == "__main__":
    main()

"""
POR HACER:
1. SGH: Solomon Greedy Heuristic
2. ESGH: Enhanced Solomon Greedy Heuristic
3. Correr todo con máximo media hora
"""

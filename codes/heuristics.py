import numpy as np
from math import inf

from utilities import *
from time import perf_counter
from sortedcollections import SortedDict
from numpy.random import random

def distance(i,j):
    return D[(i,j)]

def algorithm(ins, b, alpha, s, mu, vis  = False, Q = None, initial = False):
    nodes = ins.nodes
    nnodes = ins.n
    global D
    D = ins.cost

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

def prim_solution(ins, vis  = False):

    nodes = ins.nodes # ignore demand
    start = perf_counter()
    nnodes = len(nodes)

    pred = SortedDict()
    arrival_time = SortedDict()
    for i in range(nnodes):
        pred[i] = -1
        arrival_time[i] = 0

    itree = set() # muestra que es lo ultimo que se ha añadido

    d = inf
    for j in nodes[1:]:
        if distance(0,j) < d:
            d = distance(0,j)
            v = j

    itree.add(0) #orden en que son nombrados
    itree.add(v)

    pred[v] = 0
    arrival_time[v] = cost = d

    numnod = 2

    while numnod < nnodes: # mientras no estén todos los nodos
        min_tree = inf
        for j in range(nnodes): # par cada nodo candidato
            min_node = inf
            # Dado un nodo j que no está en el árbol, busco el mejor predecesor
            if j not in itree:
                for ki in itree: # k: parent, j: offspring
                    # calcula si alcanza a llegar desde alguno de los nodos que ya estan colocados
                    dkj = distance(ki,j)
                    crit_node = dkj
                    if crit_node < min_node:
                        min_node = crit_node
                        k = ki

                crit_tree = min_node
                if crit_tree < min_tree:
                    kk = k
                    jj = j
                    min_tree = crit_tree

        numnod += 1
        itree.add(jj)
        pred[jj] = kk
        # visualize(ins.xcoords, ins.ycoords, pred)
        cost += distance(kk,jj)
        arrival_time[jj] = arrival_time[kk] + distance(kk,jj)

    time = perf_counter() - start

    if vis:
        visualize(ins.xcoords, ins.ycoords, pred)
    best_bound = None
    gap = None
    return cost, time, best_bound, gap

def random_prim_solution(ins, Q = None, vis  = False):

    global D
    D = ins.cost

    if Q is None:
        Q = ins.capacity

    earliest = ins.earliest
    latest = ins.latest

    nodes = ins.nodes # ignore demand
    start = perf_counter()
    nnodes = len(nodes)

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
        if distance(0,j) < d:
            d = distance(0,j)
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
        min_tree = [inf]
        kk = [-1]
        jj = [-1]
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
            crit_tree = crit_node

            if crit_tree < min_tree[-1]:
                    kk.append(k)
                    jj.append(j)
                    min_tree.append(crit_tree)
        
        pos = -1
        while True:
            if random() < 0.5 or abs(pos) == len(jj) - 1:
                break
            else:
                pos -= 1

        jj = jj[pos]
        kk = kk[pos]
        min_tree = min_tree[pos]

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

    if vis:
        visualize(ins.xcoords, ins.ycoords, pred)
    best_bound = None
    gap = None
    return cost, time, best_bound, gap

def SGH_solution(ins, vis  = False, initial = False):
    return algorithm(ins, b = np.array([1,0,0,0,0,0]), alpha = 0, s = 0, mu = 0, vis  = vis, Q = 10000000, initial = initial)

def SGHC_solution(ins, vis  = False, initial = False): # 
    return algorithm(ins,  b = np.array([1,0,0,0,0,0]), alpha = 0, s = 0, mu = 0,  vis  = vis, initial = initial)

def ESGH_solution(ins, vis  = False, initial = False, b = np.array([1,0,1,0.2,0.4,0]), alpha = 1):
    return algorithm(ins,  b = b, alpha = alpha, s = 0, mu = 0, vis  = vis, initial = initial)

def LPDH_solution(ins,  vis  = False, initial = False, b = np.array([1,0,1,0.2,0.4,0]), alpha = 1, s = 7, mu = 1):
    return algorithm(ins,  b = b, alpha = alpha, s = s, mu = mu, vis  = vis, initial = initial)

def main():
    name, capacity, node_data = read_instance(f"instances/c101.txt")
    ins = instance(name, capacity, node_data, 100)
    ins.capacity = 20

    obj, time, best_bound, gap = random_prim_solution(ins, vis = True)
    print(obj, time, best_bound, gap)

if __name__ == "__main__":
    main()

import gurobipy as gp
from gurobipy import GRB

import numpy as np
from math import inf
from random import choice, seed, random, sample
from sortedcollections import SortedDict
from time import perf_counter
from disjoint_set import DisjointSet

from heuristics import LPDH_solution # usar LPDH solution u otra mejor
from exact_solutions_gurobi import gurobi_solution
from utilities import read_instance,  visualize, instance

def update_tau(Tau,rho,s,cost):
    Tau *= (1-rho)
    parent = s[0]
    
    #actualización global
    for j in parent.keys()[1:]:
        Tau[parent[j],j] += 1/cost # tiene que usarse siempre en orden padre-hijo

    Tau[Tau < TauMin] = TauMin
    Tau[Tau > TauMax] = TauMax
    return Tau

def explore(ins, atractive):

    nodes = ins.nodes
    nnodes = ins.n
    global D
    D = ins.cost
    Q = ins.capacity

    earliest = ins.earliest
    latest = ins.latest

    start = perf_counter()

    pred = np.ones(nnodes) * -1
    arrival_time = np.zeros(nnodes)
    waiting_time = np.zeros(nnodes)
    gate = np.zeros(nnodes)
    load = np.zeros(nnodes)
        
    itree = set() # muestra que es lo ultimo que se ha añadido
    nodes_left = set(nodes)

    itree.add(0) #orden en que son nombrados
    nodes_left.remove(0)

    while len(nodes_left) > 0:
        for j in nodes_left:
            for ki in itree:# k: parent, j: offspring
                p_vector = atractive[ki, list(nodes_left)]

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

def ant_solution(ins,initial_solution = None, acceptance = 0.05, semilla = None, vis  = False, verbose = False, limit = 10, limit_type = "t",
                    poblacion =10, iterMax=200, alpha = 1, beta=2, rho = 0.2, q0 = 0, Tau_period = 10000, update_param = 20):
    """
    limit_type: 't' if time 'it if iterations
    q0: probabilidad de elegir derechamente el que tiene mejor probabilidad
    """
    
    if semilla is not None:
        np.random.seed(semilla)
        seed(semilla)

    it = 0
    start = perf_counter()
    copy = lambda s: (s[0].copy(), s[1].copy(), s[2].copy(), s[3].copy())
    
    global Q, earliest, latest
    D = ins.cost
    for i in range(len(D)):
        D[i,i] = inf
    Q = ins.capacity
    earliest = ins.earliest
    latest = ins.latest
    n = len(D)

    # inicializar algoritmo

    start = perf_counter()
    get_counter = lambda : it if limit_type != 't' else perf_counter() - start

    if initial_solution is not None: 
        s, cost_best = initial_solution(ins)
    else:
        s, cost_best = 1, 1
        pass

    eta = 1/D
    avg = n/2

    global TauMax, TauMin
    TauMax = 1/(rho*cost_best)
    TauMin = TauMax*(1-(0.05)**(1/n))/((avg-1)*(0.05)**(1/n))
    Tau = np.full((n,n),TauMax) 

    while get_counter() < limit:
        atractive = (Tau ** alpha) * (eta) ** beta 
        local_cost = inf #iteration best
        for _ in range(poblacion): #cada individuo tiene que construir su camino
            s, candidate_cost = explore(atractive)
            if candidate_cost < local_cost:
                local_cost = candidate_cost
                s_local = copy(s)

        if cost_best > local_cost:
            s_best, cost_best = copy(s_local), local_cost

        if iter % Tau_period == 0:
            Tau = np.full((n,n),TauMax) 
        elif iter%update_param == 0:
            Tau = update_tau(Tau,rho,s_best,cost_best)
        else: 
            Tau = update_tau(Tau,rho,s_local,local_cost)

        if verbose: print(it, candidate_cost)
        else:
            count = get_counter()
            text = f'{count:^6.2f}/{limit} [{"#"*int(count*50//limit):<50}] cost: {candidate_cost:^8.3f} best: {cost_best:8^.3f}'
            print(text, end = "\r")
            pass
        
        #if feasible: feasible_count += 1
        #else: unfeasible_count += 1

        # costs_list.append(candidate_cost)
        # bestCosts_list.append(cost_best)
        # solutions_list.append(s)
        # feasibility_list.append(feasible)

        if abs(cost_best - candidate_cost) / cost_best > acceptance:
            s = (s_best[0].copy(), s_best[1].copy(), s_best[2].copy(), s_best[3].copy())
        
        #if (it + 1) % feasibility_param == 0:
        #    try: s = (s_best_unfeasible[0].copy(), s_best_unfeasible[1].copy(), s_best_unfeasible[2].copy(), s_best_unfeasible[3].copy())
        #    except: s = s

        # if (it + 1) % elite_param == 0:
        #     x = choice(elite.values())
        #     x = x[0]
        #     s = (x[0].copy(), x[1].copy(), x[2].copy(), x[3].copy())

        # if (it + 1) % elite_revision_param == 0:
        #     for cost in elite:
        #         ss, rev = elite[cost]
        #         if not rev:
        #             ss, cost_after, feasible = optimal_branch((ss[0].copy(), ss[1].copy(), ss[2].copy(), ss[3].copy()))
        #             # print(cost, "->", cost_after)
        #             elite[cost][1] = True
        #             if feasible and cost_after < cost:
        #                 elite[cost_after] = [(ss[0].copy(), ss[1].copy(), ss[2].copy(), ss[3].copy()), True]

        #                 if cost_after < cost_best:
        #                     s_best = (ss[0].copy(), ss[1].copy(), ss[2].copy(), ss[3].copy())
        #                     cost_best = cost_after
        #                     best_it = it + 1

        #             while len(elite) > elite_size:
        #                 # print("size elite: ", len(elite))
        #                 elite.popitem()
        it += 1
    
    time = perf_counter() - start
    best_bound = None
    gap = None

    if vis: visualize(ins.xcoords, ins.ycoords, s_best[0])
    if not verbose:
        text = f'{count:^6}/{limit} [{"#"*int(count*50//limit):<50}] cost: {candidate_cost:^8.3f} best: {cost_best:8^.3f}'
        print(text)

    return cost_best, time, best_bound, gap


def main():
    name, capacity, node_data = read_instance('Instances/r101.txt')
    ins = instance(name, capacity, node_data, 100)
    cost_best, time, best_bound, gap = ant_solution(ins)
    pass

if __name__ == "__main__":
    main()


# versión preparada para  ver que pasaría si yo hiciera una versión distinta del problema en que se penalizara las infactibiidades
# puede ser bastante más interesante en el caso de las colonias de hormigas
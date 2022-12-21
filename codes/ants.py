import gurobipy as gp
from gurobipy import GRB

import numpy as np
from math import inf
from random import choice, seed, random, sample
from sortedcollections import SortedDict
from time import perf_counter
from disjoint_set import DisjointSet

from heuristics import LPDH_solution # usar LPDH solution u otra mejor
from exact_solutions_gurobi import gurobi_initial_array
from utilities import read_instance,  visualize_ants, instance
from heuristics import LPDH_solution_vector

def update_tau(Tau,rho,s,cost):
    Tau *= (1-rho)
    parent = s[0]
    
    #actualización global
    for j,k in enumerate(parent):
        if j != 0:
            Tau[k,j] += 1/cost # tiene que usarse siempre en orden padre-hijo

    Tau[Tau < TauMin] = TauMin
    Tau[Tau > TauMax] = TauMax
    return Tau

def explore(ins, p):
    Q = ins.capacity
    n = ins.n
    nodes = np.array(range(n), dtype = int)
    pred = np.ones(n, dtype = int) * -1
    gate = np.zeros(n, dtype = int)
    load = np.zeros(n, dtype = int)
    arrival = np.zeros(n)

    next = np.ones((n,n))

    itree = [0]
    nodes_left = set(nodes) - set([0])

    cost = 0
    while len(nodes_left) > 0:
        p[(load[gate][:,None] + next > Q) | (arrival[:,None] + D > latest)] = 0 # establecer filtro
        nleft = list(nodes_left)
        prob = p[itree,:][:, nleft].sum(axis = 1)
        kk  = np.random.choice(itree, p = prob/prob.sum(), size  =1)[0]

        prob = p[kk, nleft]
        jj = np.random.choice(nleft, p = prob/prob.sum(), size  =1)[0]

    # ver si tiene sentido meter esto en una función de actualización para acortar el código
        itree.append(jj)
        nodes_left.remove(jj)
    
        pred[jj] = kk

        
        if gate[kk] == 0:
            gate[jj] = jj
        else:
            gate[jj] = gate[kk]
            
        load[gate[jj]] += 1
        arrival[jj] = arrival[kk] + D[kk,jj]

        if not arrival[jj] >= earliest[jj]:
            arrival[jj] = earliest[jj] 

        cost += D[kk,jj]

    if (latest - arrival  < 0).any():
        print(pred)
        print(gate)
        print(load)
        print(arrival)
        exit(0)
    return (pred, gate, load, arrival), cost

def ant_solution(ins,initial_solution = None,  semilla = None, vis  = False, verbose = False, limit = 10, limit_type = "t",
                    poblacion = 5, alpha = 1, beta=1, rho = 0.15, q0 = 0, Tau_period = 100, update_param = 5):
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


    # inicializar algoritmo

    start = perf_counter()
    get_counter = lambda : it if limit_type != 't' else perf_counter() - start

    if initial_solution is not None: 
        s_best, cost_best = initial_solution(ins)
    else:
        s_best, cost_best = explore(ins, np.ones((n,n)))
        pass

    global Q, earliest, latest, D
    D = ins.cost
    for i in range(ins.n):
        D[i,i] = inf
    Q = ins.capacity
    earliest = np.array(ins.earliest)
    latest = np.array(ins.latest)
    n = len(D)

    eta = 1/D
    # epsilon = 1/latest[:,None]
    avg = n/2

    global TauMax, TauMin
    TauMax = 1/(rho*cost_best)
    TauMin = TauMax*(1-(0.05)**(1/n))/((avg-1)*(0.05)**(1/n))
    Tau = np.full((n,n),TauMax) 

    update_tau(Tau,rho,s_best,cost_best)

    while get_counter() < limit:
        atractive = (Tau ** alpha) * (eta ** beta) # * epsilon
        local_cost = inf #iteration best
        for _ in range(poblacion): #cada individuo tiene que construir su camino
            s, candidate_cost = explore(ins, atractive.copy())
            if candidate_cost < local_cost:
                local_cost = candidate_cost
                s_local = copy(s)
                # Tau = update_tau(Tau,rho,s_local,local_cost)

        if cost_best > local_cost:
            s_best, cost_best = copy(s_local), local_cost

        if it % Tau_period == 0:
            Tau = np.full((n,n),TauMax) 
        elif it%update_param == 0:
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

    if vis: visualize_ants(ins.xcoords, ins.ycoords, s_best[0])
    if not verbose:
        text = f'{count:^6}/{limit} [{"#"*int(count*50//limit):<50}] cost: {candidate_cost:^8.3f} best: {cost_best:8^.3f}'
        print(text)

    return cost_best, time, best_bound, gap

def main():
    from exact_solutions_gurobi import gurobi_solution
    # name, capacity, node_data = read_instance('Instances/r108.txt')
    # ins = instance(name, capacity, node_data, 100)
    # ins.capacity = 10
    
    name, capacity, node_data = read_instance('Instances/r101.txt')
    ins = instance(name, capacity, node_data, 100)
    ins.capacity = 5
    
    print("Valor objetivo r108-10: 646.40")
    print(f"gurobi 20s: 648.24")

    # generate_solution = lambda x: gurobi_initial_array(x, vis = False, time_limit= 20, verbose = False)
    generate_solution = lambda x: LPDH_solution_vector(ins)
    (parent, gate, load, arrival), objective_value= generate_solution(ins)
    initial_solution = lambda x: ((parent.copy(), gate.copy(), load.copy(), arrival.copy()), objective_value)
    print(f"solucion inicial: {objective_value}")
    for i in range(10):
        cost_best, time, best_bound, _ = ant_solution(ins,limit = 60, vis = False, semilla = i, initial_solution=initial_solution,
                                        poblacion = 5, alpha = 2, beta = 1.5, rho  = 0.2, Tau_period=200, update_param=10)

if __name__ == "__main__":
    main()


# versión preparada para  ver que pasaría si yo hiciera una versión distinta del problema en que se penalizara las infactibiidades
# puede ser bastante más interesante en el caso de las colonias de hormigas
import gurobipy as gp
from gurobipy import GRB

import numpy as np
from math import inf
from random import choice, seed, sample, random
from sortedcollections import SortedDict
from time import perf_counter
from disjoint_set import DisjointSet

from heuristics import LPDH_solution, prim # usar LPDH solution u otra mejor
from exact_solutions_gurobi import gurobi_solution
from utilities import read_instance,  visualize, instance

PENALIZATION = 0.5
PERTURBATION_A = 1
PERTURBATION_B = 0
LOCAL_SEARCH_PARAM = 1 # best_father
BRANCH_TIME = 1
INITIAL_TRIGGER = 20
LOCAL_SEARCH_PROPORTION = 0.02

env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

def copy(s):
    return (s[0].copy(), s[1].copy(), s[2].copy(), s[3].copy())

def distance(i,j):
    return D[(i,j)]

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
    for u in range(1,n): 
        ds.find(u) 

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
    theta = np.arctan2(y_sets, x_sets) + (np.random.rand() * (2 * np.pi))
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

    for i in sorted(con, key = lambda x: np.random.random()):
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
    if np.random.random() < LOCAL_SEARCH_PROPORTION:
        return merge_branches(s)
    else:
        return best_father(s)

def ILS_solution(ins, semilla = None, acceptance = 0.05, b = [1,0,0,0,0,0], mu = 0, alpha = 0,
                feasibility_param = 100, elite_param = 250, elite_size = 20, iterMax = 15000, p = PENALIZATION,
                pa = PERTURBATION_A, pb = PERTURBATION_B, lsp = LOCAL_SEARCH_PARAM, initial_solution = None,
                elite_revision_param = 1500, vis  = False, verbose = False, time_limit = 60, limit_type = "t"):
    
    global PENALIZATION, Q, earliest, latest, PERTURBATION_A, PERTURBATION_B, LOCAL_SEARCH_PARAM, D
    PENALIZATION = p
    PERTURBATION_A = pa
    PERTURBATION_B = pb
    LOCAL_SEARCH_PARAM = lsp
    
    if semilla is not None:
        np.random.seed(semilla)
        seed(semilla)

    start = perf_counter()
    D = ins.cost
    Q = ins.capacity
    
    earliest = ins.earliest
    latest = ins.latest

    start = perf_counter()
    if initial_solution is None:
        s, cost_best = LPDH_solution(ins,b = np.array(b), alpha = alpha ,mu = mu,  vis = False, initial = True)
    else:
        s, cost_best = initial_solution(ins)

    s, cost = copy(s), cost_best
    candidate_cost = cost_best
    cost_best_unfeasible = inf
    feasible = True

    s_best = copy(s)
    s_best_unfeasible = None

    best_it = 0
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
 
    get_counter = lambda : it if limit_type != 't' else perf_counter() - start
    limit = iterMax if limit_type != 't' else time_limit

    while get_counter() < limit:
        s = perturbation(s)
        s, candidate_cost, feasible = local_search(s)
        if feasible:
            if cost_best > candidate_cost:
                s_best = copy(s)
                cost_best = candidate_cost
                best_it = it + 1
            # print(elite.keys()[-1])
            # if elite.keys()[-1]  > candidate_cost:
                elite[candidate_cost] = [copy(s), False]
                if len(elite) > elite_size: 
                    elite.popitem()
        else:
            if cost_best_unfeasible > candidate_cost:
                s_best_unfeasible = copy(s)
                cost_best_unfeasible = candidate_cost

        if verbose: print(it, candidate_cost)
        else:
            count = get_counter()
            text = f'{count:^10.2f}/{limit} [{"#"*int(count*50//limit):<50}] cost: {candidate_cost:^10.3f} best: {cost_best:^10.3f} it: {it+1}'
            print(text, end = "\r")
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
                            best_it = it + 1

                    while len(elite) > elite_size:
                        # print("size elite: ", len(elite))
                        elite.popitem()

                    # mejoras += 1
            # print("revisiòn lista")
        it += 1
    
    time = perf_counter() - start
    best_bound = None
    gap = None

    if vis: visualize(ins.xcoords, ins.ycoords, s_best[0])
    if not verbose:
        count = get_counter()
        text = f'{count:^10.2f}/{limit} [{"#"*int(count*50//limit):<50}] cost: {candidate_cost:^10.3f} best: {cost_best:^10.3f} best_it: {best_it}'
        print(text)
        print(it)

    return cost_best, time, best_bound, gap

def main():
    name, capacity, node_data = read_instance("instances/c207.txt")
    ins = instance(name, capacity, node_data, 100)

    global xcoords, ycoords, D, latest, earliest, Q
    xcoords, ycoords = ins.xcoords, ins.ycoords
    D = ins.cost
    
    earliest, latest = ins.earliest, ins.latest
    ins.capacity = 10
    Q = ins.capacity

    # generate_solution = lambda x: gurobi_solution(x, vis = False, time_limit= tim, verbose = True, initial=True)
    
    initial_solution = prim(ins, vis = False, initial = True)
    (P, gate, load, arrival), objective_value= gurobi_solution(ins, vis = False, time_limit= 30, verbose = False, initial=True, start =initial_solution)
    # (P, gate, load, arrival), objective_value = prim(ins, vis = False, initial = True)
    initial_solution = lambda x: ((P.copy(), gate.copy(), load.copy(), arrival.copy()), objective_value)

    obj, time, best_bound, gap = ILS_solution(
            ins, semilla = 0, acceptance = 0.05,
            feasibility_param = 1000, elite_param = 2500, elite_size = 20, p = 0.5,
            pa = 0.4, pb = 0.6, lsp = 0.8, initial_solution = None, #   initial_solution,
            elite_revision_param = 1500 , vis  = True, verbose = False, time_limit = 30 )
    pass


if __name__ == "__main__":
    
    main()

# hacer un archivo que solo funcione con gurobi y otro solo con cplex
# darle una vuelta al relajamiento
# ejecutar todo de nuevo: cplex, gurobi, [7], [8]

# la mejor solucion, la mejor por reseteo,
# hacer experimento con la elite, c/!00, 



# Experimento gurobi corriendo 20 y 40 segundos
# experimento en que demoran ambos 1 minuto

# Experimento 11 pero con el prim aleatorio
# Experimento: 11 pero cambiandole los paeametros
# Experimento: Agregar a la elite respecto de la peor de la elite, sin reseteo, resoectoal anterior
# luego lo msmo pero con la mediana
# ver que pasa si se corre, luego encontramos una mejor solución y luego se da

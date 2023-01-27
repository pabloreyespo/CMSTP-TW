import gurobipy as gp
from gurobipy import GRB

import pandas as pd
from metaheuristic_gurobi import ILS_solution
from utilities import read_instance, instance
import os
from math import inf

import numpy as np
from random import choice, seed, random, sample
from sortedcollections import SortedDict
from time import perf_counter
from disjoint_set import DisjointSet
from heuristics import LPDH_solution

from utilities import read_instance,  visualize, instance

PENALIZATION = 0.5
PERTURBATION_A = 1
PERTURBATION_B = 0
LOCAL_SEARCH_PARAM = 1

env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

def gurobi_solution(ins, vis = False, time_limit = 1800, verbose = False, initial = False):

    nnodes = ins.n

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
    if not verbose:
        mdl = gp.Model(env = env)
    else:
        mdl = gp.Model()
    x = mdl.addVars(edges, vtype = GRB.BINARY, name = "x") #
    y = mdl.addVars(edges, vtype = GRB.CONTINUOUS, name = "y", lb = 0)
    d = mdl.addVars(nodes, vtype = GRB.CONTINUOUS, name = "d", lb = 0)

    mdl.setObjective(x.prod(cost))

    R1 = mdl.addConstrs((gp.quicksum(x[(i,j)] for i in nodes if i!=j) == 1 for j in nodesv),name = "R1")
    # restrictions
    R2 = mdl.addConstrs((gp.quicksum(y[(i,j)] for i in nodes if i!=j) - gp.quicksum(y[(j,i)] for i in nodesv if i!=j) == demands[j] for j in nodesv), name = "R2")

    R3 = mdl.addConstrs((x[(i,j)] <= y[(i,j)] for i,j in edges),name = "R3")

    # R4 = mdl.addConstrs((y[(i,j)] <= (Q - demands[i]) * x[(i,j)] for i,j in edges), name = "R4")
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
            solution_edges = SortedDict()
            for i,j in edges:
                if x[i,j].X > 0.9:
                    solution_edges[j] = i
            visualize(ins.xcoords, ins.ycoords, solution_edges)

        return objective_value, time, best_bound, gap

    else: 
        parent = SortedDict()
        departure = SortedDict()
        gate= SortedDict()
        arrival = SortedDict()
        
        
        parent[0] = -1
        departure[0] = 0
        gate[0] = 0
        arrival[0] = 0
        for i,j in edges:
            if x[(i,j)].X > 0.9:
                parent[j] = i
                departure[j] = d[j].X

        
        load = { j : 0 for j in parent.keys()}
        
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
        return (parent, gate, load, arrival), objective_value

def ILS_solution_timelimit(ins, semilla = None, acceptance = 0.05, b = [1,0,0,0,0,0], mu = 0, alpha = 0,
                feasibility_param = 100, elite_param = 250, elite_size = 20, iterMax = 15000, p = PENALIZATION,
                pa = PERTURBATION_A, pb = PERTURBATION_B, lsp = LOCAL_SEARCH_PARAM, initial_solution = None,
                elite_revision_param = 1500, vis  = False, verbose = False, time_limit = 60):
    
    if semilla is not None:
        np.random.seed(semilla)
        seed(semilla)

    start = perf_counter()

    global PENALIZATION, Q, earliest, latest, PERTURBATION_A, PERTURBATION_B, LOCAL_SEARCH_PARAM, D
    PENALIZATION = p
    PERTURBATION_A = pa
    PERTURBATION_B = pb
    LOCAL_SEARCH_PARAM = lsp

    D = ins.cost
    Q = ins.capacity
    
    earliest = ins.earliest
    latest = ins.latest

    start = perf_counter()
    if initial_solution is None:
        s, cost_best = LPDH_solution(ins,b = np.array(b), alpha = alpha ,mu = mu,  vis = False, initial = True)
    else:
        s, cost_best = initial_solution(ins)
    s_inicial, cost_inicial = (s[0].copy(), s[1].copy(), s[2].copy(), s[3].copy()), cost_best
    candidate_cost = cost_best
    cost_best_unfeasible = inf
    feasible = True

    s_best = (s[0].copy(), s[1].copy(), s[2].copy(), s[3].copy())
    s_best_unfeasible = None

    best_it = 0
    feasible_count = 1
    unfeasible_count = 0
    mejoras = 0

    costs_list = [cost_best]
    bestCosts_list = [cost_best]
    solutions_list = [s_best]
    feasibility_list = [feasible]

    elite = SortedDict()
    elite[cost_best] = [(s[0].copy(), s[1].copy(), s[2].copy(), s[3].copy()),False]
    it = 0
    while perf_counter() -  start < time_limit:
        s = perturbation(s)
        s, candidate_cost, feasible = local_search(s)
        if feasible:
            if cost_best > candidate_cost:
                s_best = (s[0].copy(), s[1].copy(), s[2].copy(), s[3].copy())
                
                cost_best = candidate_cost
                best_it = it + 1
            # print(elite.keys()[-1])
            # if elite.keys()[-1]  > candidate_cost:
                elite[candidate_cost] = [ (s[0].copy(), s[1].copy(), s[2].copy(), s[3].copy()), False]
                if len(elite) > elite_size: 
                    elite.popitem()
        else:
            if cost_best_unfeasible > candidate_cost:
                s_best_unfeasible = (s[0].copy(), s[1].copy(), s[2].copy(), s[3].copy())
                cost_best_unfeasible = candidate_cost

        if verbose: print(it, candidate_cost)
        else:
            text = f'{it+1:^6}/{iterMax} [{"#"*((it+1)*50//iterMax):<50}] cost: {candidate_cost:^8.3f} best: {cost_best:8^.3f}'
            print(text, end = "\r")
            

        if feasible: feasible_count += 1
        else: unfeasible_count += 1

        costs_list.append(candidate_cost)
        bestCosts_list.append(cost_best)
        solutions_list.append(s)
        feasibility_list.append(feasible)

        if abs(cost_best - candidate_cost) / cost_best > acceptance or not feasible:
            s = (s_best[0].copy(), s_best[1].copy(), s_best[2].copy(), s_best[3].copy())
        
        if (it + 1) % feasibility_param == 0:
            try: s = (s_best_unfeasible[0].copy(), s_best_unfeasible[1].copy(), s_best_unfeasible[2].copy(), s_best_unfeasible[3].copy())
            except: s = s

        if (it + 1) % elite_param == 0:
            x = choice(elite.values())
            x = x[0]
            s = (x[0].copy(), x[1].copy(), x[2].copy(), x[3].copy())

        if (it + 1) % elite_revision_param == 0:
            for cost in elite:
                ss, rev = elite[cost]
                if not rev:
                    ss, cost_after, feasible = optimal_branch((ss[0].copy(), ss[1].copy(), ss[2].copy(), ss[3].copy()))
                    print(cost, "->", cost_after)
                    elite[cost][1] = True
                    if feasible and cost_after < cost:
                        elite[cost_after] = [(ss[0].copy(), ss[1].copy(), ss[2].copy(), ss[3].copy()), True]

                        if cost_after < cost_best:
                            s_best = (ss[0].copy(), ss[1].copy(), ss[2].copy(), ss[3].copy())
                            cost_best = cost_after
                            best_it = it + 1

                    while len(elite) > elite_size:
                        # print("size elite: ", len(elite))
                        elite.popitem()

                    mejoras += 1
            # print("revisiòn lista")
        it += 1
    
    time = perf_counter() - start
    best_bound = None
    gap = None

    if vis: visualize(ins.xcoords, ins.ycoords, s_best[0])
    if not verbose:
        text = f'{it+1:^6}/{iterMax} [{"#"*((it+1)*50//iterMax):<50}] best: {cost_best:8^.3f} time: {time:.2f} best iteration: {best_it}'
        print(text)

    return cost_best, time, best_bound, gap

def fitness(s):
    F = s[0]
    arrival = s[3]
    cost = 0 
    feasible = True
    for j in F:
        if j != 0:
            i = F[j]
            cost += distance(i,j)
            if arrival[j] > latest[j]:
                feasible = False
                cost += (arrival[j] - latest[j]) * PENALIZATION
    return cost, feasible

def optimal_branch(s):

    P = s[0]
    gate = s[1]
    load = s[2]
    arrival = s[3]

    for i in set(gate.values()):
        if i != 0:
            lo = load[i]
            if lo <= 20 and lo >= 2:
                branch = [j for j in range(1, len(P)) if gate[j] == i]
                if lo < 5 and lo >= 2:
                    bb = branch_bound(branch)
                    aux = bb.best_solution
                else:
                    aux = branch_gurobi(branch)

                for j in branch:
                    P[j] = aux[0][j]
                    gate[j] = aux[1][j]
                    load[j] = aux[2][j]
                    arrival[j] = aux[3][j]

    cost, feasible = fitness((P, gate, load, arrival))
    return (P, gate, load, arrival), cost, feasible     

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
                self.best_solution = (s[0].copy(), s[1].copy(), s[2].copy(), s[3].copy())

        else:
            for j in nodes_left:
                for i in s[0].keys():
                    if cost + distance(i, j) < self.best_cost:
                        P, gate, load, arrival = s[0].copy(), s[1].copy(), s[2].copy(), s[3].copy()
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

def branch_gurobi(branch):

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
    
    mdl.Params.TimeLimit = 1
    mdl.Params.Threads = 1

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
    
    j = choice(P.keys())
    while j == 0:
        j = choice(P.keys())

    
    parents = [j]
    lo = 1
    arrival[j] = max(distance(0,j), earliest[j])
    
    old_gate = gate[j]
    gate[j] = j
    P[j] = 0
    arrival[j] = max(distance(0,j), earliest[j])
    
    for i in sorted(P.keys(), key = lambda x: arrival[x]):
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

    j = choice(P.keys()) # chooses the branch
    while j == 0:
        j = choice(P.keys())

    parents = [j]
    nodes = [j]

    # discover which nodes are part of the branch
    for i in sorted(P.keys(), key = lambda x: arrival[x]):
        if P[i] in parents:
            parents.append(i) 
            nodes.append(i)


    lo = len(nodes)
    load[gate[j]] -= lo

    for i in sorted(P.keys(), key = lambda x: random()):
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
    return best_father(s)

def distance(i,j):
    return D[(i,j)]




def test(cap, nnodes ,pa, pb, lsp, e_param, f_param, elite_search_param , time_limit, gurobi_time, nombre):
    
    instances = os.listdir("instances")
    results = list()
    for inst in instances:
        print(f"{inst}-Q{cap}")
        name, capacity, node_data = read_instance(f"instances/{inst}")
        ins = instance(name, capacity, node_data, nnodes)
        ins.capacity = cap

        best_obj = inf
        time_sum = 0
        solution_sum = 0

        for i in range(10):
            initial_solution = lambda x: gurobi_solution(x, vis = False, time_limit= gurobi_time, verbose = True, initial=True)
            obj, time, best_bound, gap = ILS_solution_timelimit(ins, semilla = i, pa = pa , pb = pb, lsp = lsp, 
                    feasibility_param= f_param,elite_param=e_param, elite_revision_param = elite_search_param,time_limit=time_limit, initial_solution=initial_solution, vis = True) 
            if obj < best_obj:
                best_obj = obj
            time_sum += time
            solution_sum += obj
        dic = {"name": f"{name}","min": best_obj, "avg": solution_sum/10,  "t_avg": time_sum/10 }
        results.append(dic)

    df = pd.DataFrame(results)
    df.to_excel(f"{nombre}.xlsx", index= False)

def main():

    name, capacity, node_data = read_instance(f"instances/c205.txt")
    ins = instance(name, capacity, node_data, 100)
    ins.capacity = 5
    initial_solution = lambda x: gurobi_solution(x, vis = False, time_limit= 20, verbose = True, initial=True)
    obj, time, best_bound, gap = ILS_solution_timelimit(ins,semilla = 0, pa = 0.4 , pb = 0.6, lsp = 0.8, 
                    feasibility_param= 1000,elite_param=2500, elite_revision_param = 1500, time_limit=60, initial_solution = initial_solution) 

if __name__ == "__main__":
    # caps = [10000,20,15,10,5]
    # nnodes = 100
    # for cap in caps:
    #     test(cap, nnodes, 0.4, 0.6, 0.8, 1000, 2500, 1500,time_limit=60, gurobi_time=10, nombre = f"GUROBI INICIAL 10s + ILS Q-{cap}")
    #     test(cap, nnodes, 0.4, 0.6, 0.8, 1000, 2500, 1500,time_limit=60, gurobi_time=20, nombre = f"GUROBI INICIAL 20s + ILS Q-{cap}")
    #     test(cap, nnodes, 0.4, 0.6, 0.8, 1000, 2500, 1500,time_limit=60, gurobi_time=30, nombre = f"GUROBI INICIAL 30s + ILS Q-{cap}")

    main()
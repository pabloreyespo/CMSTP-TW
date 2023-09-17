from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import matplotlib.pyplot as plt
from operators_cython import perturbation, best_father, fitness

import numpy as np
import pandas as pd
import math
import os
from math import inf, sqrt
from random import choice, seed, sample, random, randint, uniform
from sortedcollections import SortedDict
from time import perf_counter
from disjoint_set import DisjointSet

env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

PERTURBATION1 = 0.1
PERTURBATION2 = 0.1 + 0.7 * 0.392
PERTURBATION3 = PERTURBATION2 + 0.7 * (1 - 0.392)    
LOCAL_SEARCH_PARAM1 = 0.1 # best_father
LOCAL_SEARCH_PARAM2 = 0.1 + 0.9 * 0.402 # best_father

class instance():
    def __init__(self, name, capacity, node_data, num, penalization, reset_demand = True):
        self.name = name
        self.n = n = num + 1
        self.capacity = int(capacity)
        self.index, self.xcoords, self.ycoords, self.demands, self.earliest, self.latest = extract_data(node_data[:n])
        xc = self.xcoords - self.xcoords[0]
        yc = self.ycoords - self.ycoords[0]
        self.thetacoords = np.arctan2(yc, xc) 
        self.nodes = np.array(range(n), dtype = int)
        self.edges = [(i,j) for i in self.nodes for j in self.nodes[1:] if i != j]
        self.penalization = penalization

        #demands = 1 for all nodes 
        if reset_demand: 
            self.demands = {i:1 for i in self.nodes}

        # cost = time = distance for simplicity
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                if i != j:
                    D[i,j] = D[j,i]= self.dist(i,j)

        self.distance = D
        self.cost = self.distance
        self.maxcost = self.cost.mean()

    def dist(self,i,j):
        x = self.xcoords[i] - self.xcoords[j]
        y = self.ycoords[i] - self.ycoords[j]
        return sqrt(x**2 + y**2)

class tree:
    def __init__(self, ins):
        self.n = ins.n
        self.parent = np.ones(self.n, dtype = int) * -1
        self.gate = np.zeros(self.n, dtype = int)
        self.load = np.zeros(self.n, dtype = int)
        self.arrival = np.zeros(self.n)
        self.capacity = ins.capacity
        self.distance = ins.distance
        self.xcoords = ins.xcoords
        self.ycoords = ins.ycoords
        self.demands = ins.demands
        self.earliest = ins.earliest
        self.latest = ins.latest
        self.penalization = ins.penalization

    def connect(self, k, j):
        """k: parent, j: children"""
        if self.gate[k] == 0:
            self.gate[j] = gate = j
        else:
            self.gate[j] = gate = self.gate[k]
        self.load[gate] += 1
        self.parent[j] = k
        self.arrival[j] = self.arrival[k] + self.distance[k,j]
        if not self.arrival[j] >= self.earliest[j]:
            self.arrival[j] = self.earliest[j]
    
    def __repr__(self):
        parent = f'parent: [{", ".join([str(elem) for elem in self.parent])}]\n'
        gate = f'gate: [{", ".join([str(elem) for elem in self.gate])}]\n'
        load = f'load: [{", ".join([str(elem) for elem in self.load])}]\n'
        arrival = f'arrival: [{", ".join([str(elem) for elem in self.arrival])}]\n'
        return parent + gate + load + arrival

class MLM:
    def __init__(self, instance, phi1, phi2, theta1, theta2,theta3, elite_size, acceptance ,unfeasible_introduction ,elite_introduction ,elite_revision ,verbose = False):
        self.instance = instance
        self.capacity = instance.capacity
        self.cost = instance.distance
        self.n = instance.n
        self.bigM = instance.latest.max() + self.cost.max()

        self.branch_time = 5
        self.initial_trigger = 40
        self.phi1 = phi1
        self.phi2 = phi2
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.elite_size = elite_size
        self.acceptance = acceptance
        self.unfeasible_introduction = unfeasible_introduction
        self.elite_introduction = elite_introduction
        self.elite_revision = elite_revision
        self.verbose = verbose

    def prim(self):
        nodes= self.instance.nodes
        s = tree(self.instance)
        itree = set() # muestra que es lo ultimo que se ha añadido
        nodes_left = set(nodes)

        d = inf
        for j in nodes[1:]:
            di = self.cost[0,j]
            if di < d:
                d = di
                v = j

        itree.add(0) #orden en que son nombrados
        itree.add(v)
        nodes_left.remove(0)
        nodes_left.remove(v)
        
        s.connect(0,v)
        cost = self.cost[0,v]

        while len(nodes_left) > 0:
            min_tree = inf
            for j in nodes_left:
                min_node = inf
                for ki in itree:# k: parent, j: offspring
                    # calcula si alcanza a llegar desde alguno de los nodos que ya estan colocados
                    dkj = self.cost[ki,j]
                    # criterion = dkj
                    tj = s.arrival[ki] + dkj
                    Qj = s.load[s.gate[ki]]

                    if tj <= self.instance.latest[j] and Qj < self.capacity: # isFeasible() # reescribir

                        if tj < self.instance.earliest[j]:
                            tj = self.instance.earliest[j]

                        crit_node = dkj
                        if crit_node < min_node:
                            min_node = crit_node
                            k = ki
                    
                ### best of the node
                crit_tree = crit_node

                if crit_tree < min_tree:
                    kk, jj = k, j
                    min_tree = crit_tree

            itree.add(jj)
            nodes_left.remove(jj)
            s.connect(kk,jj)
            cost += self.cost[kk,jj]
        return s, cost   

    def __gurobi_model(self, nodes, nodesv, edges, time_limit, start, rando):  
            edges, cost = gp.multidict({(i,j): self.cost[i,j] for (i,j) in edges})
            nodes, earliest, latest, demands = gp.multidict({i: (self.instance.earliest[i], self.instance.latest[i], self.instance.demands[i]) for i in nodes }) 
            M = self.bigM

            mdl = gp.Model(self.instance.name, env = env) if not self.verbose else gp.Model(self.instance.name, env = env)

            if rando:
                x = gp.tupledict()
                for i, j in edges:
                    ii = start.parent[j]
                    if ii == i:
                        x[i,j] = mdl.addVar(vtype = GRB.BINARY, lb=0, ub=1, name = "x[%d,%d]" % (i,j))
                        x[i,j].VarHintVal = 1
                        # x[i,j].VarHintPri = ins.maxcost * (1 + ins.cost[i,j])
                        x[i,j].VarHintPri = int(self.instance.maxcost * (1 + 1/self.instance.cost[i,j]))
                        # ajsutar una probabilidad según la longitud del arco
                    else:
                        x[i,j] = mdl.addVar(vtype = GRB.BINARY, lb=0, ub=1, name = "x[%d,%d]" % (i,j))
                        x[i,j].VarHintVal = 0
                        if self.instance.cost[i,j] > self.instance.maxcost:
                            x[i,j].VarHintVal = 0
                            x[i,j].VarHintPri = int(self.instance.cost[i,j])
            else:
                x = mdl.addVars(edges, vtype = GRB.BINARY, name = "x") #
            y = mdl.addVars(edges, vtype = GRB.CONTINUOUS, name = "y", lb = 0)
            d = mdl.addVars(nodes, vtype = GRB.CONTINUOUS, name = "d", lb = 0)

            if start is not None:
                for j in nodesv:
                    i = start.parent[j]
                    x[(i,j)].Start = 1

            mdl.setObjective(x.prod(cost))
            mdl.addConstrs((gp.quicksum(x[(i,j)] for i in nodes if i!=j) == 1 for j in nodesv),name = "R1")
            mdl.addConstrs((gp.quicksum(y[(i,j)] for i in nodes if i!=j) - gp.quicksum(y[(j,i)] for i in nodesv if i!=j) == demands[j] for j in nodesv), name = "R2") 
            mdl.addConstrs((x[(i,j)] <= y[(i,j)] for i,j in edges),name = "R3") 
            mdl.addConstrs((y[(i,j)] <= self.capacity * x[(i,j)] for i,j in edges), name = "R4") 
            mdl.addConstrs((d[i] + cost[(i,j)] - d[j] <= M * (1 - x[(i,j)]) for i,j in edges), name = "R5") 
            mdl.addConstrs((d[i] >= earliest[i] for i in nodes), name = "R6") 
            mdl.addConstrs((d[i] <= latest[i] for i in nodes), name = "R7")
            mdl.Params.TimeLimit = time_limit
            mdl.Params.Threads = 1
            
            if rando:
                mdl.Params.SolutionLimit = 2

            solution = mdl.optimize()
            return mdl, x, d
    
    def gurobi_solution(self, time_limit, initial = False, vis = False ,start = None, rando = False):
        nodes = self.instance.nodes
        nodesv = nodes[1:]
        edges = self.instance.edges
        mdl, x, d = self.__gurobi_model(nodes, nodesv, edges, time_limit, start, rando)
        print("gap:", mdl.MIPGap)
        objective_value = mdl.getObjective().getValue()
        s = tree(self.instance)

        if not initial:

            time = mdl.Runtime
            best_bound = mdl.ObjBound
            gap = mdl.MIPGap

            if vis:
                for i,j in edges:
                    if x[i,j].X > 0.9:
                        s.parent[j] = i
                visualize(s)

            return objective_value, time, best_bound, gap

        else: 
            departure = np.zeros(self.n)
            for i,j in edges:
                if x[(i,j)].X > 0.9:
                    s.parent[j] = i
                    departure[j] = d[j].X

            for j in sorted(nodes, key = lambda x: departure[x]):
                if j != 0:
                    k = s.parent[j]
                    s.connect(k,j)

            if vis:
                visualize(s)
            optimal = True if mdl.MIPGap < 0.0001 else False

            return s, objective_value, optimal                 

    def branch_bound(self, branch, s):
        self.branch_nodes = [0] + branch
        self.branch_size = len(self.branch_nodes)
        self.branch_solution = None
        self.branch_cost = inf

        self.t = {num:j for j,num in enumerate(self.branch_nodes)}
        self.decode = {j:num for j,num in enumerate(self.branch_nodes)}
        self.decode[-1] = -1

        parent = np.zeros(self.branch_size, dtype = int)
        gate = np.zeros(self.branch_size, dtype = int)
        load = np.zeros(self.branch_size, dtype = int)
        arrival = np.zeros(self.branch_size)
        parent[0] = -1

        nodes_left = set(self.branch_nodes[1:])
        itree = set()
        itree.add(0)
        
        sol = (parent, gate, load, arrival)
        self.__explore_bbound(sol, 0, nodes_left, itree)

        for ki in range(self.branch_size):
            i = self.decode[ki]
            s.parent[i] = self.decode[self.branch_solution[0][ki]]
            s.gate[i] = self.decode[self.branch_solution[1][ki]]
            s.load[i] = self.branch_solution[2][ki]
            s.arrival[i] = self.branch_solution[3][ki]
            
        return s

    def __explore_bbound(self, sol, cost, nodes_left, itree):
        if len(nodes_left) == 0:
            if cost < self.branch_cost:
                self.branch_cost = cost
                self.branch_solution = deepcopy(sol)
        else:
            for j in nodes_left:
                kj = self.t[j]
                for i in itree:
                    ki = self.t[i]
                    
                    if cost + self.cost[i,j] < self.branch_cost:
                        
                        parent, gate, load, arrival = deepcopy(sol)
                        parent[kj] = ki
                        gate[kj] = gate[ki] if i != 0 else kj
    
                        if arrival[ki] + self.cost[i,j] <= self.instance.latest[j] and load[gate[kj]] < self.instance.capacity: # isFeasible() # reescribir
                            
                            load[gate[kj]] += 1
                            arrival[kj] = arrival[ki] + self.cost[i,j]
                            if arrival[kj] < self.instance.earliest[j]:
                                arrival[kj] = self.instance.earliest[j]
                            
                            self.__explore_bbound((parent, gate, load, arrival), cost + self.cost[i, j], nodes_left - {j}, itree | {j})
                            load[gate[kj]] -= 1

    def branch_gurobi(self, branch, s, initial = False):
        nodes = [0] + branch
        nodesv = branch
        edges =  [(i,j) for i in nodes for j in nodesv if i != j]
        if initial:
            mdl, x, d = self.__gurobi_model(nodes, nodesv, edges, time_limit = self.branch_time, start = s, rando  = False)
        else:
            mdl, x, d = self.__gurobi_model(nodes, nodesv, edges, time_limit = self.branch_time, start = None, rando  = False)
        
        departure = SortedDict()
        for i,j in edges:
            if x[(i,j)].X > 0.9:
                s.parent[j] = i
                departure[j] = d[j].X

        for j in nodesv:
            s.load[j] = 0
            s.arrival[j] = 0

        for j in sorted(nodesv, key = lambda x: departure[x]):
            i = s.parent[j]
            s.gate[j] = s.gate[i] if i != 0 else j
            s.load[s.gate[j]] += 1
            s.arrival[j] = s.arrival[i] + self.cost[i,j]
            if s.arrival[j] < self.instance.earliest[j]:
                s.arrival[j] = self.instance.earliest[j]

        return s

    def merge_branches(self, s):
        ds = get_disjoint(s.parent)
        ds = [list(i) for i in ds.itersets()]
        nds = len(ds)
        
        theta_sets = np.zeros(nds)
        for i, st in enumerate(ds):
            m = len(st)
            for j in st:
                theta_sets[i] += self.instance.thetacoords[j]
            theta_sets = theta_sets / len(st)

        theta = theta_sets + (random() * (2 * np.pi))
        for i,j in enumerate(theta):
            if j > 2 * np.pi:
                theta[i] -= 2 * np.pi
            
        branches = list(range(nds))
        branches = sorted(branches, key = lambda x: theta[x])
        
        for i in range(nds//2):
            if perf_counter() - self.metaheuristic_start <= self.metaheuristic_limit:
                s1, s2 = i*2, i*2+1
                branch = ds[s1] + ds[s2]

                lo = len(branch)
                if lo < 5 and lo >= 2:
                    s = self.branch_bound(branch, s)
                elif lo <= self.initial_trigger:
                    s = self.branch_gurobi(branch, s)
                else:
                    s = self.branch_gurobi(branch, s, initial = True)
            else:
                break
        cost, feasible = fitness(s)
        return s, cost, feasible 
    
    def local_search(self, s):
        x = random()
        if x < self.phi1:
            # print("LA")
            return self.merge_branches(s)
        elif x < self.phi2:
            # print("LB")
            return best_father(s,1)
        else:
            # print("LC")
            return best_father(s,5)  
        
    def optimal_branch(self,s):
        for i in set(s.gate):
            if perf_counter() - self.metaheuristic_start <= self.metaheuristic_limit:
                if i != 0:
                    lo = s.load[i]
                    if lo <= 20 and lo >= 2:
                        branch = [j for j in range(1, len(s.parent)) if s.gate[j] == i]
                        if lo < 5 and lo >= 2:
                            s = self.branch_bound(branch,s)
                        else:
                            s = self.branch_gurobi(branch,s)
            else:
                break

        cost, feasible = fitness(s)
        return s, cost, feasible 

    def ILS(self, semilla = None, initial_solution = None, vis  = False, time_limit = 20):
        
        if semilla is not None:
            np.random.seed(semilla)
            seed(semilla)

        self.metaheuristic_start = perf_counter()
        self.metaheuristic_limit = time_limit

        s, cost_best =  initial_solution() if initial_solution is not  None else self.prim()
        s, candidate_cost = deepcopy(s), cost_best
        cost_best_unfeasible = inf
        feasible = True

        s_best = deepcopy(s)
        s_best_unfeasible = None

        elite = SortedDict()
        elite[cost_best] = [deepcopy(s),False]

        it = best_it = 1    
        get_counter = lambda : perf_counter() - self.metaheuristic_start

        while get_counter() < time_limit:
            s = perturbation(s, self.theta1, self.theta2, self.theta3)
            s, candidate_cost, feasible = self.local_search(s)
            if feasible:
                if cost_best > candidate_cost:
                    s_best, cost_best = deepcopy(s), candidate_cost
                    best_it = it 
                    elite[candidate_cost] = [deepcopy(s), False]
                    if len(elite) > self.elite_size: 
                        elite.popitem()
            else:
                if cost_best_unfeasible > candidate_cost:
                    s_best_unfeasible, cost_best_unfeasible = deepcopy(s), candidate_cost

            if self.verbose: 
                count = get_counter()
                text = f'{count:^10.2f}/{time_limit} [{"#"*int(count*50//time_limit):<50}] cost: {candidate_cost:^10.3f} best: {cost_best:^10.3f} it: {it}'
                print(text, end = "\r")

            if candidate_cost > cost_best * (1 + self.acceptance) or not feasible:
                s = deepcopy(s_best)
            
            if it % self.unfeasible_introduction == 0:
                if s_best_unfeasible is not None:
                    s = deepcopy(s_best_unfeasible)

            if it % self.elite_introduction == 0:
                x = choice(elite.values())[0]
                s = deepcopy(x)

            if it % self.elite_revision == 0:
                for cost in elite:
                    if perf_counter() - self.metaheuristic_start <= self.metaheuristic_limit:
                        ss, rev = elite[cost]
                        if not rev:
                            ss, cost_after, feasible = self.optimal_branch(deepcopy(ss))
                            # print(cost, "->", cost_after)
                            elite[cost][1] = True
                            if feasible and cost_after < cost:
                                elite[cost_after] = [deepcopy(ss), False]

                                if cost_after < cost_best:
                                    s_best = deepcopy(ss)
                                    cost_best = cost_after
                                    best_it = it 
                    else:
                        break

                while len(elite) > self.elite_size:
                    elite.popitem()
            it += 1
        
        time = perf_counter() - self.metaheuristic_start
        count = get_counter()
       
        text = f'{count:^10.2f}/{time_limit} [{"#"*int(count*50//time_limit):<50}] cost: {candidate_cost:^10.3f} best: {cost_best:^10.3f} best_it: {best_it}/{it}'
        print(text)
        return s_best, cost_best
    
    def SA(self, semilla = None, initial_solution = None, time_limit = 20):
        temperature = 100
        alfa = 0.99
        neighboors = 5
        
        if semilla is not None:
            np.random.seed(semilla)
            seed(semilla)

        start = perf_counter()

        s, cost_best =  initial_solution() if initial_solution is not  None else self.prim()
        current_cost = cost_best
        s, candidate_cost = deepcopy(s), cost_best
        cost_best_unfeasible = inf
        feasible = True

        s_best = deepcopy(s)
        s_best_unfeasible = None

        elite = SortedDict()
        elite[cost_best] = [deepcopy(s),False]

        it = best_it = 1    
        get_counter = lambda : perf_counter() - start

        while get_counter() < time_limit:

            for _ in range(neighboors):
                candidate_s = perturbation(deepcopy(s), self.theta1, self.theta2, self.theta3)
                candidate_s, candidate_cost, feasible = self.local_search(candidate_s)
                if candidate_cost < current_cost:
                    if feasible:
                        current_cost, s, current_feasible = candidate_cost, deepcopy(candidate_s), feasible
                        if cost_best  > current_cost:
                            s_best, cost_best = deepcopy(s), current_cost
                            best_it = it 
                            elite[current_cost] = [deepcopy(s), False]
                            if len(elite) > self.elite_size: 
                                elite.popitem()
                elif candidate_cost * 2 < current_cost:
                    if not feasible:
                        if cost_best_unfeasible > current_cost:
                            s_best_unfeasible, cost_best_unfeasible = deepcopy(s), current_cost

                elif uniform(0, 1) < math.exp(-abs(candidate_cost - current_cost) / temperature):
                    if uniform(0,1) < 0.1:
                        x = choice(elite.keys())
                        current_cost, s, current_feasible = x, deepcopy(elite[x][0]), True
                    else:
                        current_cost, s, current_feasible = candidate_cost, deepcopy(candidate_s), feasible

            if self.verbose: 
                count = get_counter()
                text = f'{count:^10.2f}/{time_limit} [{"#"*int(count*50//time_limit):<50}] cost: {current_cost:^10.3f} best: {cost_best:^10.3f} it: {it}'
                print(text, end = "\r")

            if it % self.elite_revision == 0:
                for cost in elite:
                    if perf_counter() - self.metaheuristic_start <= self.metaheuristic_limit:
                        ss, rev = elite[cost]
                        if not rev:
                            ss, cost_after, feasible = self.optimal_branch(deepcopy(ss))
                            # print(cost, "->", cost_after)
                            elite[cost][1] = True
                            if feasible and cost_after < cost:
                                elite[cost_after] = [deepcopy(ss), True]

                                if cost_after < cost_best:
                                    s_best = deepcopy(ss)
                                    cost_best = cost_after
                                    best_it = it 
                    else:
                        break

                while len(elite) > self.elite_size:
                    elite.popitem()

            it += 1
            temperature *= alfa
        
        time = perf_counter() - start
        count = get_counter()
       
        text = f'{count:^10.2f}/{time_limit} [{"#"*int(count*50//time_limit):<50}] cost: {candidate_cost:^10.3f} best: {cost_best:^10.3f} best_it: {best_it}/{it}'
        print(text)
        return s_best, cost_best
    
    def run(self, time_limit = 300, gurobi_time = 40, meta_limit = 20):
        semilla = 42
        s, cost = self.prim()
        print("prim:", cost)
        s, cost, optimal = self.gurobi_solution(vis = False, time_limit= gurobi_time, initial=True, start = s)
        if not optimal:
            print("gurobi:", cost)
            initial_solv = lambda: (deepcopy(s), cost)
            last = "ILS"

            s, cost = self.ILS(semilla = semilla, initial_solution = initial_solv, time_limit = meta_limit) 
            print("ILS:", cost)
            
            time_left = time_limit - gurobi_time - meta_limit
            while time_left > 0:
                s, cost, optimal = gurobi_solution(self.instance, vis = False, time_limit= min(gurobi_time//2, time_left), initial=True, start = s, rando = True)
                if optimal: 
                    break

                time_left -= gurobi_time//2
                print("gurobi:", cost)

                intermediate_solution = lambda: (deepcopy(s), cost)
                if time_left > 0:
                    #if last == "ILS":
                    #s, cost = self.SA(initial_solution = intermediate_solution, time_limit = min(meta_limit//2, time_left))         
                    #    last = "SA"
                    #else:
                    s, cost = self.ILS(initial_solution = intermediate_solution, time_limit = min(meta_limit//2, time_left))         
                    #    last = "ILS"

                    time_left -= meta_limit//2
                    print(f"{last}:", cost)

    def run_test(self, tries, time_limit = 300, gurobi_time = 40, meta_limit = 20):

        s, cost = self.prim()
        print("prim:", cost)
        s, cost, optimal = self.gurobi_solution(vis = False, time_limit= gurobi_time, initial=True, start = s)
        print("gurobi:", cost)

        initial_s = deepcopy(s)
        initial_cost = cost
        initial_solv = lambda: (deepcopy(initial_s), initial_cost)
        if not optimal:

            solutions = []
            best_solution = cost
            times = []

            for seed in range(tries):
                print(seed)
                time = perf_counter()
                s, cost = self.ILS(semilla = seed, initial_solution = initial_solv, time_limit = meta_limit) 
                print("ILS:", cost)
                
                time_left = time_limit - gurobi_time - meta_limit
                while time_left > 0:
                    s, cost, optimal = self.gurobi_solution(vis = False, time_limit= min(gurobi_time//2, time_left), initial=True, start = s, rando = True)
                    if optimal:
                        break
                    time_left -= gurobi_time//2
                    print("gurobi:", cost)

                    intermediate_solution = lambda: (deepcopy(s), cost)
                    if time_left > 0:
                        s, cost = self.ILS(initial_solution = intermediate_solution, time_limit = min(meta_limit//2, time_left))         
                        time_left -= meta_limit//2
                        print("ILS:", cost)

                time = perf_counter() - time
                times.append(time)
                solutions.append(cost)
                if best_solution > cost:
                    best_solution = cost
        
            return best_solution, sum(solutions) / tries, gurobi_time + sum(times) / tries
        else:
            return cost, cost, gurobi_time

def gurobi_solution(ins, vis = False, time_limit = 1800, verbose = False, initial = False, start = None, rando = False):
    edges, cost = gp.multidict({(i,j): ins.cost[i,j] for (i,j) in ins.edges})
    nodes, earliest, latest, demands = gp.multidict({i: (ins.earliest[i], ins.latest[i], ins.demands[i]) for i in ins.nodes })
    nodesv = nodes[1:]

    M = max(latest.values()) + max(cost.values())

    # model and variables
    if verbose:
        mdl = gp.Model(ins.name)
    else:
        mdl = gp.Model(ins.name, env = env)
        
    if rando:
        x = gp.tupledict()
        for i, j in edges:
            ii = start.parent[j]
            if ii == i:
                x[i,j] = mdl.addVar(vtype = GRB.BINARY, lb=0, ub=1, name = "x[%d,%d]" % (i,j))
                x[i,j].VarHintVal = 1
                # x[i,j].VarHintPri = ins.maxcost * (1 + ins.cost[i,j])
                x[i,j].VarHintPri = int(ins.maxcost * (1 + 1/ins.cost[i,j]))
                # ajsutar una probabilidad según la longitud del arco
            else:
                x[i,j] = mdl.addVar(vtype = GRB.BINARY, lb=0, ub=1, name = "x[%d,%d]" % (i,j))
                # x[i,j].VarHintVal = 0
                if ins.cost[i,j] > ins.maxcost:
                    x[i,j].VarHintVal = 0
                    x[i,j].VarHintPri = int(ins.cost[i,j])
    else:
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
    R4 = mdl.addConstrs((y[(i,j)] <= ins.capacity * x[(i,j)] for i,j in edges), name = "R4") 
    R5 = mdl.addConstrs((d[i] + cost[(i,j)] - d[j] <= M * (1 - x[(i,j)]) for i,j in edges), name = "R5") 
    R6 = mdl.addConstrs((d[i] >= earliest[i] for i in nodes), name = "R6") 
    R7 = mdl.addConstrs((d[i] <= latest[i] for i in nodes), name = "R7")


    mdl.Params.TimeLimit = time_limit
    mdl.Params.Threads = 1
    
    if rando:
        mdl.Params.SolutionLimit = 2

    solution = mdl.optimize()
    obj = mdl.getObjective()
    objective_value = obj.getValue()
    s = tree(ins)
    print("gap:", mdl.MIPGap)

    if not initial:

        time = mdl.Runtime
        best_bound = mdl.ObjBound
        gap = mdl.MIPGap

        if vis:
            for i,j in edges:
                if x[i,j].X > 0.9:
                    s.parent[j] = i
            visualize(s)

        return objective_value, time, best_bound, gap

    else: 
        departure = np.zeros(ins.n)
        for i,j in edges:
            if x[(i,j)].X > 0.9:
                s.parent[j] = i
                departure[j] = d[j].X

        for j in sorted(nodes, key = lambda x: departure[x]):
            if j != 0:
                k = s.parent[j]
                s.connect(k,j)

        if vis:
            visualize(s)

        optimal = True if mdl.MIPGap < 0.0001 else False

        return s, objective_value, optimal

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
        if s.load[i] > s.capacity:
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

def visualize(s):
    fig, ax = plt.subplots(1,1)

    # root node
    ax.scatter(s.xcoords[0],s.ycoords[0], color ='green',marker = 'o',s = 275, zorder=2)
    # other nodes
    ax.scatter(s.xcoords[1:],s.ycoords[1:], color ='indianred',marker = 'o',s = 275, zorder=2)

    # edges activated
    for j,k in  enumerate(s.parent): 
        if j != 0:
            ax.plot([s.xcoords[k],s.xcoords[j]],[s.ycoords[k],s.ycoords[j]], color = 'black',linestyle = ':',zorder=1)

    # node label
    for i in range(s.n): 
        plt.annotate(str(i) ,xy = (s.xcoords[i],s.ycoords[i]), xytext = (s.xcoords[i]-0.6,s.ycoords[i]-0.6), color = 'black', zorder=4)
    plt.show()

def main():
    penalization = 7.001
    LS1 = 0.013
    LS2 = 0.157
    P1 = 0.121
    P2 = 0.677
    P3 = 0.186
    name, capacity, node_data = read_instance("gehring instances/200/C2_2_8.TXT")
    ins = instance(name, capacity, node_data, 200, penalization = penalization)
    ins.capacity = 20
    solver = MLM(instance = ins, phi1 = LS1, phi2=LS1+ LS2, theta1 = P1,  theta2=P1+P2, theta3=P1+P2+P3 ,
                 elite_size=20,acceptance=0.032,
                 unfeasible_introduction=10000,elite_introduction=20000,elite_revision=6000,verbose=True)
    
    solver.run(time_limit=300, gurobi_time=40,meta_limit=260)

def test(global_time, q, nnodes, ins_folder):

    instances = os.listdir(ins_folder)
    results = list()

    penalization = 7.001
    LS1 = 0.013
    LS2 = 0.157
    P1 = 0.121
    P2 = 0.677
    P3 = 0.186

    for p in instances:
        print(p)

        name, capacity, node_data = read_instance(ins_folder + "/"+  p)
        ins = instance(name, capacity, node_data, nnodes, penalization=penalization)
        ins.capacity = q

        solver = MLM(instance = ins, phi1 = LS1, phi2=LS1+ LS2, theta1 = P1,  theta2=P1+P2, theta3=P1+P2+P3 ,
                 elite_size=20,acceptance=0.032,
                 unfeasible_introduction=10000,elite_introduction=20000,elite_revision=6000,verbose=False)
    
        tries = 10
        best_obj, avg_solution, avg_time =solver.run_test(tries = tries, time_limit=global_time, gurobi_time=40,meta_limit=global_time-40)

        dic = {"name": f"{name}","min": best_obj, "avg": avg_solution,  "t_avg": avg_time}
        results.append(dic)

    df = pd.DataFrame(results)
    df.to_excel(f"{nombre}.xlsx", index= False)

if __name__ == "__main__":
    # main()
    capacities = [10000, 20, 15, 10, 5]
    global_time = 300
    nnodes = 100
    for q in capacities:
        ins_folder = "Instances"
        nombre = f"MLM solo ILS {global_time} Q-{q} n-{nnodes}"
        test(global_time, q, nnodes, ins_folder)

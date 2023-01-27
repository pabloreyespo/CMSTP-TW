import gurobipy as gp
from gurobipy import GRB
import numpy as np

from sortedcollections import SortedDict
from utilities import instance, visualize, read_instance, visualize_relaxed

env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

def gurobi_solution(ins, vis = False, time_limit = 1800, verbose = False, initial = False, start = None):

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
        for j in parent.keys():
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
            solution_edges = SortedDict()
            for i,j in edges:
                if x[i,j].X > 0.9:
                    solution_edges[j] = i
            visualize(ins.xcoords, ins.ycoords, solution_edges)

        return objective_value, time, best_bound, gap

    else: 
        parent = SortedDict()
        departure = SortedDict()
        for i,j in edges:
            if x[(i,j)].X > 0.9:
                parent[j] = i
                departure[j] = d[j].X

        parent[0] = -1
        departure[0] = 0

        gate= SortedDict()
        gate[0] = 0
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
        return (parent, gate, load, arrival), objective_value

def gurobi_initial_array(ins, vis = False, time_limit = 1800, verbose = False):
    (parent, gate, load, arrival), objective_value =gurobi_solution(ins, vis, time_limit, verbose, initial = True)
    n = ins.n
    pred_array = np.ones(n, dtype = int) * -1
    gate_array = np.zeros(n, dtype = int)
    load_array = np.zeros(n, dtype = int)
    arrival_array = np.zeros(n)
    for i in range(ins.n):
        pred_array[i] = parent[i]
        gate_array[i] = gate[i]
        load_array[i] = load[i]
        arrival_array[i] = arrival[i]

    return (pred_array,gate_array,load_array,arrival_array), objective_value
    
def relaxed_gurobi_solution(ins, vis = False, time_limit = 1800, verbose = False):

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
    mdl = gp.Model(ins.name)
    x = mdl.addVars(edges, vtype = GRB.CONTINUOUS, name = "x", lb = 0, ub = 1) #
    y = mdl.addVars(edges, vtype = GRB.CONTINUOUS, name = "y", lb = 0)
    d = mdl.addVars(nodes, vtype = GRB.CONTINUOUS, name = "d", lb = 0)

    mdl.setObjective(x.prod(cost))

    R1 = mdl.addConstrs((gp.quicksum(x[(i,j)] for i in nodes if i!=j) == 1 for j in nodesv),name = "R1")
    # restrictions
    R2 = mdl.addConstrs((gp.quicksum(y[(i,j)] for i in nodes if i!=j) - gp.quicksum(y[(j,i)] for i in nodesv if i!=j) == demands[j] for j in nodesv), name = "R2")
    R3 = mdl.addConstrs((x[(i,j)] <= y[(i,j)] for i,j in edges),name = "R3")
    R4 = mdl.addConstrs((y[(i,j)] <= Q * x[(i,j)] for i,j in edges), name = "R4")
    R5 = mdl.addConstrs((d[i] + cost[(i,j)] - d[j] <= M * (1 - x[(i,j)]) for i,j in edges), name = "R5")
    R6 = mdl.addConstrs((d[i] >= earliest[i] for i in nodes), name = "R6")
    R7 = mdl.addConstrs((d[i] <= latest[i] for i in nodes), name = "R7")

    mdl.Params.TimeLimit = time_limit
    mdl.Params.Threads = 1

    solution = mdl.optimize()

    solution_edges = list()
    intensity = dict()

    for i,j in edges:
        if x[i,j].X > 0:
            solution_edges.append((i,j))
            intensity[(i,j)] = x[i,j].X

    obj = mdl.getObjective()
    objective_value = obj.getValue()

    time = mdl.Runtime
    best_bound = mdl.ObjBound
    gap = None

    if vis:
        visualize_relaxed(ins.xcoords, ins.ycoords, solution_edges, intensity)

    global relaxed_edges
    relaxed_edges = solution_edges[::]

    return objective_value, time, best_bound, gap

def after_relaxation(ins, relaxed_edges, vis = False, time_limit = 1800, verbose = False):

    nnodes = ins.n

    Q = ins.capacity
    earliest = ins.earliest
    latest = ins.latest
    demands = ins.demands
    global D
    D = ins.cost

    for (i,j) in relaxed_edges:
        if (j,i) not in relaxed_edges:
            relaxed_edges.append((j,i))

    edges, cost = gp.multidict({(i,j): D[i,j] for (i,j) in relaxed_edges})

    nodes, earliest, latest, demands = gp.multidict({i: (ins.earliest[i], ins.latest[i], ins.demands[i]) for i in ins.nodes })
    nodesv = nodes[1:]

    M = max(latest.values()) + max(cost.values())

    # model and variables
    if verbose:
        mdl = gp.Model()
    else: 
        mdl = gp.Model(env = env)
    x = mdl.addVars(edges, vtype = GRB.BINARY, name = "x") #
    y = mdl.addVars(edges, vtype = GRB.CONTINUOUS, name = "y", lb = 0)
    d = mdl.addVars(nodes, vtype = GRB.CONTINUOUS, name = "d", lb = 0)

    mdl.setObjective(x.prod(cost))

    R1 = mdl.addConstrs((gp.quicksum(x[(i,j)] for i in nodes if i!=j and (i,j) in edges) == 1 for j in nodesv),name = "R1")
    # restrictions
    R2 = mdl.addConstrs((gp.quicksum(y[(i,j)] for i in nodes if i!=j and (i,j) in edges) - gp.quicksum(y[(j,i)] for i in nodesv if i!=j and (i,j) in edges) == demands[j] for j in nodesv), name = "R2")

    R3 = mdl.addConstrs((x[(i,j)] <= y[(i,j)] for i,j in edges),name = "R3")

    # R4 = mdl.addConstrs((y[(i,j)] <= (Q - demands[i]) * x[(i,j)] for i,j in edges), name = "R4")
    R4 = mdl.addConstrs((y[(i,j)] <= Q * x[(i,j)] for i,j in edges), name = "R4")

    R5 = mdl.addConstrs((d[i] + cost[(i,j)] - d[j] <= M * (1 - x[(i,j)]) for i,j in edges), name = "R5")

    R6 = mdl.addConstrs((d[i] >= earliest[i] for i in nodes), name = "R6")

    R7 = mdl.addConstrs((d[i] <= latest[i] for i in nodes), name = "R7")


    mdl.Params.TimeLimit = time_limit
    mdl.Params.Threads = 1

    solution = mdl.optimize()

    solution_edges = SortedDict()
    for i,j in edges:
        if x[i,j].X > 0.9:
            solution_edges[j] = i

    obj = mdl.getObjective()
    objective_value = obj.getValue()

    time = mdl.Runtime
    best_bound = mdl.ObjBound
    gap = mdl.MIPGap

    # to display the solution given by cplex
    # if verbose == True:
    #     solution.display()
    # to visualize the graph
    if vis:
        visualize(ins.xcoords, ins.ycoords, solution_edges)

    return objective_value, time, best_bound, gap

def distance(i,j):
    return D[(i,j)]

def main():
    from heuristics import prim
    name, capacity, node_data = read_instance(f"gehring instances/200/C1_2_10.TXT")
    ins = instance(name, capacity, node_data, 200)
    # ins.capacity = 10
    # obj, time, best_bound, gap = relaxed_gurobi_solution(ins, vis = True)    
    initial_solution = prim(ins, vis = False, initial = True)
    print(initial_solution[1])
    # initial_solution = None
    (parent, gate, load, arrival), objective_value = gurobi_solution(ins, verbose= True, vis = True, start= initial_solution, time_limit=60, initial = True)
    print(objective_value)
if __name__ == "__main__": 
    main()

# Dada la solución relajada ir reparandola, tomar el subarbol y desde ahi agarrar todos hasta.
# ¿ Como determinar cual pertence a cada cluster?
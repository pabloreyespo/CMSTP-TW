from tokenize import ContStr
import gurobipy as gp
from gurobipy import GRB

from sortedcollections import SortedDict
from utilities import instance, visualize, read_instance, visualize_relaxed

def gurobi_solution(ins, vis = False, time_limit = 1800, verbose = False):

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
    mdl = gp.Model(ins.name)
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
    name, capacity, node_data = read_instance("Instances/r102.txt")
    ins = instance(name, capacity, node_data, 100)
    ins.capacity = 10
    # obj, time, best_bound, gap = relaxed_gurobi_solution(ins, vis = True)    
    obj, time, best_bound, gap = gurobi_solution(ins, vis = True)
    print(obj, time, best_bound, gap)

if __name__ == "__main__":
    main()

# Dada la solución relajada ir reparandola, tomar el subarbol y desde ahi agarrar todos hasta.
# ¿ Como determinar cual pertence a cada cluster?
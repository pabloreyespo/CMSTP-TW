import gurobipy as gp
from gurobipy import GRB

import pandas as pd
from utilities import read_instance, instance
import os

env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

def distance(i,j):
    return D[(i,j)]

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
    if verbose:
        mdl = gp.Model(ins.name)
    else:
        mdl = gp.Model(ins.name, env = env)

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
    try:
        obj = mdl.getObjective()
        objective_value = obj.getValue()
    except:
        objective_value = None

    time = mdl.Runtime
    best_bound = mdl.ObjBound
    gap = mdl.MIPGap


    return objective_value, time, best_bound, gap

def test(cap, nnodes, nombre, time_limit):
    
    instances = os.listdir("gehring instances/200")
    results = list()
    for inst in instances:
        print(f"{inst}-Q{cap}")
        name, capacity, node_data = read_instance(f"gehring instances/200/{inst}")
        ins = instance(name, capacity, node_data, nnodes)
        ins.capacity = cap
        obj, time, best_bound, gap = gurobi_solution(ins, vis = False, time_limit= time_limit, verbose = True) 
        dic = {"name": name, "LB": best_bound, "UB": obj, "gap": gap, "t": time}
        results.append(dic)

    df = pd.DataFrame(results)
    df.to_excel(f"{nombre}.xlsx", index= False)

if __name__ == "__main__":
    caps = [10000,20,15,10,5]
    nnodes = 200
    time_limit = 60
    for cap in caps:
        test(cap, nnodes, f"GUROBI n200 Q-{cap}", time_limit)
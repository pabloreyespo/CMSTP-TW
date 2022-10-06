import numpy as np
from math import sqrt

from docplex.mp.model import Model
import gurobipy as gp
from gurobipy import GRB

import numpy as np
from math import sqrt, inf
from sortedcollections import SortedDict
from time import perf_counter

from utilities import extract_data, visualize

class instance():
    def __init__(self, name, capacity, node_data, num, reset_demand = True):
        self.name = name
        self.n = num + 1
        self.capacity = int(capacity)
        self.index, self.xcoords, self.ycoords, self.demands, self.earliest, self.latest\
            = extract_data(node_data[:num+1])

        self.nodes = [i for i in range(num+1)]
        self.edges = [(i,j) for i in self.nodes for j in self.nodes[1:] if i != j]

        #demands = 1 for all nodes
        if reset_demand:
            self.demands = {i:1 for i in self.nodes}

        # cost = time = distance for simplicity

        global D
        D = np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(i+1,self.n):
                D[i,j] = self.dist(i,j)
                D[j,i] = D[i,j]

        self.D = D

    def dist(self,i,j):
        x = self.xcoords[i] - self.xcoords[j]
        y = self.ycoords[i] - self.ycoords[j]
        return sqrt(x**2 + y**2)


def cplex_solution(ins, vis = False, time_limit = 1800, verbose = False):

    nodes = ins.nodes
    nnodes = ins.n
    edges = ins.edges
    nodesv = nodes[1:]
    Q = ins.capacity
    earliest = ins.earliest
    latest = ins.latest

    demands = ins.demands

    # model and variables
    mdl = Model(ins.name)
    x = mdl.binary_var_dict(edges, name = "x") #
    y = mdl.integer_var_dict(edges, name = "y", lb = 0)
    d = mdl.continuous_var_dict(nodes, name = "d", lb = 0)

    # objective function
    mdl.minimize(mdl.sum(distance(i,j) * x[(i,j)] for i,j in edges))

    # restrictions
    for j in nodesv:
        mdl.add_constraint(mdl.sum(x[(i,j)] for i in nodes if i!=j) == 1)

    for j in nodesv:
        mdl.add_constraint(mdl.sum(y[(i,j)] for i in nodes if i!=j) - mdl.sum(y[(j,i)] for i in nodesv if i!=j) == demands[j])

    for i,j in edges:
        mdl.add_constraint(x[(i,j)] <= y[(i,j)])

    for i,j in edges:
        mdl.add_constraint(y[(i,j)] <= Q * x[(i,j)]) #  (Q - demands[i]) * x[(i,j)])

    for i,j in edges:
        mdl.add_indicator(x[(i,j)], d[i] + distance(i,j) <= d[j])

    for i in nodes:
        mdl.add_constraint(d[i] >= earliest[i])

    for i in nodes:
        mdl.add_constraint(d[i]  <= latest[i])

    mdl.parameters.timelimit = time_limit # timelimit = 30 minutes
    mdl.parameters.threads = 1 # only one cpu thread in use
    solution = mdl.solve(log_output = False)

    solution_edges = SortedDict()
    for i,j in edges:
        if x[(i,j)].solution_value > 0.9:
            solution_edges[j] = i

    objective_value = mdl.objective_value
    time = mdl.solve_details.time
    best_bound = mdl.solve_details.best_bound
    gap = mdl.solve_details.mip_relative_gap

    # to display the solution given by cplex
    if verbose == True:
        solution.display()
    # to visualize the graph
    if vis:
        visualize(ins.xcoords, ins.ycoords, solution_edges)

    return objective_value, time, best_bound, gap

def gurobi_solution(ins, vis = False, time_limit = 1800, verbose = False):

    nnodes = ins.n

    Q = ins.capacity
    earliest = ins.earliest
    latest = ins.latest
    demands = ins.demands

    edges, cost = gp.multidict({(i,j): D[i,j] for (i,j) in ins.edges})
    nodes, earliest, latest, demands = gp.multidict({i: (ins.earliest[i], ins.latest[i], ins.demands[i]) for i in ins.nodes })
    nodesv = nodes[1:]

    M =  10000000 #max(latest) + max(cost.values()) + 1

    # model and variables
    mdl = gp.Model(ins.name)
    x = mdl.addVars(edges, vtype = GRB.BINARY, name = "x") #
    y = mdl.addVars(edges, vtype = GRB.INTEGER, name = "y", lb = 0)
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
        if x[i,j].X > 0:
            solution_edges[j] = i

    obj = mdl.getObjective()
    objective_value = obj.getValue()

    time = mdl.Runtime
    best_bound = mdl.ObjBound
    gap = None # mdl.MIPGap

    # to display the solution given by cplex
    # if verbose == True:
    #     solution.display()
    # to visualize the graph
    if vis:
        visualize(ins.xcoords, ins.ycoords, solution_edges)

    return objective_value, time, best_bound, gap


def distance(i,j):
    return D[(i,j)]

if __name__ == "__main__":
    pass

from docplex.mp.model import Model
from sortedcollections import SortedDict

from utilities import read_instance, visualize, instance, visualize_relaxed

def cplex_solution(ins, vis = False, time_limit = 1800, verbose = False):

    nodes = ins.nodes
    nnodes = ins.n
    edges = ins.edges
    nodesv = nodes[1:]
    Q = ins.capacity
    earliest = ins.earliest
    latest = ins.latest
    global D
    D = ins.cost
    demands = ins.demands

    # model and variables
    mdl = Model(ins.name)
    x = mdl.binary_var_dict(edges, name = "x") #
    y = mdl.continuous_var_dict(edges, name = "y", lb = 0)
    d = mdl.continuous_var_dict(nodes, name = "d", lb = 0)

    M = max(latest) + D.max() * 2

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
        # mdl.add_indicator(x[(i,j)], d[i] + distance(i,j) <= d[j])
        mdl.add_constraint(d[i] + distance(i,j) - d[j] <= M * (1 - x[(i,j)]))

    for i in nodes:
        mdl.add_constraint(d[i] >= earliest[i])

    for i in nodes:
        mdl.add_constraint(d[i]  <= latest[i])

    mdl.parameters.timelimit = time_limit # timelimit = 30 minutes
    mdl.parameters.threads = 1 # only one cpu thread in use
    solution = mdl.solve(log_output = verbose)


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

def cplex_solution_indicator(ins, vis = False, time_limit = 1800, verbose = False):

    nodes = ins.nodes
    nnodes = ins.n
    edges = ins.edges
    nodesv = nodes[1:]
    Q = ins.capacity
    earliest = ins.earliest
    latest = ins.latest
    global D
    D = ins.cost
    demands = ins.demands

    # model and variables
    mdl = Model(ins.name)
    x = mdl.binary_var_dict(edges, name = "x") #
    y = mdl.continuous_var_dict(edges, name = "y", lb = 0)
    d = mdl.continuous_var_dict(nodes, name = "d", lb = 0)

    M = max(latest) + D.max() * 2

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
        # mdl.add_constraint(d[i] + distance(i,j) - d[j] <= M * (1 - x[(i,j)]))

    for i in nodes:
        mdl.add_constraint(d[i] >= earliest[i])

    for i in nodes:
        mdl.add_constraint(d[i]  <= latest[i])

    mdl.parameters.timelimit = time_limit # timelimit = 30 minutes
    mdl.parameters.threads = 1 # only one cpu thread in use
    solution = mdl.solve(log_output = verbose)


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

def relaxed_cplex_solution(ins, vis = False, time_limit = 1800, verbose = False):

    nodes = ins.nodes
    nnodes = ins.n
    edges = ins.edges
    nodesv = nodes[1:]
    Q = ins.capacity
    earliest = ins.earliest
    latest = ins.latest
    global D
    D = ins.cost
    demands = ins.demands

    # model and variables
    mdl = Model(ins.name)
    x = mdl.continuous_var_dict(edges, name = "x", lb = 0, ub = 1) #
    y = mdl.continuous_var_dict(edges, name = "y", lb = 0)
    d = mdl.continuous_var_dict(nodes, name = "d", lb = 0)

    M = max(latest) + D.max() * 2

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
        mdl.add_constraint(d[i] + distance(i,j) - d[j] <= M * (1 - x[(i,j)]))

    for i in nodes:
        mdl.add_constraint(d[i] >= earliest[i])

    for i in nodes:
        mdl.add_constraint(d[i]  <= latest[i])

    mdl.parameters.timelimit = time_limit # timelimit = 30 minutes
    mdl.parameters.threads = 1 # only one cpu thread in use
    solution = mdl.solve(log_output = False)

    solution_edges = list()
    intensity = dict()
    for i,j in edges:
        if x[(i,j)].solution_value > 0:
            solution_edges.append((i,j))
            intensity[(i,j)] = x[(i,j)].solution_value

    objective_value = mdl.objective_value
    time = mdl.solve_details.time
    best_bound = mdl.solve_details.best_bound
    gap = mdl.solve_details.mip_relative_gap

    # to display the solution given by cplex
    if verbose == True:
        solution.display()
    # to visualize the graph
    if vis:
        visualize_relaxed(ins.xcoords, ins.ycoords, solution_edges, intensity)

    return objective_value, time, best_bound, gap

def distance(i,j):
    return D[(i,j)]

def main():
    name, capacity, node_data = read_instance("Instances/r101.txt")
    ins = instance(name, capacity, node_data, 100)
    #obj, time, best_bound, gap = relaxed_cplex_solution(ins, vis = True)
    obj, time, best_bound, gap = cplex_solution(ins, vis = True)

if __name__ == "__main__":
    main()

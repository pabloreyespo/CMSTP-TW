from docplex.mp.model import Model
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from math import sqrt

from sympy import O
from utilities import extract_data, read_salomon,  visualizar

class CMSTP_ATW():
    def __init__(self, name, capacity, nodes):
        self.name = name
        self.capacity = int(capacity)
        self.nodes = nodes
        self.index, self.xcoords, self.ycoords, self.demands, self.earliest_times, self.latest_times\
            = extract_data(nodes)

    def dist(self,i,j):
        x = self.xcoords[i] - self.xcoords[j]
        y = self.ycoords[i] - self.ycoords[j]
        return round(sqrt(x**2 + y**2),4)

    def solve_cplex(self, num = 25, vis = False, Q = None, verbose = False):

        if Q is None:
            Q = self.capacity

        xcoords = self.xcoords[:num+1]
        ycoords = self.ycoords[:num+1]

        nodes = [i for i in range(num+1)]
        nodesv = nodes[1:]

        edges = [(i,j) for i in nodes for j in nodesv if i != j]
        edgesv = [(i,j) for i in nodesv for j in nodesv if i != j]

        earliest = {(i,j): min(self.earliest_times[i], self.earliest_times[j]) for i,j in edges}
        latest = {(i,j): max(self.latest_times[i], self.latest_times[j]) for i,j in edges}
        demands = {i:1 for i in nodes}

        cost = {(i,j): self.dist(i,j) for i,j in edges}

        mdl = Model(self.name)

        x = mdl.binary_var_dict(edges, name = "x")
        y = mdl.continuous_var_dict(edges, name = "y", lb = 0)
        s = mdl.continuous_var_dict(edges, name = "s", lb = 0)

        # Cálculos de big-M

        # M = max(latest.values()) + max(cost.values()) - min(latest.values()) + 1
        M1 = max(latest.values()) + max(cost.values()) + 1
        M = M1
        print(f"m: {M}")
        mdl.minimize(mdl.sum(cost[(i,j)] * x[(i,j)] for i,j in edges))

        for j in nodesv:
            mdl.add_constraint(mdl.sum(x[(i,j)] for i in nodes if i!=j) == 1)

        for j in nodesv:
            mdl.add_constraint(mdl.sum(y[(i,j)] for i in nodes if i!=j) - mdl.sum(y[(j,i)] for i in nodesv if i!=j) == demands[j])

        for i,j in edges:
            mdl.add_constraint(x[(i,j)] <= y[(i,j)])

        for i,j in edges:
            mdl.add_constraint(y[(i,j)] <= (Q - demands[i]) * x[(i,j)])

            # si uso edgesv en vez de edges la restricción de Q desaparece, sin embargo genera la misma topologia

        for i,j in edges:
            for w in nodesv:
                if w != j and i != w:
                    mdl.add_constraint(s[(i,j)] + cost[(i,j)] - s[(j,w)] <= (1-x[(j,w)])*M1 + (1-x[(i,j)])*M1)
                    # mdl.add_indicator(x[(i,j)] and x[(j,w)], s[(i,j)] + cost[(i,j)] <= s[(j,w)])

        for i,j in edges:
            mdl.add_constraint(s[(i,j)] >= earliest[(i,j)]*x[(i,j)])

        for i,j in edges:
            mdl.add_constraint(s[(i,j)] + cost[(i,j)]  <= latest[(i,j)]*x[(i,j)] + (1- x[(i,j)])*M)
            # mdl.add_indicator(x[(i,j)], s[(i,j)] + cost[(i,j)] <= latest[(i,j)])

        msg = mdl.export_to_string()
        a = open("salida.txt", "w")
        a.write(msg)
        a.close()
        mdl.parameters.timelimit = 1000
        solution = mdl.solve(log_output = True)
        print(mdl.get_solve_status())
        solution.display() #estas lienas se encuentran como comentario porque preferí simplificar la información que despliega el programa

        solution_edges = [i for i in edges if x[i].solution_value > 0.9]
        objective_value = mdl.objective_value

        if verbose:
            times_dict = {(i,j) : (s[(i,j)].solution_value,s[(i,j)].solution_value + cost[(i,j)]) for i,j in solution_edges}
            
            print(f"({i:2},{j:2}) |{'earl':6} - {'late':6}| {'time':6} {'depart':6} {'arriv':6} {'cargo':6}")
            for (i,j) in solution_edges:
                print(f"({i:2},{j:2}) |{earliest[(i,j)]:6.2f} - {latest[(i,j)]:6.2f}| {cost[(i,j)]:6.2f} {s[(i,j)].solution_value:6.2f} {times_dict[(i,j)][1]:6.2f} {y[(i,j)].solution_value:6.2f}")
        if vis:
            visualizar(xcoords, ycoords, solution_edges)

        return solution_edges, objective_value

    def solve_alt(self, num = 25, vis = False, Q = None, verbose = False):

        if Q is None:
            Q = self.capacity

        xcoords = self.xcoords[:num+1]
        ycoords = self.ycoords[:num+1]

        nodes = [i for i in range(num+1)]
        nodesv = nodes[1:]

        edges = [(i,j) for i in nodes for j in nodesv if i != j]
        edgesv = [(i,j) for i in nodesv for j in nodesv if i != j]

        earliest = self.earliest_times
        latest = self.latest_times

        demands = {i:1 for i in nodes}

        cost = {(i,j): self.dist(i,j) for i,j in edges}

        mdl = Model(self.name)

        x = mdl.binary_var_dict(edges, name = "x")
        y = mdl.integer_var_dict(edges, name = "y", lb = 0)
        d = mdl.continuous_var_dict(nodes, name = "s", lb = 0)

        # Cálculos de big-M

        # M = max(latest.values()) + max(cost.values()) - min(latest.values()) + 1
        M1 = max(latest) + max(cost.values()) + 1
        M = M1
        print(f"m: {M}")
        mdl.minimize(mdl.sum(cost[(i,j)] * x[(i,j)] for i,j in edges))

        for j in nodesv:
            mdl.add_constraint(mdl.sum(x[(i,j)] for i in nodes if i!=j) == 1)

        for j in nodesv:
            mdl.add_constraint(mdl.sum(y[(i,j)] for i in nodes if i!=j) - mdl.sum(y[(j,i)] for i in nodesv if i!=j) == demands[j])

        for i,j in edges:
            mdl.add_constraint(x[(i,j)] <= y[(i,j)])

        for i,j in edges:
            mdl.add_constraint(y[(i,j)] <= (Q - demands[i]) * x[(i,j)])

            # si uso edgesv en vez de edges la restricción de Q desaparece, sin embargo genera la misma topologia

        for i,j in edges:
            mdl.add_indicator(x[(i,j)], d[i] + cost[(i,j)] <= d[j])

        for i in nodes:
            mdl.add_constraint(d[i] >= earliest[i])

        for i in nodes:
            mdl.add_constraint(d[i]  <= latest[i])
            # mdl.add_indicator(x[(i,j)], s[(i,j)] + cost[(i,j)] <= latest[(i,j)])

        msg = mdl.export_to_string()
        a = open("salida.txt", "w")
        a.write(msg)
        a.close()
        mdl.parameters.timelimit = 1000
        solution = mdl.solve(log_output = True)
        print(mdl.get_solve_status())
        solution.display() #estas lienas se encuentran como comentario porque preferí simplificar la información que despliega el programa
    
        solution_edges = [i for i in edges if x[i].solution_value > 0.9]
        objective_value = mdl.objective_value

        if verbose:
            times_dict = {(i,j) : (s[(i,j)].solution_value,s[(i,j)].solution_value + cost[(i,j)]) for i,j in solution_edges}
            
            print(f"({i:2},{j:2}) |{'earl':6} - {'late':6}| {'time':6} {'depart':6} {'arriv':6} {'cargo':6}")
            for (i,j) in solution_edges:
                print(f"({i:2},{j:2}) |{earliest[(i,j)]:6.2f} - {latest[(i,j)]:6.2f}| {cost[(i,j)]:6.2f} {s[(i,j)].solution_value:6.2f} {times_dict[(i,j)][1]:6.2f} {y[(i,j)].solution_value:6.2f}")

        if vis:
            visualizar(xcoords, ycoords, solution_edges)

        return solution_edges, objective_value

    def solve_gurobi(self, num = 20, vis = False, Q = None):

        earl = self.earliest_times
        late = self.latest_times
        if Q is None:
            Q = self.capacity
        xcoords = self.xcoords[:num+1]
        ycoords = self.ycoords[:num+1]

        nodes, demands = gp.multidict({i:1 for i in range(num+1)})
        nodesv = nodes[1:]

        edges, cost, earliest, latest = \
             gp.multidict({(i,j): [self.dist(i,j), min(earl[i],earl[j]), max(late[i], late[j])] for i in nodes for j in nodesv if i != j})
        # edgesv = [(i,j) for i in nodesv for j in nodesv if i != j]

        mdl = gp.Model(self.name)

        x = mdl.addVars(edges, name = "x", vtype = GRB.BINARY)
        y = mdl.addVars(edges, name = "y")
        s = mdl.addVars(edges, name = "s")

        M = max(latest.values()) + max(cost.values()) - min(latest.values()) + 1
        M1 = max(latest.values()) + max(cost.values()) + 1
        
        mdl.setObjective(x.prod(cost))

        R1 = mdl.addConstrs((gp.quicksum(x[(i,j)] for i in nodes if i!=j) == 1 for j in nodesv),name = "R1")
        R2 = mdl.addConstrs((gp.quicksum(y[(i,j)] for i in nodes if i!=j) - gp.quicksum(y[(j,i)] for i in nodesv if i!=j) == demands[j] for j in nodesv), name = "R2")
        R3 = mdl.addConstrs((x[(i,j)] <= y[(i,j)] for i,j in edges),name = "R3")
        R4 = mdl.addConstrs((y[(i,j)] <= (Q - demands[j]) * x[(i,j)] for i,j in edges),name = "R4")
        R5 = mdl.addConstrs((s[(i,j)] + cost[(i,j)] - s[(j,w)] <= (1-x[(j,w)])*M1 + (1-x[(i,j)])*M1 for w in nodesv for (i,j) in edges if w != j and i!=j), name = "R5")
        R6 = mdl.addConstrs((s[(i,j)] >= earliest[(i,j)]*x[(i,j)] for i,j in edges), name = "R6")
        R7 = mdl.addConstrs((s[(i,j)] + cost[(i,j)] - latest[(i,j)]*x[(i,j)] <= (1- x[(i,j)])*M for i,j in edges), name = "R7")

        # print(mdl.export_to_string())
        
        solution = mdl.optimize()    

        solution_edges = [i for i in edges if x[i].X > 0.9]
        obj = mdl.getObjective()
        objective_value = obj.getValue()

        for (i,j) in solution_edges:
            print(f"({i:2},{j:2}) |{earliest[(i,j)]:6.2f} - {latest[(i,j)]:6.2f}| {cost[(i,j)]:6.2f} {s[(i,j)].X:6.2f}")

        if vis:
            visualizar(xcoords, ycoords, solution_edges)

        return solution_edges, objective_value
        

def main():
    name, capacity, nodes = read_salomon("instances/r101.txt")
    ins = CMSTP_ATW(name, capacity, nodes)
    solution_edges, obj = ins.solve_alt(100,vis = True, Q = 20)
    print(solution_edges)
    print(obj)
    return None

if __name__ == "__main__":
    main()
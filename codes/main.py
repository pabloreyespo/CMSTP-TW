from exact_solutions_cplex import *
from exact_solutions_gurobi import *
from metaheuristic_cplex import *
from metaheuristic_gurobi import *
from heuristics import *
from utilities import *


def main():
    name, capacity, node_data = read_instance("Instances/rc105.txt")
    ins = instance(name, capacity, node_data, 100)

if __name__ == "__main__":
    main()

# darle una vuelta al relajamiento
# ejecutar todo de nuevo: cplex, gurobi, [7], [8]



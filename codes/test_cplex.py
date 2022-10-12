from exact_solutions_cplex import *
from utilities import *

name, capacity, node_data = read_instance(f"instances/c105.txt")
ins = instance(name, capacity, node_data, 100)

obj, time, best_bound, gap = cplex_solution(ins, vis = True, verbose = True)
print("Big-M")
print(obj, time)

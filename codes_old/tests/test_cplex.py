import pandas as pd
from exact_solutions_cplex import cplex_solution
from utilities import read_instance, instance
import os

def test(cap, nnodes, nombre):
    
    instances = os.listdir("instances")
    results = list()
    for inst in instances:
        print(f"{inst}-Q{cap}")
        name, capacity, node_data = read_instance(f"instances/{inst}")
        ins = instance(name, capacity, node_data, nnodes)
        ins.capacity = cap
        obj, time, best_bound, gap = cplex_solution(ins, vis = False, time_limit= 1800, verbose = False) 
        dic = {"name": name, "LB": best_bound, "UB": obj, "gap": gap, "t": time}
        results.append(dic)

    df = pd.DataFrame(results)
    df.to_excel(f"resultados/{nombre}.xlsx", index= False)

if __name__ == "__main__":
    caps = [10000,20,15,10,5]
    nnodes = 100
    for cap in caps:
        test(cap, nnodes, f"CPLEX Q-{cap}")

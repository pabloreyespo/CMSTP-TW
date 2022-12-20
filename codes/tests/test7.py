import pandas as pd
from metaheuristic_gurobi import ILS_solution
from utilities import read_instance, instance
import os
from math import inf

def test(cap, nnodes ,pa, pb, lsp, e_param, f_param, elite_search_param ,iterMax, nombre):
    
    instances = os.listdir("instances")
    results = list()
    for inst in instances:
        print(f"{inst}-Q{cap}")
        name, capacity, node_data = read_instance(f"instances/{inst}")
        ins = instance(name, capacity, node_data, nnodes)
        ins.capacity = cap

        best_obj = inf
        time_sum = 0
        solution_sum = 0
        for i in range(10):
            obj, time, best_bound, gap = ILS_solution(ins, semilla = i, pa = pa , pb = pb, lsp = lsp, 
                    feasibility_param= f_param,elite_param=e_param, elite_revision_param = elite_search_param,iterMax = iterMax) 
                    
            if obj < best_obj:
                best_obj = obj
            time_sum += time
            solution_sum += obj
        dic = {"name": f"{name}","min": best_obj, "avg": solution_sum/10,  "t_avg": time_sum/10 }
        results.append(dic)

    df = pd.DataFrame(results)
    df.to_excel(f"{nombre}.xlsx", index= False)

if __name__ == "__main__":
    caps = [10000,20,15,10,5]
    nnodes = 100
    for cap in caps:
        test(cap, nnodes, 0.4, 0.6, 0.8, 250, 100, 1500, 15000, f"Experimento 7 Q-{cap}")
import pandas as pd
from metaheuristic import *
import os

def test8(method ,cap, nnodes ,pa, pb, lsp, e_param, f_param, elite_search_param,iterMax, nombre):
    
    # instances = os.listdir("instances")
    instances = ["r204.txt"]
    results = list()
    for inst in instances:
        print(f"{inst}-Q{cap}")
        name, capacity, node_data = read_instance(f"instances/{inst}")
        ins = instance(name, capacity, node_data, nnodes)
        ins.capacity = cap

        if method == "ils":

            best_obj = inf
            time_sum = 0
            solution_sum = 0

            for i in range(10):
                obj, time, best_bound, gap = ILS_solution(ins, semilla = i, pa = pa , pb = pb, lsp = lsp, b = [1,0,1,0.2,0.4,0], mu = 1, alpha = 1,
                        feasibility_param= f_param,elite_param=e_param, elite_revision_param = elite_search_param,iterMax = iterMax) 

                if obj < best_obj:
                    best_obj = obj

                time_sum += time
                solution_sum += obj

            dic = {"name": f"{name}","min": best_obj, "avg": solution_sum/10,  "t_avg": time_sum/10 }
            results.append(dic)

            df = pd.DataFrame(results)
            df.to_excel(f"{nombre}.xlsx", index= False)
        
        elif method == "cplex":
            obj, time, best_bound, gap = cplex_solution(ins, vis = False, time_limit= 1800, verbose = False) 
            dic = {"name": name, "Best": best_bound, "gap": gap, "t": time}
            results.append(dic)
    
        elif method == "gurobi":
            obj, time, best_bound, gap = gurobi_solution(ins, vis = False, time_limit= 1800, verbose = False) 
            dic = {"name": name, "Best": best_bound, "gap": gap, "t": time}
            results.append(dic)
        else: 
            return None

    df = pd.DataFrame(results)
    df.to_excel(f"resultados/{nombre}.xlsx", index= False)

if __name__ == "__main__":
    # caps = [10000,20,15,10,5]
    caps = [10000]
    nnodes = 100
    for cap in caps:
        test8("ils", cap, nnodes, 0.4, 0.6, 1 , 250, 100, 1500, 15000, f"Experimento 8 Q-{cap}")
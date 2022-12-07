import pandas as pd
from codes.heuristics import ESGH_solution
from metaheuristic_gurobi import *
from metaheuristic_cplex import *
from heuristics import LPDH_solution, ESGH_solution
from utilities import *
import os

def test_heuristics(cap, alg):
    instances = os.listdir("Instances")
    results = list()

    if alg == "LPDH":
        solution_alg = LPDH_solution
    elif alg == "ESGH":
        solution_alg = ESGH_solution
    else:
        return
    for inst in instances:
        print(inst)
        name, capacity, node_data = read_instance(f"instances/{inst}")
        ins = instance(name, capacity, node_data, 100)
        ins.capacity = cap

        mejor = inf
        bb = [0,0.5,1]

        values = np.array(np.meshgrid(1,bb,bb,bb,bb,bb)).T.reshape(-1,6)
        times = 0

        for i in range(len(values)):
            b = values[i]

            pred, obj, time, best_bound, gap = solution_alg(ins, b = b, vis = False)

            times += time

            if obj < mejor:
                mejor = obj
                b_mejor  = b.copy()
                print(b_mejor, obj)

        dic = {"name": name,"obj": mejor, "time": times/len(values), "b": b_mejor }
        results.append(dic)

    df = pd.DataFrame(results)
    df.to_excel(f"Resultados {alg} Q{cap}.xlsx", index= False)

def test(method ,cap, nnodes ,pa, pb, lsp, e_param, f_param, elite_search_param ,iterMax, nombre):
    
    instances = os.listdir("instances")
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
    caps = [10000,20,15,10,5]
    nnodes = 100
    for cap in caps:
        test("ils", cap, nnodes, 0.4, 0.6, 1 , 250, 100, 1500, 15000, f"Experimento 4 Q-{cap}")
        test("ils", cap, nnodes, 0.4, 0.6, 0.8, 250, 100, 1500, 15000, f"Experimento 6 Q-{cap}")
        # test("ils", cap, nnodes, 1, 1, 1, 250, 100, 10**8, 50000, f"Experimento 7 Q-{cap}")
        test("cplex", cap, nnodes, None, None, None, None,None, None, None, f"CPLEX Q-{cap}")
        test("gurobi", cap, nnodes, None, None, None, None,None, None, None, f"GUROBI 6 Q-{cap}")


    # [1] ultimate_test(20, 1, 0, 1, "Mejor configuracion 15000it.xlsx")
    # [2] ultimate_test(20, 1, 0, 0.8, "Mejor configuracion 15000it con busqueda intensiva")
    # [3] ultimate_test(20, 1/3, 2/3, 1, "Mejor configuracion 15000it 3 perturbaciones parejas")  
    # [4] ultimate_test(20, 0.4, 0.6, 1, "Mejor configuracion 15000it 3 perturbaciones cargado")
    # [5] ultimate_test(20, 0.5, 0, 1, "Mejor configuracion 15000it 2 perturbaciones parejas")
    # [6] ultimate_test(20, 0.5, 0, 0.8, "Mejor configuracion 15000it 2 perturbaciones 40-60-40 y busqueda intensiva")
    # [7] ultimate_test(20, 1, 1, 1, "Mejor configuracion 50000it 2 perturbaciones parejas")
    
        # test_iter(10**6, 10**6, 0, 1000, [1,0,0,0,0,0], 0, it, f"ILS {it} iteraciones")

        # final_test(10**6, 10**6, 0, cap, f"ILS Normal Q{cap}")
        # final_test(250, 100, 0.5, cap, [1,0,0,0,0,0], 0, f"ILS Prim - Infactibles - Elite Q{cap}")
        # final_test(10**6, 10**6, 0.5, cap, f"ILS Penalización Q{cap}")
        # final_test(10**6, 100, 0.5, cap, f"ILS Penalización e Infactibles Q{cap}")
        # final_test(250, 100, 0.5, cap, f"ILS Penalización y Elite Q{cap}")


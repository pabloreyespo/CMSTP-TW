# from exact_solutions_cplex import *
from exact_solutions_gurobi import *
# from metaheuristic_cplex import *
from metaheuristic_gurobi import *
from heuristics import *
from utilities import *
import os

env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

def test(q, a, f, e, s, n, x, y, z, r, l, t, g, nnodes, ins_folder, nombre):

    instances = os.listdir(ins_folder)
    results = list()
    for p in instances:
        print(p)
        best_obj = inf
        time_sum = 0
        solution_sum = 0

        name, capacity, node_data = read_instance(ins_folder + "/"+  p)
        ins = instance(name, capacity, node_data, nnodes)
        ins.capacity = q

        global BRANCH_TIME
        BRANCH_TIME = t

        generate_solution = lambda x: gurobi_solution(x, vis = False, time_limit= g, verbose = False, initial=True)
        (parent, gate, load, arrival), objective_value= generate_solution(ins)
        initial_solution = lambda x: ((parent.copy(), gate.copy(), load.copy(), arrival.copy()), objective_value)
        solution_sum = 0
        for i in range(10):
            pa = x
            pb = x + y

            obj, time, best_bound, gap = ILS_solution_timelimit(
                ins, semilla = i, acceptance = a,
                feasibility_param = f, elite_param = e, elite_size = s, p = n,
                pa = pa, pb = pb, lsp = l, initial_solution = initial_solution,
                elite_revision_param = r, vis  = False, verbose = False, time_limit = 60 - g)
            if obj < best_obj:
                best_obj = obj
            time_sum += time
            solution_sum += obj
            
        dic = {"name": f"{name}","min": best_obj, "avg": solution_sum/10,  "t_avg": time_sum/10 }
        results.append(dic)

    df = pd.DataFrame(results)
    df.to_excel(f"{nombre}.xlsx", index= False)

if __name__ == "__main__":
    capacities = [10000, 20, 15, 10, 5]
    g = 30
    a, f, e, s, n, x, y, z, r, l, t, = 0.008 ,9500 ,8000 ,50 ,6.229 ,0.125 ,0.568 ,0.307 ,5000 ,0.152 ,4
    configuracion = "conf30-4"
    for q in capacities:
        ins_folder = "Instances"
        test(q, a, f, e, s, n, x, y, z, r, l, t, g, 100, ins_folder, f"{configuracion} Q-{q} n-100")
        # ins_folder = "gehring instances/200"
        # test(q, a, f, e, s, n, x, y, z, r, l, t, g, 150, ins_folder, f"{configuracion} Q-{q} n-150")
        # ins_folder = "gehring instances/200"
        # test(q, a, f, e, s, n, x, y, z, r, l, t, g, 200, ins_folder, f"{configuracion} Q-{q} n-200")

from matheuristic_cython_resignado import *
import getopt,sys

env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

def test(gurobi_time, q, nnodes, ins_folder):
    global xcoords, ycoords, latest, earliest, D
    instances = os.listdir(ins_folder)
    results = list()

    pp1 = p1
    pp2 = p1 + p2
    pp3 = p1 + p2 + p3
    lsp1 = l1
    lsp2 = l1 + l2

    for p in instances:
        print(p)
        best_obj = inf
        time_sum = 0
        solution_sum = 0

        name, capacity, node_data = read_instance(ins_folder + "/"+  p)
        ins = instance(name, capacity, node_data, nnodes)
        ins.capacity = q
        initialize(ins)

        s, cost = prim(ins, vis = False, initial = True)
        print("prim:", cost)
        s, cost = gurobi_solution(ins, vis = False, time_limit= gurobi_time, verbose = False, initial=True, start = s)
        print("gurobi:", cost)
        initial_solution = lambda x: (deepcopy(s), cost)
        for i in range(10):
            obj, time, best_bound, gap = ILS_solution(
                ins, semilla = i, acceptance = acceptance, elite_size = size_elite, 
                pp1 = pp1, pp2 = pp2, pp3 = pp3, lsp1 = lsp1, lsp2= lsp2, initial_solution = initial_solution,
                elite_revision_param = revision_param, vis  = False, verbose = False, time_limit = 60 - gurobi_time)
            if obj < best_obj:
                best_obj = obj
            solution_sum += obj
            time_sum += time

        dic = {"name": f"{name}","min": best_obj, "avg": solution_sum/10,  "t_avg": time_sum/10 }
        results.append(dic)

    df = pd.DataFrame(results)
    df.to_excel(f"{nombre}.xlsx", index= False)


if __name__ == "__main__":
    argv = "-a 0.048 -s 20 -x 0.058 -y 0.744 -z 0.178 -c 0.019 -r 6500 -u 0.013 -v 0.1 -w 0.887 -t 5"
    argv = argv.split()
    try:
        opts, args = getopt.getopt(argv, 'a:s:x:y:z:c:r:u:v:w:t:')
    except getopt.GetoptError:
        print ('test.py -a acceptance -f feasibility_param -e elite_param -s size_elite -n penalization -x pert1 -y pert2 -z pert3 -c pert4 -r revision_param -u local1 -v local2 -w local3 -t branch_time')
        sys.exit(2)

    for opt, arg in opts:
        print(opt,arg)
        if opt == '-a':
            acceptance = float(arg)
        elif opt == '-s':
            size_elite = int(arg)
        elif opt == '-x':
            p1 = float(arg)
        elif opt == '-y':
            p2 = float(arg)
        elif opt == '-z':
            p3 = float(arg)
        elif opt == '-c':
            p4 = float(arg)
        elif opt == '-r':
            revision_param = int(round(float(arg)))
        elif opt == '-u':
            l1 = float(arg)
        elif opt == '-v':
            l2 = float(arg)
        elif opt == '-w':
            l3 = float(arg)
        elif opt == '-t':
            BRANCH_TIME = float(arg)

    capacities = [10000, 20, 15, 10, 5]
    gurobi_time = 30
    nnodes = 100
    for q in capacities:
        ins_folder = "Instances"
        nombre = f"3LM RESIGNADO {gurobi_time} Q-{q} n-{nnodes}"
        test(gurobi_time, q, nnodes, ins_folder)
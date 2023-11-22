from matheuristic_prompt import *
import getopt,sys

env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

def main(gurobi_prop, ils_prop, global_time, q, nnodes):
    results = list()

    gurobi_time = global_time * gurobi_prop
    ils_time = global_time * ils_prop
    global_time = global_time - gurobi_time

    name, capacity, node_data = read_instance(path)
    ins = instance(name, capacity, node_data, nnodes)
    ins.capacity = q
    initialize(ins)

    mdl = write_model(ins)
    ins.mdl = mdl.copy()

    s, cost = prim(ins, vis = False, initial = True)
    print("prim:", cost)

    s, cost, optimal = gurobi_fast_solution(ins, time_limit= gurobi_time, start = s)
    print("gurobi:", cost)

    initial_solution = (deepcopy(s), cost)
    if not optimal:
        solution_sum = 0
        tries = 10
        for seed in range(tries):
            print(seed)
            time = perf_counter()
            try:
                s, cost = ILS_solution(ins, semilla = seed, initial_solution = deepcopy(initial_solution), time_limit = ils_time )
                print("ILS:", cost)

                while perf_counter() - time < global_time:
                    time_left = global_time - perf_counter() + time
                    print(time_left)
                    s, cost, optimal = gurobi_fast_solution(ins, time_limit= min(gurobi_time, time_left), start = deepcopy(s), rando = True)
                    print("gurobi:", cost)

                    initial_ = (deepcopy(s), cost)
                    if perf_counter() - time < global_time:
                        time_left = global_time - perf_counter() + time
                        print(time_left)
                        s, cost = ILS_solution(ins, semilla = seed, initial_solution = deepcopy(initial_), time_limit = min(ils_time, time_left) )
                        print("ILS:", cost)
            except:
                print(f"Error en la ejecución {seed} de la instancia {path} con carga {q}")
                s, cost = deepcopy(initial_solution)

            solution_sum += cost
        print("Mejor", solution_sum/10)
    else:
        print("Mejor", cost)

if __name__ == "__main__":
    argv = sys.argv[1:] # + "-a 0.032 -f 10000 -e 20000 -r 6000 -s 20 -n 7.001 -x 0.121 -y 0.677 -z 0.186 -c 0.016 -u 0.013 -v 0.157 -w 0.830 -b 5".split()
    try:
        opts, args = getopt.getopt(argv, 'p:Q:K:a:d:f:e:r:s:n:x:y:z:c:u:v:w:b:', 
                                   ["path = ","capacity = ", "nnodes = ",
                                    "acceptance = ", "rando = ","feasibility_param = ","elite_param = ","size_elite = ","penalization = ",
                                    "p1 = ","p2 = ","p3 = ","p4 = ","revision_param = ","local1 = ","local2 = ","local3 = ","branch_time = "])
        print("Leido")
    except getopt.GetoptError:
        print ('test.py -q capacity -k nnodes -a acceptance -f feasibility_param -e elite_param -s size_elite -n penalization -x pert1 -y pert2 -z pert3 -c pert4 -r revision_param -u local1 -v local2 -w local3 -b branch_time')
    
    for opt, arg in opts:
        if opt in ['-p','--path']:
            path = str(arg)
        elif opt in ['-Q','--capacity']:
            capacity = int(arg)
        elif opt in ['-K','--nnodes']:
            nnodes = int(arg)
        elif opt in ['-a','--acceptance']:
            mu_acceptance = float(arg)
        elif opt in ['-d','--rando']:
            RANDO_PARAM = float(arg)
        elif opt in ['-f','--feasibility_param']:
            alpha_unfeasible = int(round(float(arg)))
        elif opt in ['-e','--elite_param']:
            beta_elite = int(round(float(arg)))
        elif opt in ['-r','--revision_param']:
            gamma_intensification = int(round(float(arg)))
        elif opt in ['-s','--size_elite']:
            elite_size = int(arg)
        elif opt in ['-n','--penalization']:
            rho = float(arg)
        elif opt in ['-x','--p1']:
            theta1 = float(arg)
        elif opt in ['-y','--p2']:
            theta2 = float(arg)
        elif opt in ['-z','--p3']:
            theta3 = float(arg)
        elif opt in ['-c','--p4']:
            theta4 = float(arg)
        elif opt in ['-u','--local1']:
            phi1 = float(arg)
        elif opt in ['-v','--local2']:
            phi2 = float(arg)
        elif opt in ['-w','--local3']:
            phi3 = float(arg)
        elif opt in ['-b','--branch_time']:
            BRANCH_TIME = float(arg)

    INITIAL_TRIGGER = 40
    gurobi_prop = 20
    ils_prop = 10
    global_time = 60
    main(gurobi_prop=gurobi_prop/60, ils_prop=ils_prop/60, global_time=global_time, q=capacity, nnodes=nnodes)

# -p instances/r106.txt -Q 10000 -K 100 -a 0.032 -f 10000 -e 20000 -r 6000 -s 20 -n 7.001 -x 0.121 -y 0.677 -z 0.186 -c 0.016 -u 0.013 -v 0.157 -w 0.830 -b 5
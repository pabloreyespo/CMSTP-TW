# from exact_solutions_cplex import *
from exact_solutions_gurobi import *
# from metaheuristic_cplex import *
from metaheuristic_gurobi import *
from heuristics import *
from utilities import *

import getopt,sys

env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

def main():
    generate_solution = lambda x: gurobi_solution(x, vis = False, time_limit= gurobi_time, verbose = False, initial=True)
    (parent, gate, load, arrival), objective_value= generate_solution(ins)
    initial_solution = lambda x: ((parent.copy(), gate.copy(), load.copy(), arrival.copy()), objective_value)
    solution_sum = 0
    for i in range(10):

        print(i)

        pa = p1
        pb = p1 + p2

        obj, time, best_bound, gap = ILS_solution(
            ins, semilla = i, acceptance = acceptance,
            feasibility_param = feasibility_param, elite_param = elite_param, elite_size = size_elite, p = penalization,
            pa = pa, pb = pb, lsp = local_search_param, initial_solution = initial_solution,
            elite_revision_param = revision_param, vis  = False, verbose = False, time_limit = 60 - gurobi_time)
        solution_sum += obj

    print("Mejor", solution_sum/10)


if __name__ == "__main__":
    argv = sys.argv[1:]

    # path = 'Instances/r101.txt' #"-p"
    # Q = 10
    # acceptance = 0.05 # '-a'
    # feasibility_param = 1000 # '-f'
    # elite_param = 2500 # '-e'
    # size_elite = 20 # '-s'
    # penalization = 0.5 # '-z'
    # p1 = 0.4 # '-p1'
    # p2 = 0.2 # '-p2'
    # p3 = 0.4 # '-p3'
    # revision_param = 1500 # '-r'
    # local_search_param = 0.8 # '-l'
    # BRANCH_TIME = 1 # '-t'
    # gurobi_time = 30 # '-g'

    try:
        opts, args = getopt.getopt(argv, 'p:Q:a:f:e:s:n:x:y:z:r:l:t:g:')
    except getopt.GetoptError:
        print ('test.py -p path -a acceptance -f feasibility_param -e elite_param -s size_elite -n penalization -p1 prob1 -p2 prob2 -p3 prob3 -r revision_param -l local_search -t branch_time -g gurobi_time')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-p':
            print(arg)
            name, capacity, node_data = read_instance(arg)
            ins = instance(name, capacity, node_data, 100)
        elif opt == '-Q':
            Q = int(arg)
            ins.capacity = Q
        elif opt == '-a':
            acceptance = float(arg)
        elif opt == '-f':
            feasibility_param = int(round(float(arg)))
        elif opt == '-e':
            elite_param = int(round(float(arg)))
        elif opt == '-s':
            size_elite = int(arg)
        elif opt == '-n':
            penalization = float(arg)
        elif opt == '-x':
            p1 = float(arg)
        elif opt == '-y':
            p2 = float(arg)
        elif opt == '-z':
            p3 = float(arg)
        elif opt == '-r':
            revision_param = int(round(float(arg)))
        elif opt == '-l':
            local_search_param = float(arg)
        elif opt == '-t':
            BRANCH_TIME = float(arg)
        elif opt == '-g':
            gurobi_time = float(arg)

    main()

# python codes/main.py -p Instances/r101.txt -Q 10 -a 0.05 -f 1000 -e 2500 -s 20 -n 0.5 -x 0.4 -y 0.2 -z 0.4 -r 1500 -l 0.8 -t 1 -g 30

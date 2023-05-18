cimport numpy as np
import numpy as np
import cython
from random import randint, sample, random, choice

cpdef object perturbation(object s, double theta1, double theta2):
    cdef double x = random()
    if x <= theta1:
        return all_to_root(s)
    elif x <= theta2:
        return branch_to_root(s) 
    else:
        return branch_to_branch(s)

cdef object find_branch(object s, int j):
    cdef list nodes = [j]
    cdef list to_root = [0]
    cdef list visited = [False] * len(s.parent)
    cdef int load

    visited[0] = True
    visited[j] = True

    for j,k in enumerate(s.parent):
        if j != 0:
            if visited[j] == False:
                nodes, to_root, visited = aux_find_branch(s,j,nodes, to_root,visited)
    load = len(nodes)
    return nodes, load

cdef object aux_find_branch(object s, int j, list nodes, list to_root, list visited):
    cdef int k = s.parent[j]

    if not visited[k]:
        nodes, to_root, visited = aux_find_branch(s,k,nodes, to_root,visited)
    
    if k in to_root:
        to_root.append(j)
    else:
        nodes.append(j)
    visited[j] = True
    return nodes, to_root, visited

cdef object branch_to_root(object s):
    cdef int n = len(s.parent)
    cdef int j  = randint(1,n-1)
    cdef list nodes
    cdef int i,k
    cdef int load
    
    nodes, load = find_branch(s,j)

    s.load[s.gate[j]] -= load
    s.parent[j] = 0
    s.gate[j] = j
    s.load[s.gate[j]] += load
    for i in nodes:
        s.gate[i] = j
    return s

cdef object all_to_root(object s):
    cdef int n = len(s.parent)
    cdef int j  = randint(1,n-1)
    cdef int i
    cdef int load
    cdef list nodes

    nodes, load = find_branch(s,j)
    s.load[s.gate[j]] -= load
    
    for i in nodes: #this should avoid using j
        s.connect(0,j)
    return s
    
cdef object branch_to_branch(object s):
    cdef int n = len(s.parent)
    cdef int j  = randint(1,n-1)
    cdef int i,k

    nodes, load = find_branch(s,j)

    for i in sample(range(1,n), n-1):
        if i not in nodes:
            if s.load[s.gate[i]] + load <= s.capacity:
                s.load[s.gate[j]] -= load
                s.parent[j] = i
                if s.gate[i] == 0:
                    s.gate[j] = gate = j
                else:
                    s.gate[j] = gate = s.gate[i]

                for i in nodes: #this should avoid using j
                    s.gate[i] = s.gate[j]
                return s
    else:
        s.load[s.gate[j]] -= load
        s.parent[j] = 0
        s.gate[j] = j
        s.load[s.gate[j]] += load
        for i in nodes:
            s.gate[i] = j 
        return s

cdef object branch_to_branch_2(object s):
    cdef int n = len(s.parent)
    cdef int j  = randint(1,n-1)
    cdef int i, ii,k
    cdef int load
    cdef list nodes
    nodes, load = find_branch(s,j)

    s.parent[j] = -1
    s.load[s.gate[j]] -= load
    j = choice(nodes)

    for i in sample(range(1,n), n-1):
        if i not in nodes:
            if s.load[s.gate[i]] + load <= s.capacity:    
                ii = i  
                while True:
                    p = s.parent[j]
                    s.parent[j] = i
                    if s.gate[i] == 0: s.gate[j] = gate = j
                    else: s.gate[j] = gate = s.gate[i]

                    if p == -1:
                        break
                    else:
                        i = j
                        j = p 

                for i in nodes:
                    s.gate[i] == s.gate[ii]
                s.load[s.gate[ii]] += load
                return s
    else:
        ii = i = 0
        while True:
            p = s.parent[j]
            s.parent[j] = i
            if s.gate[i] == 0: s.gate[j] = gate = j
            else: s.gate[j] = gate = s.gate[i]
                
            if p == -1:
                break
            else:
                i = j
                j = p 

            for i in nodes:
                s.gate[i] == s.gate[ii]
            s.load[s.gate[ii]] += load
        return s

cpdef object best_father(object s, int times, np.ndarray latest, double penalization): #perturbation 1
    cdef int n = len(s.parent)
    cdef list tries = sample(range(1,n), times)
    cdef int i,j,k,load
    cdef list connected
    cdef double cost, minim
    cdef bint feasible

    for j in tries:

        # discover which nodes are part of the branch
        nodes, load = find_branch(s, j)
        connected =  [i for i in range(n) if i not in nodes]

        s.load[s.gate[j]] -= load # descontar lo que està de sobra
        minim = np.inf
        for i in connected:
            if s.load[s.gate[i]] + load <= s.capacity and s.distance[i,j] < minim:
                minim = s.distance[i,j]
                k = i

        s.parent[j] = k
        for i in nodes:
            s.gate[i] = s.gate[j]
        s.load[s.gate[j]] += load

    return s

cpdef object fitness_old(object s, np.ndarray latest, double penalization):
    cdef double cost = 0
    cdef bint feasible = True
    cdef int j,k
    for j,k in enumerate(s.parent):
        if j != 0:
            cost += s.distance[k,j]
            arr = s.arrival[j]
            lat = latest[j]
            if arr > lat:
                feasible = False
                cost += (arr - lat) * penalization
    return cost, feasible

cpdef object fitness(object s, np.ndarray earliest, np.ndarray latest, double penalization):
    cdef double cost = 0
    cdef bint feasible = True
    cdef list visited = [False] * len(s.parent)
    cdef int j,k

    visited[0] = True
    for j,k in enumerate(s.parent):
        if j != 0:
            if visited[j] == False:
                aux, visited = fitness_aux(s, j, earliest, visited)
                cost += aux
            if s.arrival[j] > latest[j]:
                feasible = False
                cost += (s.arrival[j] - latest[j]) * penalization
    return cost, feasible

cdef object fitness_aux(object s, int j, np.ndarray earliest, list visited):
    cdef int k = s.parent[j]
    cdef double cost

    if visited[k] == False:
        cost, visited = fitness_aux(s,k,earliest ,visited)
    else: 
        cost = 0

    s.arrival[j] = s.arrival[k] + s.distance[k,j]
    if not s.arrival[j] >= earliest[j]:
        s.arrival[j] = earliest[j]
    visited[j] = True
    return cost + s.distance[k,j] , visited

    






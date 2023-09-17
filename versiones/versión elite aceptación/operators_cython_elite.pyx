cimport numpy as np
import numpy as np
import cython
from random import randint, sample, random, choice

cpdef object perturbation(object s, double theta1, double theta2, double theta3):
    cdef double x = random()
    if x <= theta1:
        return all_to_root(s)
    elif x <= theta2:
        return branch_to_root(s) 
    elif x <= theta3:
        return branch_to_branch(s)
    else:
        return branch_to_branch_enchanced(s)

cdef object branch_to_root(object s):
    cdef int n = len(s.parent)
    cdef int j  = randint(1,n-1)
    cdef list nodes = [j]
    cdef list nodes_arrival = [s.arrival[j]]
    cdef int i,k
    cdef int lo = 1
    
    for i in np.argsort(s.arrival, kind = "stable"):
        if s.parent[i] in nodes:
            nodes.append(i)
            nodes_arrival.append(s.arrival[i])
            lo += 1

    s.load[s.gate[j]] -= lo
    s.parent[j] = 0
    for i in np.argsort(nodes_arrival, kind = "stable"): #this should avoid using j
        j = nodes[i]
        k = s.parent[j]
        s.connect(k,j)
    return s

cdef object all_to_root(object s):
    cdef int n = len(s.parent)
    cdef int j  = randint(1,n-1)
    cdef list nodes = [j]
    cdef list nodes_arrival = [s.arrival[j]]
    cdef int i
    cdef int lo = 1

    for i in np.argsort(s.arrival, kind = "stable"):
        if s.parent[i] in nodes:
            nodes.append(i)
            nodes_arrival.append(s.arrival[i])
            lo += 1

    s.load[s.gate[j]] -= lo
    
    for i in np.argsort(nodes_arrival, kind = "stable"): #this should avoid using j
        j = nodes[i]
        s.connect(0,j)
    return s
    
cdef object branch_to_branch(object s):
    cdef int n = len(s.parent)
    cdef int j  = randint(1,n-1)
    cdef list nodes = [j]
    cdef list nodes_arrival = [s.arrival[j]]
    cdef int i,k
    cdef int lo = 1

    for i in np.argsort(s.arrival, kind = "stable"):
        if s.parent[i] in nodes:
            nodes.append(i)
            nodes_arrival.append(s.arrival[i])
            lo += 1

    for i in sample(range(1,n), n-1):
        if i not in nodes:
            if s.load[s.gate[i]] + lo <= s.capacity:
                s.load[s.gate[j]] -= lo
                s.parent[j] = i
                for i in np.argsort(nodes_arrival, kind = "stable"): #this should avoid using j
                    j = nodes[i]
                    k = s.parent[j]
                    s.connect(k,j)
                return s
    else:
        s.load[s.gate[j]] -= lo
        s.parent[j] = 0
        for i in np.argsort(nodes_arrival, kind = "stable"): #this should avoid using j
            j = nodes[i]
            k = s.parent[j]
            s.connect(k,j)
        return s

cdef object branch_to_branch_enchanced(object s):
    cdef int n = len(s.parent)
    cdef int j  = randint(1,n-1)
    cdef list nodes = [j]
    cdef list nodes_arrival = [s.arrival[j]]
    cdef dict nodes_position = {j:0}
    cdef int i,k
    cdef int lo = 1
    cdef int count = 1 

    for i in np.argsort(s.arrival, kind = "stable"):
        if s.parent[i] in nodes:
            nodes.append(i)
            nodes_arrival.append(s.arrival[i])
            nodes_position[i] = count
            count += 1
            lo += 1

    s.parent[j] = -1
    s.load[s.gate[j]] -= lo
    j = choice(nodes)

    for i in sample(range(1,n), n-1):
        if i not in nodes:
            if s.load[s.gate[i]] + lo <= s.capacity:      
                while True:
                    p = s.parent[j]
                    s.connect(i,j)
                    nodes_arrival[nodes_position[j]] = s.arrival[j]
                    s.load[s.gate[j]] -= 1
                    if p == -1:
                        break
                    else:
                        i = j
                        j = p 

                for i in np.argsort(nodes_arrival, kind = "stable"): #this should avoid using j
                    j = nodes[i]
                    k = s.parent[j]
                    s.connect(k,j)
                return s
    else:
        i = 0
        while True:
            p = s.parent[j]
            s.connect(i,j)
            s.connect(i,j)
            nodes_arrival[nodes_position[j]] = s.arrival[j]
            s.load[s.gate[j]] -= 1
            if p == -1:
                break
            else:
                i = j
                j = p 
        for i in np.argsort(nodes_arrival, kind = "stable"): #this should avoid using j
            j = nodes[i]
            k = s.parent[j]
            s.connect(k,j)
        return s

cpdef object best_father(object s, int times, np.ndarray latest, double penalization): #perturbation 1
    cdef int n = len(s.parent)
    cdef list tries = sample(range(1,n), times)
    cdef int i,j,k, lo
    cdef list nodes, nodes_arrival, con
    cdef double cost, minim
    cdef bint feasible

    for j in tries:
        nodes = [j]
        nodes_arrival  = [s.arrival[j]]
        lo = 1

        # discover which nodes are part of the branch
        for i in np.argsort(s.arrival, kind = "stable"):
            if s.parent[i] in nodes:
                nodes.append(i)
                nodes_arrival.append(s.arrival[i])
                lo += 1

        connected =  [i for i in range(n) if i not in nodes]

        lo = len(nodes)
        s.load[s.gate[j]] -= lo # descontar lo que està de sobra
        minim = np.inf
        for i in connected:
            if s.load[s.gate[i]] + lo <= s.capacity and s.distance[i,j] < minim:
                minim = s.distance[i,j]
                k = i

        s.parent[j] = k
        for i in np.argsort(nodes_arrival, kind = "stable"): #this should avoid using j
            j = nodes[i]
            k = s.parent[j]
            s.connect(k,j)

    cost, feasible = fitness(s, latest, penalization)

    return s, cost, feasible

cpdef object fitness(object s, np.ndarray latest, double penalization):
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





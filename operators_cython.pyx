cimport numpy as np
import numpy as np
import cython
from random import randint, sample, random

cpdef  object perturbation(object s, double theta):
    if random() <= theta:
        return branch_to_root(s) 
    else:
        return branch_to_branch(s)

cdef object branch_to_root(object s):
    cdef int n = len(s.parent)
    cdef int j  = randint(1,n-1)
    cdef list parents = [j]
    cdef list nodes = [j]
    cdef list nodes_arrival = [s.arrival[j]]
    cdef int i,k
    
    cdef int old_gate = s.gate[j]
    s.connect(0,j)

    for i in np.argsort(s.arrival, kind = "stable"):
        if s.parent[i] in parents:
            parents.append(i)
            nodes.append(i)
            nodes_arrival.append(s.arrival[i])

    cdef int lo = len(nodes)
    s.load[old_gate] -= lo
    
    for i in np.argsort(nodes_arrival, kind = "stable"): #this should avoid using j
        if i != 0:
            j = nodes[i]
            k = s.parent[j]
            s.connect(k,j)
    return s
    
cdef object branch_to_branch(object s):
    cdef int n = len(s.parent)
    cdef int j  = randint(1,n-1)
    cdef list parents = [j]
    cdef list nodes = [j]
    cdef list nodes_arrival = [s.arrival[j]]
    cdef int i,k

    for i in np.argsort(s.arrival, kind = "stable"):
        if s.parent[i] in parents:
            parents.append(i)
            nodes.append(i)
            nodes_arrival.append(s.arrival[i])

    cdef int lo = len(nodes)
    s.load[s.gate[j]] -= lo

    for i in sample(range(1,n), n-1):
        if i not in nodes:
            if s.load[s.gate[i]] + lo <= s.capacity:
                s.parent[j] = i
                for i in np.argsort(nodes_arrival, kind = "stable"): #this should avoid using j
                    j = nodes[i]
                    k = s.parent[j]
                    s.connect(k,j)
                return s
    else:
        s.parent[j] = 0
        for i in np.argsort(nodes_arrival, kind = "stable"): #this should avoid using j
            j = nodes[i]
            k = s.parent[j]
            s.connect(k,j)
        return s

cpdef object best_father(object s, int times, np.ndarray latest, double penalization): #perturbation 1
    cdef int n = len(s.parent)
    cdef list tries = sample(range(1,n), times)
    cdef int i,j,k, lo
    cdef list parents, nodes, nodes_arrival, con
    cdef double cost, minim
    cdef bint feasible

    for j in tries:
        parents = [j]
        nodes = [j]
        nodes_arrival  = [s.arrival[j]]

        # discover which nodes are part of the branch
        for i in np.argsort(s.arrival, kind = "stable"):
            if s.parent[i] in parents:
                parents.append(i)
                nodes.append(i)
                nodes_arrival.append(s.arrival[i])

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






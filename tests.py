from matheuristic import instance, read_instance  #, prim
from matheuristic import prim as prim_python
from operators_cython import best_father as btr_cython
from operators_cython import create_tree
from matheuristic import best_father as btr_python
from time import perf_counter
from numpy import inf


name, capacity, node_data = read_instance("instances/rc102.txt")
ins = instance(name, capacity, node_data, 100)
ins.capacity = 15

global xcoords, ycoords, latest, earliest, D, Q
xcoords, ycoords = ins.xcoords, ins.ycoords
earliest, latest = ins.earliest, ins.latest
D = ins.cost
Q = ins.capacity
PENALIZATION = 9.378
create_tree(100, Q, D, earliest, latest, PENALIZATION)



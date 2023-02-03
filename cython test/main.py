from factorial_cython import fact as fact_cython2
from time import perf_counter
import sys
sys.setrecursionlimit(10 ** 8)

n = 1000000
inicio = perf_counter()
f = fact_cython2(n)
print(perf_counter() - inicio)

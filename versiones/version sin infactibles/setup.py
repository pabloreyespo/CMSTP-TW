from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("operators_cython_unfeasible.pyx"),
    include_dirs=[numpy.get_include()]
)
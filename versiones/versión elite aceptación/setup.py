from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("operators_cython_elite.pyx"),
    include_dirs=[numpy.get_include()]
)
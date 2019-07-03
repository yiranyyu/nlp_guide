from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(
    ['word2vec.pyx'], annotate=True
))

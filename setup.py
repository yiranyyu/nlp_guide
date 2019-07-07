from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(
    ['word2vec_inner.pyx'], annotate=False, compiler_directives={'language_level' : "3"}
))

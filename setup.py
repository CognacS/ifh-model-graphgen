from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy


"""setup(
    name='graph-generation',
    ext_modules=cythonize(
        "src/datatypes/graphops/graphops.pyx",
        annotate=True,
        build_dir="src/datatypes/graphops/"
    ),
    zip_safe=False,
    include_dirs=[numpy.get_include()]
)"""

ext_modules=[
    Extension("src.datatypes.graphops.graphops_c",    # location of the resulting compiled module
             ["src/datatypes/graphops/graphops_c.pyx"],) ]


setup(
    name='graph-generation',
    packages=find_packages(),
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    zip_safe=False,
    include_dirs=[numpy.get_include()]
)
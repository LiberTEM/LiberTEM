from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension(
        "test",
        # ["bm.pyx", "bm3.cpp"],
        ["bm.pyx"],
        extra_objects=["bm4.o"],
        language='c++',
        extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name='test',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)

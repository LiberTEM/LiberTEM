from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension(
        "u12mod",
        ["u12mod.pyx"],
        extra_objects=["u12decode.o"],
        language='c++',
        extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name='u12mod',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)

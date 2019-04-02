from os.path import join
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

easyjit_path = '/home/alex/source/easy-just-in-time/'

ext_modules = [
    Extension(
        "maskjit",
        ["bm.pyx"],
        extra_objects=["bm2.o"],
        language='c++',
        extra_compile_args=["-O3", "-march=native"],
        extra_link_args=['-lEasyJitRuntime', '-L%s' % join(easyjit_path, 'build', 'bin')],
    )
]

setup(
    name='maskjit',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)

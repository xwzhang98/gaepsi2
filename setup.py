from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

def create_extension(name, sources):
    """Create a Cython extension with optimized compile flags."""
    return Extension(
        name, 
        sources,
        include_dirs=["./", numpy.get_include()],
        extra_compile_args=['-O3'],
        extra_link_args=['-O3']
    )

extensions = [
    create_extension("gaepsi2.svr", ["gaepsi2/svr.pyx"]),
    create_extension("gaepsi2._painter", ["gaepsi2/_painter.pyx"])
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': 3}),
    zip_safe=False,
)


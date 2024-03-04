from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

examples_extension = Extension(
    name="py2cpp",
    sources=["bridge.pyx"],
    language="c++"

)
setup(
    name="py2cpp",
    ext_modules=cythonize([examples_extension])
)
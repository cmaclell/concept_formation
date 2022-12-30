from glob import glob
from setuptools import setup

from pybind11.setup_helpers import intree_extensions
from pybind11.setup_helpers import build_ext

ext_modules = intree_extensions(glob('concept_formation/*.cpp'))

setup(
    setup_requires=['pbr'],
    pbr=True,
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)

from glob import glob
from setuptools import setup

from pybind11.setup_helpers import intree_extensions
from pybind11.setup_helpers import build_ext

ext_modules = intree_extensions(glob('concept_formation/*.cpp'))

# Specify the C++ standard for each extension module
# for module in ext_modules:
#     module.cxx_std = '2a'
#     module.extra_link_args.append("-ltbb")

setup(
    name="concept_formation",
    author="Christopher J. MacLellan, Erik Harpstead",
    author_email="maclellan.christopher@gmail.com, whitill29@gmail.com",
    url="https://github.com/cmaclell/concept_formation",
    description="A library for doing incremental concept formation using"
                "algorithms in the COBWEB family.",
    long_description=open('README.rst').read(),
    description_content_type="text/x-rst; charset=UTF-8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: PyPy"
    ],
    keywords="clustering,machine-learning",
    license="MIT",
    license_file="LICENSE.txt",
    packages=["concept_formation"],
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)

import setuptools

setuptools.setup(
    name='concept_formation',
    version='0.3.3',
    author='Christopher J. MacLellan, Erik Harpstead',
    author_email='maclellan.christopher@gmail.com, whitill29@gmail.com',
    packages=setuptools.find_packages(),
    include_package_data=True,
    url='https://pypi.python.org/pypi/concept_formation/',
    license='MIT License',
    description=('A library for doing incremental concept formation '
                 'using algorithms in the COBWEB family.'),
    long_description=open('README.rst').read(),
    install_requires=['py_search>=2.0.0', 'munkres>=1.0.12'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'matplotlib', 'numpy', 'scikit-learn', 'scipy'],
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: Implementation :: PyPy'],
)

import setuptools

setuptools.setup(
    name='concept_formation',
    version='0.3.0',
    author='Christopher J. MacLellan, Erik Harpstead',
    author_email='maclellan.christopher@gmail.com, whitill29@gmail.com',
    packages=setuptools.find_packages(),
    include_package_data = True,
    url='http://pypi.python.org/pypi/concept_formation/',
    license='LICENSE.txt',
    description='A library for doing incremental concept formation using algorithms in the COBWEB family.',
    long_description=open('README.rst').read(),
    install_requires=['py_search>=1.0.2', 'munkres>=1.0.8'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest','matplotlib','numpy','scikit-learn','scipy'],
)

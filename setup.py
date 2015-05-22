from distutils.core import setup

setup(
    name='concept_formation',
    version='0.1.3',
    author='Christopher J. MacLellan, Erik Harpstead',
    author_email='maclellan.christopher@gmail.com, whitill29@gmail.com',
    packages=['concept_formation', 
              'concept_formation.test',
              'concept_formation.examples'],
    url='http://pypi.python.org/pypi/concept_formation/',
    license='LICENSE.txt',
    description='A library for doing incremental concept formation using algorithms in the COBWEB family',
    long_description=open('README.rst').read(),
    install_requires=[],
    include_package_data = True,
)

from distutils.core import setup

setup(
    name='concept_formation',
    version='0.1.0',
    author='Christopher J. MacLellan, Erik Harpstead',
    author_email='maclellan.christopher@gmail.com, whitill29@gmail.com',
    packages=['concept_formation', 'concept_formation.test'],
    scripts=['bin/cobweb3_cluster_iris.py',
             'bin/cobweb3_cluster_simulated.py',
             'bin/cobweb3_predict_iris.py',
             'bin/cobweb_cluster_mushroom.py',
             'bin/cobweb_predict_mushroom.py',
             'bin/cobweb_predict_mushroom.py',
             'bin/trestle_cluster_rumbleblocks.py',
             'bin/trestle_predict_rumbleblocks.py'],
    url='http://pypi.python.org/pypi/concept_formation/',
    license='LICENSE.txt',
    description='A library for doing incremental concept formation using algorithms in the COBWEB family',
    long_description=open('README.md').read(),
    install_requires=[],
)

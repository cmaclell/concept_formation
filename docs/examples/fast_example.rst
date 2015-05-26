Fast Example
============

.. ipython::

    In [1]: from pprint import pprint

    In [2]: from concept_formation.trestle import TrestleTree

    In [3]: from concept_formation.cluster import cluster

    # Data is stored in a list of dictionaries where values can be either nominal,
    # numeric, component, or relational.
    In [4]: data = [{'f1': 'v1', #nominal value
       ...:          'f2': 2.6, #numeric value
       ...:          'f3': {'sub-feature1': 'v1'}, # component value
       ...:          'f4': {'sub-feature1': 'v1'}, # component value
       ...:          'f5': ['some-relation', 'f3', 'f4'] #relational value
       ...:         },
       ...:         {'f1': 'v1', #nominal value
       ...:          'f2': 2.8, #numeric value
       ...:          'f3': {'sub-feature1': 'v2'}, # component value
       ...:          'f4': {'sub-feature1': 'v1'}, # component value
       ...:          'f5': ['some-relation', 'f3', 'f4'] #relational value
       ...:         }]

    # Data can be clustered with a TrestleTree, which supports all data types or
    # with a specific tree (CobwebTree or Cobweb3Tree) that supports subsets of
    # datatypes (CobwebTree supports only Nominal and Cobweb3Tree supports only
    # nominal or numeric).
    In [5]: tree = TrestleTree()

    In [6]: tree.fit(data)

    # Trees can be printed in plaintext or exported in JSON format
    In [7]: print(tree)

    In [8]: pprint(tree.root.output_json())

    # Trees can also be used to predict missing attributes of new data points.
    In [9]: new = {'f2': 2.6, 'f3': {'sub-feature1': 'v1'}, 'f4': {'sub-feature1': 'v1'},
      ....:        'f5': ['some-relation', 'f3', 'f4']}

    In [10]: concept = tree.categorize(new)

    In [11]: print(concept.predict('f1'))

    # Or to get the probability of a particular attribute value
    In [12]: print(concept.get_probability('f1', 'v1'))

    # Trees can also be used to produce flat clusterings
    In [13]: new_tree = TrestleTree()

    In [14]: clustering = cluster(new_tree, data, minsplit=1, maxsplit=1)

    In [15]: print(clustering)


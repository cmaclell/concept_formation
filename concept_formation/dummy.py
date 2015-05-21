from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.trestle import TrestleTree
from concept_formation.structure_mapper import flattenJSON
from concept_formation.structure_mapper import structure_map

class DummyTree(TrestleTree):

    def __init__(self):
        self.root = Cobweb3Node()
        self.root.root = self.root
        self.root.alpha = 0
        self.root.scaling = False

    def ifit(self, instance, do_mapping=False):
        """
        Just maintain a set of counts at the root and use these for prediction.

        The structure_map parameter determines whether or not to do structure
        mapping. This is disabled by default to get a really naive model.
        """
        if do_mapping:
            temp_instance = structure_map(self.root, instance)
        else:
            temp_instance = flattenJSON(instance)
        self.root.increment_counts(temp_instance)
        return self.root

    def categorize(self, instance):
        return self.root



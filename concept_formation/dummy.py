from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.trestle import TrestleTree
from concept_formation.structure_mapper import StructureMapper
from concept_formation.structure_mapper import Tuplizer
from concept_formation.structure_mapper import ListProcessor
from concept_formation.structure_mapper import NameStandardizer
from concept_formation.structure_mapper import SubComponentProcessor
from concept_formation.structure_mapper import Flattener
from concept_formation.structure_mapper import Pipeline

class DummyTree(TrestleTree):

    def __init__(self):
        self.root = Cobweb3Node()
        self.root.tree = self
        self.alpha = 0
        self.scaling = False

    def ifit(self, instance, do_mapping=False):
        """
        Just maintain a set of counts at the root and use these for prediction.

        The structure_map parameter determines whether or not to do structure
        mapping. This is disabled by default to get a really naive model.
        """
        if do_mapping:
            structure_mapper = StructureMapper(self.root)
            temp_instance = structure_mapper.transform(instance)
        else:
            pipeline = Pipeline(Tuplizer(), ListProcessor(),
                                 NameStandardizer(),
                                 SubComponentProcessor(), Flattener())
            temp_instance = pipeline.transform(instance)
        self.root.increment_counts(temp_instance)
        return self.root

    def categorize(self, instance):
        return self.root



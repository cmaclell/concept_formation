from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.trestle import TrestleTree
from concept_formation.structure_mapper import StructureMapper
from concept_formation.preprocessor import Tuplizer
from concept_formation.preprocessor import ListProcessor
from concept_formation.preprocessor import NameStandardizer
from concept_formation.preprocessor import SubComponentProcessor
from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Pipeline

class DummyTree(TrestleTree):
    """
    The DummyTree is designed to serve as a naive baseline to compare
    :class:`Trestle <concept_formation.trestle.TrestleTree>` to. The DummyTree
    can perform :meth:`structure mapping
    <concept_formation.structure_mapper.StructureMapper.transform>` but in all other
    respects it is effectively a tree that consists of only a root.
    """

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

        **This process modifies the tree's knoweldge.** For a non-modifying version
        see: :meth:`DummyTree.categorize`.

        :param instance: an instance to be categorized into the tree.
        :type instance: {a1:v1, a2:v2, ...}
        :param do_mapping: a flag for whether or not to do structure mapping.
        :type do_mapping: bool
        :return: the root node of the tree containing everything ever added to it.
        :rtype: Cobweb3Node
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
        """
        Return the root of the tree. Because the DummyTree contains only 1 node
        then it will always categorize instances to that node.

        **This process does not modify the tree's knoweldge.** For a modifying version
        see: :meth:`DummyTree.ifit`.
        
        :param instance: an instance to be categorized into the tree.
        :type instance: {a1:v1, a2:v2, ...}
        :return: the root node of the tree containing everything ever added to it.
        :rtype: Cobweb3Node
        """
        return self.root



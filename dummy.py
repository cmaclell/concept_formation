from cobweb3 import Cobweb3Node
from trestle import TrestleTree
#from structure_mapper import structure_map
from structure_mapper import flattenJSON

class DummyTree(TrestleTree):

    def __init__(self):
        self.root = Cobweb3Node()
        self.root.root = self.root
        self.root.alpha = 0
        self.root.scaling = False

    def ifit(self, instance, structure_map=False):
        """
        Just maintain a set of counts at the root and use these for prediction.

        The structure_map parameter determines whether or not to do structure
        mapping. This is disabled by default to get a really naive model.
        """
        if structure_map:
            temp_instance = structure_map(self.root, instance)
        else:
            temp_instance = flattenJSON(instance)
        self.root.increment_counts(temp_instance)
        return self.root

    def categorize(self, instance):
        return self.root



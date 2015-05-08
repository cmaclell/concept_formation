from cobweb3 import Cobweb3Node
from trestle import TrestleTree
from structure_mapper import structure_map

class DummyTree(TrestleTree):

    def __init__(self):
        self.root = Cobweb3Node()
        self.root.root = self.root
        self.root.alpha = 0
        self.root.scaling = False

    def ifit(self, instance):
        temp_instance = structure_map(self.root, instance)
        self.root.increment_counts(temp_instance)
        return self.root

    def categorize(self, instance):
        return self.root



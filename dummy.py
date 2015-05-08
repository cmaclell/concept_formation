from cobweb3 import Cobweb3Node
from trestle import TrestleTree

class DummyTree(TrestleTree):

    def __init__(self):
        self.root = Cobweb3Node()
        self.root.root = self.root

    def ifit(self, instance):
        temp_instance = self.structure_map(instance)
        self.root.increment_counts(temp_instance)
        return self.root

    def categorize(self, instance):
        return self.root



from cobweb3 import Cobweb3Tree
from cobweb3 import Cobweb3Node
from structure_mapper import structure_map

class TrestleTree(Cobweb3Tree):

    def __init__(self, alpha=0.001, scaling=False):
        self.root = Cobweb3Node()
        self.root.root = self.root
        self.root.alpha = alpha
        self.root.scaling = scaling

    def ifit(self, instance):
        """
        A modification of ifit to call Trestle instead.
        """
        return self.trestle(instance)

    def trestle_categorize(self, instance):
        """
        The Trestle categorize function, this Trestle categorizes all the
        sub-components before categorizing itself.
        """
        temp_instance = structure_map(self.root, instance)
        return self.cobweb_categorize(temp_instance)

    def categorize(self, instance):
        """
        A categorize function that can be used polymorphicaly without 
        having to worry about the type of the underlying object.

        In Trestle's case this calls trestle_categorize()
        """
        return self.trestle_categorize(instance)

    def trestle(self, instance):
        """
        Recursively calls Trestle on all of the components in a depth-first
        traversal. Once all of the components have been classified then then it
        classifies the current node.
        """
        temp_instance = structure_map(self.root, instance)
        return self.cobweb(temp_instance)

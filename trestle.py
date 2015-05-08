import json
from cobweb3 import Cobweb3Tree
from cobweb3 import Cobweb3Node
from structure_mapper import flatMatch
from structure_mapper import renameFlat
from structure_mapper import flattenJSON

class TrestleTree(Cobweb3Tree):

    def __init__(self):
        self.root = Cobweb3Node()
        self.root.root = self.root

    def ifit(self, instance):
        """
        A modification of ifit to call Trestle instead.
        """
        return self.trestle(instance)

    def structure_map(self, instance):
        """
        Flatten the instance, perform structure mapping, rename the instance
        based on this structure mapping, and return the renamed instance.
        """
        temp_instance = flattenJSON(instance)
        mapping = flatMatch(self.root, temp_instance)
        temp_instance = renameFlat(temp_instance, mapping)
        return temp_instance

    def trestle_categorize(self, instance):
        """
        The Trestle categorize function, this Trestle categorizes all the
        sub-components before categorizing itself.
        """
        #temp_instance = flattenJSON(instance)
        #mapping = flatMatch(self.root, temp_instance)
        #temp_instance = renameFlat(temp_instance, mapping)
        temp_instance = self.structure_map(instance)
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
        #temp_instance = flattenJSON(instance)
        #mapping = flatMatch(self.root, temp_instance)
        #temp_instance = renameFlat(temp_instance, mapping)
        temp_instance = self.structure_map(instance)
        return self.cobweb(temp_instance)

if __name__ == "__main__":

    tree = TrestleTree()
    
    with open('data_files/rb_s_13_noCheck_continuous.json', "r") as json_data:
        instances = json.load(json_data)
    # print(len(instances))
    #instances = instances[0:20]
    print([set(x) for x in tree.cluster(instances, maxsplit=3)])

    #labels = tree.kc_label("data_files/instant-test-processed.json", 16000)
    #pickle.dump(labels, open('clustering.pickle', 'wb'))




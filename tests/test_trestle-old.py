from trestle import Trestle
import unittest
import random

class TestCobweb(unittest.TestCase):

    def setUp(self):
        self.tree = Trestle()

    def test_cu_missing_av(self):

        instance = {'a1' : 'v1',
                    'a2' : 'v2',
                    'a3' : 'v3',
                    'a4' : 'v3',
                    'aX' : 'v4'}
        self.tree.ifit(instance)

        instance = {'a5' : 'v1',
                    'a6' : 'v2',
                    'a7' : 'v3',
                    'a8' : 'v3',
                    'aY' : 'v4'}
        self.tree.ifit(instance)

        instance = {'a1' : 'v1',
                    'a2' : 'v2',
                    'a3' : 'v3',
                    'a4' : 'v3',
                    'aX' : 'v4'}
        self.tree.ifit(instance)

        instance = {'a7' : 'v1'}
        node = self.tree.ifit(instance)
        
        assert node.parent.parent == None

    def test_hungarian_match(self):
        instance = {'c1': {'a1': "x", 'a2': 'x'},
                    'c2': {'a1': "y", 'a2': 'y'}}
        self.tree.ifit(instance)

        instance2 = {'c2': self.tree.trestle_categorize({'a1': "x"}),
                    'c1': self.tree.trestle_categorize({'a1': "y"})}

        #TODO add more
        self.assertTrue(self.tree.hungarian_match(instance2)['c1'] == 'c2')
        self.assertTrue(self.tree.hungarian_match(instance2)['c2'] == 'c1')

    def test_values(self):
        #self.tree.train_from_json('data_files/labyrinth_test.json')
        self.tree.train_from_json('data_files/instant-test-processed2.json',50)

        # ensure all parent pointers are correct
        self.tree.verify_parent_pointers()

        # ensure all counts are correct
        self.tree.verify_counts()

        # ensure no values are parents of others (double counting)
        self.tree.val_representation_check()

        # ensure all values are leaves
        self.tree.val_leaves_check()

        ## ensure component values are values still in the tree
        self.tree.val_existance_check()

    #def test_attribute_generalize(self):
    #    root = Trestle()

    #    c3 = Trestle()
    #    c3.parent = root
    #    root.children.append(c3)

    #    c1 = Trestle()
    #    c2 = Trestle()
    #    c1.parent = c3
    #    c2.parent = c3
    #    c3.children.append(c1)
    #    c3.children.append(c2)

    #    parent = Trestle()
    #    parent.parent = root
    #    root.children.append(parent)

    #    parent.count = 3
    #    parent.av_counts["a1"] = {}
    #    parent.av_counts["a1"][c1] = 1.0
    #    parent.av_counts["a1"][c2] = 1.0
    #    filler = Trestle()
    #    filler.parent = root
    #    root.children.append(filler)
    #    parent.av_counts["a1"][filler] = 1.0

    #    child1 = Trestle()
    #    child1.count = 2
    #    child1.parent = parent
    #    parent.children.append(child1)
    #    child1.av_counts["a1"] = {}
    #    child1.av_counts["a1"][c1] = 1.0
    #    child1.av_counts["a1"][c2] = 1.0

    #    child2 = Trestle()
    #    child2.count = 1
    #    child2.parent = parent
    #    parent.children.append(child2)
    #    child2.av_counts["a1"] = {}
    #    child2.av_counts["a1"][filler] = 1.0

    #    # ensure it doesn't generalize too far.
    #    temp, cu = parent.attribute_generalize()
    #    self.assertTrue(c3 in temp.av_counts["a1"])
    #    self.assertTrue(temp.av_counts["a1"][c3] == 2)

    #    del parent.av_counts["a1"][filler]
    #    del child2.av_counts["a1"][filler]

    #    # ensure it generalizes to the common ancestor
    #    temp, cu = parent.attribute_generalize()
    #    self.assertTrue(c3 in temp.av_counts["a1"])
    #    self.assertTrue(temp.av_counts["a1"][c3] == 2)

    #def test_labyrinth(self):

    #    # test backwards compatiblity
    #    self.tree.train_from_json('data_files/cobweb_test.json')
    #    self.tree.verify_counts()
    #    self.tree.train_from_json('data_files/cobweb3_test3.json')
    #    self.tree.verify_counts()
    #    self.tree.train_from_json('data_files/labyrinth_test.json')
    #    self.tree.verify_counts()

if __name__ == "__main__":
    unittest.main()




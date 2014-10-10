from labyrinth import Labyrinth
import unittest

def verify_component_values(lab,from_root=False):
    """
    A function to verify that all component values in the tree are leaves.
    """
    assert isinstance(lab,Labyrinth)
    if from_root :
        lab = lab.get_root()
    for attr in lab.av_counts:
        for val in lab.av_counts[attr]:
            if isinstance(val, Labyrinth):
                assert not val.children
    for c in self.children:
        verify_component_values(c,False)

def verify_parent_pointers(lab,from_root=False):
    """
    A function to verify the integrity of parent pointers throughout the tree.
    """
    assert isinstance(lab,Labyrinth)
    if from_root:
        lab = lab.get_root()
    for c in lab.children:
        assert c.parent == lab
        verify_parent_pointers(c,False)
   
   
def val_check(lab, from_root=False):
    """
    A function to verify that values in a Labyrinth attribute value table are
    not relatives of each other.
    """
    assert isinstance(lab,Labyrinth)
    if from_root:
        lab = lab.get_root()
    for attr in lab.av_counts:
        for val in lab.av_counts[attr]:
            if isinstance(val, Labyrinth):
                for val2 in lab.av_counts[attr]:
                    if isinstance(val2, Labyrinth):
                        if val == val2:
                            continue
                        assert not val.is_parent(val2)
                        assert not val2.is_parent(val)
    for c in self.children:
        val_check(c,False)

class TestCobweb(unittest.TestCase):

    def setUp(self):
        self.tree = Labyrinth()
    
    def test_hungarian_match(self):
        self.assertTrue(False)

    def test_missing_values(self):
        """
        Ensure there are no values that no longer exist.
        """
        self.assertTrue(False)

    def test_labyrinth(self):

        # test backwards compatiblity
        self.tree.train_from_json('data_files/cobweb_test.json')
        self.tree.verify_counts()
        self.tree.train_from_json('data_files/cobweb3_test3.json')
        self.tree.verify_counts()

        self.tree.train_from_json('data_files/labyrinth_test.json')
        #self.tree.verify_counts()

if __name__ == "__main__":
    unittest.main()



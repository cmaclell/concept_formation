from labyrinth import Labyrinth
import unittest

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



from cobweb3 import Cobweb3
import unittest

class TestCobweb(unittest.TestCase):

    def setUp(self):
        self.tree = Cobweb3()
    
    def test_cobweb(self):

        # test backwards compatiblity
        self.tree.train_from_json('data_files/cobweb_test.json')
        self.tree.verify_counts()

        # test with numeric values
        self.tree.train_from_json('data_files/cobweb3_test3.json')
        self.tree.verify_counts()

if __name__ == "__main__":
    unittest.main()


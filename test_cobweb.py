from cobweb import Cobweb
import unittest

class TestCobweb(unittest.TestCase):

    def setUp(self):
        self.tree = Cobweb()

    def test_mean(self):
        self.assertEqual(self.tree.mean([2,3]),2.5)
        self.assertEqual(self.tree.mean([2.0,3.0]),2.5)
        with self.assertRaises(ValueError):
            self.tree.mean([])

    def test_std(self):
        self.assertEqual(self.tree.std([10]), 0.0)
        self.assertEqual(self.tree.std([2,3]), 0.5)
        self.assertEqual(self.tree.std([2.0,3.0]), 0.5)
        with self.assertRaises(ValueError):
            self.tree.std([])

    def test_verify_counts(self):
       self.tree.train_from_json('data_files/cobweb_test.json')
       for attr in self.tree.av_counts:
           for value in self.tree.av_counts[attr]:
               self.tree.av_counts[attr][value] = 0.0
       with self.assertRaises(AssertionError):
           self.tree.verify_counts()
    
    def test_cobweb(self):
       self.tree.train_from_json('data_files/cobweb_test.json')
       self.tree.verify_counts()

if __name__ == "__main__":
    unittest.main()

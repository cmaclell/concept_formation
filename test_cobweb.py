from cobweb import Cobweb
import unittest
import timeit

def verify_category_utility(node):
    if node.children:
        assert node.category_utility() == node.category_utility_old()
        
        for child in node.children:
            verify_category_utility(child)

def verify_counts(node):
    """
    Checks the property that the counts of the children sum to the same
    count as the parent. This is/was useful when debugging. If you are
    doing some kind of matching at each step in the categorization (i.e.,
    renaming such as with Labyrinth) then this will start throwing errors.
    """
    if len(node.children) == 0:
        return 

    temp = {}
    temp_count = node.count
    for attr in node.av_counts:
        if attr not in temp:
            temp[attr] = {}
        for val in node.av_counts[attr]:
            temp[attr][val] = node.av_counts[attr][val]

    for child in node.children:
        temp_count -= child.count
        for attr in child.av_counts:
            assert attr in temp
            for val in child.av_counts[attr]:
                if val not in temp[attr]:
                    print(val.concept_name)
                    print(attr)
                    print(node)
                assert val in temp[attr]
                temp[attr][val] -= child.av_counts[attr][val]

    #if temp_count != 0:
    #    print(self.count)
    #    for child in self.children:
    #        print(child.count)
    assert temp_count == 0

    for attr in temp:
        for val in temp[attr]:
            #if temp[attr][val] != 0.0:
            #    print(self)

            assert temp[attr][val] == 0.0

    for child in node.children:
        verify_counts(child)

class TestCobweb(unittest.TestCase):

    def test_expected_correct_guess(self):
        node = Cobweb()
        node.count = 10
        node.av_counts['a1'] = {}
        node.av_counts['a1']['v1'] = 1 
        node.av_counts['a1']['v2'] = 3 
        node.av_counts['a1']['v3'] = 6 

        assert node.expected_correct_guesses() == ((1/10)**2 + (3/10)**2 +
                                                   (6/10)**2)

    def test_category_utility(self):

        ## Code for timing
        #print("Current CU Time: %0.3f" %
        #      min(timeit.Timer(node.category_utility).repeat(repeat=10,number=1000)))
        node = Cobweb()
        node.count = 10
        node.av_counts['a1'] = {}
        node.av_counts['a1']['v1'] = 1 
        node.av_counts['a1']['v2'] = 3 
        node.av_counts['a1']['v3'] = 6 

        child1 = Cobweb()
        child1.count = 6
        child1.av_counts['a1'] = {}
        child1.av_counts['a1']['v3'] = 6 

        child2 = Cobweb()
        child2.count = 4
        child2.av_counts['a1'] = {}
        child2.av_counts['a1']['v1'] = 1 
        child2.av_counts['a1']['v2'] = 3 

        node.children = [child1, child2]

        assert node.category_utility() == (( (6/10) * ((1**2) - ((1/10)**2 +
                                                                (3/10)**2 +
                                                                (6/10)**2)) +
                                           (4/10) * (((1/4)**2 + (3/4)**2) -
                                                     ((1/10)**2 + (3/10)**2 +
                                                      (6/10)**2))) / 2)

    def test_cobweb(self):
        tree = Cobweb()
        tree.train_from_json('data_files/cobweb_test.json')
        verify_counts(tree)

if __name__ == "__main__":
    unittest.main()

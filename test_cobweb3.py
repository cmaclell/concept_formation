from utils import ContinuousValue
from cobweb3 import Cobweb3
import unittest
import random
import math

def verify_counts(node):
    """
    Checks the property that the counts of the children sum to the same
    count as the parent. This is/was useful when debugging. This is modified
    from the test_cobweb.verify_counts to handle continuous values properly.
    """
    if len(node.children) == 0:
        return 

    temp = {}
    temp_count = node.count
    for attr in node.av_counts:
        if isinstance(node.av_counts[attr], ContinuousValue):
            temp[attr] = node.av_counts[attr].num
        else:
            if attr not in temp:
                temp[attr] = {}
            for val in node.av_counts[attr]:
                temp[attr][val] = node.av_counts[attr][val]

    for child in node.children:
        temp_count -= child.count
        for attr in child.av_counts:
            assert attr in temp
            if isinstance(child.av_counts[attr], ContinuousValue):
                temp[attr] -= child.av_counts[attr].num
            else:
                for val in child.av_counts[attr]:
                    if val not in temp[attr]:
                        print(val.concept_name)
                        print(attr)
                        print(node)
                    assert val in temp[attr]
                    temp[attr][val] -= child.av_counts[attr][val]

    assert temp_count == 0

    for attr in temp:
        if isinstance(temp[attr], int):
            assert temp[attr] == 0.0
        else:
            for val in temp[attr]:
                assert temp[attr][val] == 0.0

    for child in node.children:
        verify_counts(child)


class TestCobweb(unittest.TestCase):

    def test_cobweb(self):
        tree = Cobweb3()
        for i in range(40):
            data = {}
            data['a1'] = random.choice(['v1', 'v2', 'v3', 'v4'])
            data['a2'] = random.choice(['v1', 'v2', 'v3', 'v4'])
            tree.ifit(data)
        verify_counts(tree)

    def test_cobweb3(self):
        tree = Cobweb3()
        for i in range(40):
            data = {}
            data['x'] = random.normalvariate(0,4)
            data['y'] = random.normalvariate(0,4)
            tree.ifit(data)
        verify_counts(tree)

    def test_expected_correct_guess(self):
        node = Cobweb3()
        node.count = 10
        node.av_counts['a1'] = {}
        node.av_counts['a1']['v1'] = 1 
        node.av_counts['a1']['v2'] = 3 
        node.av_counts['a1']['v3'] = 6 

        assert node.expected_correct_guesses() == ((1/10)**2 + (3/10)**2 +
                                                   (6/10)**2)

        node.av_counts['*a2'] = {}
        node.av_counts['*a2']['v1'] = 1 
        node.av_counts['*a2']['v2'] = 1

        assert node.expected_correct_guesses() == ((1/10)**2 + (3/10)**2 +
                                                   (6/10)**2)

        node = Cobweb3()
        node.count = 10
        node.av_counts['a1'] = {}
        v1 = ContinuousValue()
        v1.update(3)
        v1.update(5)
        v2 = ContinuousValue()
        v2.update(1)
        v2.update(11)
        v2.update(1.01)
        node.av_counts['a1'] = v1
        node.av_counts['a2'] = v2

        assert node.expected_correct_guesses() == ((2/10)**2 *
        (1/(2*math.sqrt(math.pi)*v1.unbiased_std())) + 
        (3/10)**2 * (1/(2*math.sqrt(math.pi)*v2.unbiased_std())))

        node.av_counts['*a3'] = v2
        assert node.expected_correct_guesses() == ((2/10)**2 *
        (1/(2*math.sqrt(math.pi)*v1.unbiased_std())) + 
        (3/10)**2 * (1/(2*math.sqrt(math.pi)*v2.unbiased_std())))

if __name__ == "__main__":
    unittest.main()


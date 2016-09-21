from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
import unittest
import random
from numbers import Number

from concept_formation.cobweb3 import cv_key
from concept_formation.cobweb3 import Cobweb3Tree

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
        if attr not in temp:
            temp[attr] = {}
        for val in node.av_counts[attr]:
            if val == cv_key:
                temp[attr][val] = node.av_counts[attr][val].num
            else:
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

                if val == cv_key:
                    temp[attr][val] -= child.av_counts[attr][val].num
                else:
                    temp[attr][val] -= child.av_counts[attr][val]

    assert temp_count == 0

    for attr in temp:
        if isinstance(temp[attr], Number):
            assert temp[attr] == 0.0
        else:
            for val in temp[attr]:
                assert temp[attr][val] == 0.0

    for child in node.children:
        verify_counts(child)


class TestCobweb(unittest.TestCase):

    def test_cobweb(self):
        tree = Cobweb3Tree()
        for i in range(40):
            data = {}
            data['a1'] = random.choice(['v1', 'v2', 'v3', 'v4'])
            data['a2'] = random.choice(['v1', 'v2', 'v3', 'v4'])
            tree.ifit(data)
        verify_counts(tree.root)

    def test_cobweb3(self):
        tree = Cobweb3Tree()
        for i in range(40):
            data = {}
            data['x'] = random.normalvariate(0,4)
            data['y'] = random.normalvariate(0,4)
            tree.ifit(data)
        verify_counts(tree.root)

if __name__ == "__main__":
    unittest.main()


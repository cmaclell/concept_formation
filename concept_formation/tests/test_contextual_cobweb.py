from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
# from ssl import CHANNEL_BINDING_TYPES
import cProfile
import unittest
import random
from numbers import Number
from collections import Counter

from concept_formation.cobweb3 import cv_key
from concept_formation.contextual_cobweb import ca_key
from concept_formation.contextual_cobweb import ContextualCobwebTree
from concept_formation.contextual_cobweb import ContextualCobwebNode
from concept_formation.context_instance import ContextInstance


random.seed(43)


def verify_counts(node):
    """
    Checks the property that the counts of the children sum to the same
    count as the parent. This is/was useful when debugging. This is modified
    from the test_cobweb3.verify_counts to handle contextual values properly.
    """
    if len(node.children) == 0:
        return

    temp = {}
    temp_count = node.count
    for attr in node.av_counts:
        if attr == ca_key:
            temp[attr] = node.av_counts[attr]
            continue
        temp.setdefault(attr, {})
        for val in node.av_counts[attr]:
            if val == cv_key:
                temp[attr][val] = node.av_counts[attr][val].num
            else:
                temp[attr][val] = node.av_counts[attr][val]

    for child in node.children:
        temp_count -= child.count
        for attr in child.av_counts:
            assert attr in temp

            if attr == ca_key:
                temp[attr] -= child.av_counts[attr]
                continue

            for val in child.av_counts[attr]:
                if val not in temp[attr]:
                    print(val)
                    print(attr)
                    print(node)
                assert val in temp[attr]

                if val == cv_key:
                    temp[attr][val] -= child.av_counts[attr][val].num
                else:
                    temp[attr][val] -= child.av_counts[attr][val]

    assert temp_count == 0

    for attr in temp:
        if isinstance(temp[attr], Counter):
            # All values must be 0
            assert not any(temp[attr].values())
        elif isinstance(temp[attr], Number):
            assert temp[attr] == 0.0
        else:
            for val in temp[attr]:
                assert temp[attr][val] == 0.0

    for child in node.children:
        verify_counts(child)


def verify_descendants(node):
    """
    Checks the property that each node's descendant list is the union of its
    children's (or, if it's a leaf, contains itself and only itself).
    """
    if node.children == []:
        assert node.descendants == {node}
        return

    assert node.descendants == set().union(
        *(child.descendants for child in node.children))

    for child in node.children:
        verify_descendants(child)


def verify_tree_structure(node, parent=None):
    """
    Checks the property that the parent attributes of all the nodes in the tree
    are correct."""
    assert node.parent == parent
    for child in node.children:
        verify_tree_structure(child, node)


class TestCobwebNodes(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.node_0 = ContextualCobwebNode()
        cls.node_00 = ContextualCobwebNode()
        cls.node_01 = ContextualCobwebNode()
        cls.node_0.children.extend([cls.node_00, cls.node_01])
        cls.node_00.parent = cls.node_0
        cls.node_01.parent = cls.node_0

    def test_increment_counts(self):
        instance_context = ContextInstance((self.node_0, self.node_00))
        other_ctxt = ContextInstance((self.node_0,))

        pre_inc = self.node_00.count
        self.node_00.increment_counts({ca_key: {instance_context, other_ctxt},
                                      'normal_attr': 'norm', 'numer_attr': 1})
        self.assertEqual(1, self.node_00.count - pre_inc)
        self.node_00.increment_counts({ca_key: {instance_context},
                                      'normal_attr': 'hi'})
        self.assertEqual(2, self.node_00.count - pre_inc)
        self.assertEqual(2, self.node_00.av_counts[ca_key][instance_context])
        self.assertEqual(1, self.node_00.av_counts[ca_key][other_ctxt])
        self.assertEqual(1, self.node_00.av_counts['normal_attr']['hi'])
        # P_3 with all same context should have expectation 1

    def test_context_membership(self):
        context = ContextInstance((self.node_0, self.node_01))
        self.assertTrue(context.desc_of(self.node_0))
        self.assertFalse(context.unadded_leaf(self.node_0))
        self.assertTrue(context.desc_of(self.node_01))
        self.assertTrue(context.unadded_leaf(self.node_01))
        self.assertFalse(context.desc_of(self.node_00))

        context.set_path((self.node_0, self.node_00))
        self.assertTrue(context.desc_of(self.node_00))
        self.assertTrue(context.unadded_leaf(self.node_00))
        self.assertFalse(context.desc_of(self.node_01))

        context.set_path((self.node_0,))
        self.assertTrue(context.desc_of(self.node_0))
        self.assertTrue(context.unadded_leaf(self.node_0))
        self.assertFalse(context.desc_of(self.node_00))
        self.assertFalse(context.unadded_leaf(self.node_00))

        context.set_instance(self.node_00)
        self.assertTrue(context.desc_of(self.node_00))
        self.assertFalse(context.unadded_leaf(self.node_00))
        self.assertFalse(context.desc_of(self.node_01))

        self.assertRaises(AssertionError, context.set_instance, self.node_01)

    def test_update_counts_node(self):
        node = ContextualCobwebNode()
        context = ContextInstance((self.node_0, self.node_01))
        obj1 = {ca_key: {context}, 'normal_attr': 'norm'}
        self.node_00.increment_counts(obj1)
        node.update_counts_from_node(self.node_00)
        self.assertEqual(node.av_counts, self.node_00.av_counts)

    def test_is_exact_match(self):
        node = ContextualCobwebNode()
        context = ContextInstance((self.node_0, self.node_01))
        context2 = ContextInstance((self.node_0,))
        obj1 = {ca_key: {context, context2}, 'normal_attr': 'norm'}
        obj2 = {ca_key: {context}, 'normal_attr': 'norm'}
        node.increment_counts(obj1)
        self.assertTrue(node.is_exact_match(obj1))
        self.assertFalse(node.is_exact_match(obj2))

        node.increment_counts(obj1)
        self.assertTrue(node.is_exact_match(obj1))

        node.increment_counts(obj2)
        self.assertFalse(node.is_exact_match(obj1))

    def test_expected_correct_guesses(self):
        tree = ContextualCobwebTree(ctxt_weight=2)
        root = ContextualCobwebNode()
        tree.root = root
        root.tree = tree

        context = ContextInstance((root,))
        obj = {ca_key: {context}}
        root.increment_counts(obj)
        child = ContextualCobwebNode(root)
        child.parent = root
        root.children.append(child)

        self.assertEqual(1, child.expected_correct_guesses())
        context.set_instance(child)
        self.assertEqual(1, root.expected_correct_guesses())

        child_2 = root.create_new_leaf({}, ContextInstance((root,)))

        context_2 = ContextInstance((root, child_2))
        obj_2 = {ca_key: {context_2}}
        root.increment_counts(obj_2)
        child_2.increment_counts(obj_2)

        self.assertEqual(1, child.expected_correct_guesses())
        self.assertEqual(1, child_2.expected_correct_guesses())
        context_2.set_instance(child_2)
        self.assertEqual(0.75, root.expected_correct_guesses())


class TestCobwebTree(unittest.TestCase):
    def setUp(self):
        self.tree = ContextualCobwebTree()

    def test_tree_initializer(self):
        verify_counts(self.tree.root)
        verify_descendants(self.tree.root)
        verify_tree_structure(self.tree.root)

    def test_add_1_node(self):
        self.tree.contextual_ifit([{'attr': 'val'}])
        verify_counts(self.tree.root)
        verify_descendants(self.tree.root)
        verify_tree_structure(self.tree.root)

    def test_add_2_nodes(self):
        self.tree.contextual_ifit([{'attr': 'val1'}, {'attr': 'val2'}])

    def test_add_3_nodes(self):
        self.tree.contextual_ifit([{'attr': 'v%s' % i} for i in range(3)])
        verify_counts(self.tree.root)
        verify_descendants(self.tree.root)
        verify_tree_structure(self.tree.root)

    def test_add_9_nodes(self):
        self.tree.contextual_ifit([{'attr': 'v%s' % i} for i in range(9)])
        verify_counts(self.tree.root)
        verify_descendants(self.tree.root)
        verify_tree_structure(self.tree.root)

    def test_add_2_batches(self):
        self.tree.contextual_ifit([{'attr': 'v%s' % i} for i in range(9)])
        self.tree.contextual_ifit([{'attr': 'v%s' % i} for i in range(9)])
        verify_counts(self.tree.root)
        verify_descendants(self.tree.root)
        verify_tree_structure(self.tree.root)
        self.assertLessEqual(self.tree.root.expected_correct_guesses(), 1)
        self.assertLessEqual(
            self.tree.root.children[0].expected_correct_guesses(), 1)

    def test_add_many_batches(self):
        for i in range(4):
            self.tree.contextual_ifit(
                [{'a': 'v%s' % (i+random.randint(-2, 2))} for i in range(12)])


if __name__ == "__main__":
    unittest.main()
    # cProfile.run("unittest.main()")

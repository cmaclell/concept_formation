from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
import random

import pytest

from concept_formation.cobweb import CobwebTree
from concept_formation.cobweb import CobwebNode


def verify_counts(node):
    """
    Checks the property that the counts of the children sum to the same
    count as the parent. This is/was useful when debugging. If you are
    doing some kind of matching at each step in the categorization (i.e.,
    renaming such as with Trestle) then this will start throwing errors.
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

    if temp_count != 0:
        print("Parent: %i" % node.count)
        for child in node.children:
            print("Child: %i" % child.count)
    assert temp_count == 0

    for attr in temp:
        for val in temp[attr]:
            if temp[attr][val] != 0.0:
                print(node)

            assert temp[attr][val] == 0.0

    for child in node.children:
        verify_counts(child)


def compare_two_nodes(node1, node2):
    if node1.count != node2.count:
        return False
    if node1.av_counts != node2.av_counts:
        return False
    if len(node1.children) != len(node2.children):
        return False

    for i in range(len(node1.children)):
        if not compare_two_nodes(node1.children[i], node2.children[i]):
            return False

    return True


def test_cobweb_init():
    tree = CobwebTree()
    assert isinstance(tree.root, CobwebNode)
    assert tree == tree.root.tree


def test_cobweb_clear():
    tree = CobwebTree()
    tree.ifit({'a': 'b'})
    tree.clear()

    assert tree.root.count == 0
    assert isinstance(tree.root, CobwebNode)
    assert tree == tree.root.tree


def test_cobweb_str():
    tree = CobwebTree()
    assert str(tree) == str(tree.root)


def test_cobweb_sanity_check():
    tree = CobwebTree()

    with pytest.raises(ValueError):
        tree._sanity_check_instance([set()])

    with pytest.raises(ValueError):
        tree._sanity_check_instance({1: 'a'})

    with pytest.raises(ValueError):
        tree._sanity_check_instance({'a': set([])})

    with pytest.raises(ValueError):
        tree._sanity_check_instance({'a': None})


def test_cobweb_ifit():
    tree = CobwebTree()
    tree.ifit({'a': 'b'})

    assert tree.root.count == 1
    assert len(tree.root.children) == 0
    assert 'a' in tree.root.av_counts
    assert 'b' in tree.root.av_counts['a']


def test_cobweb_fit():
    tree = CobwebTree()
    tree2 = CobwebTree()
    tree3 = CobwebTree()
    tree4 = CobwebTree()
    examples = []
    for i in range(6):
        data = {}
        data['a1'] = random.choice(['v%i' % i for i in range(20)])
        data['a2'] = random.choice(['v%i' % i for i in range(20)])
        examples.append(data)

    tree.fit(examples, randomize_first=False)
    tree2.fit(examples, randomize_first=False)
    tree3.fit(examples, randomize_first=True)
    tree4.fit(examples, iterations=2)

    assert compare_two_nodes(tree.root, tree2.root) is True
    assert compare_two_nodes(tree.root, tree3.root) is False
    assert len(tree.root.children) == len(tree2.root.children)
    assert len(tree.root.children) != len(tree4.root.children)


def test_cobweb():
    tree = CobwebTree()
    for i in range(40):
        data = {}
        data['a1'] = random.choice(['v1', 'v2', 'v3', 'v4'])
        data['a2'] = random.choice(['v1', 'v2', 'v3', 'v4'])
        tree.ifit(data)
    verify_counts(tree.root)


def test_empty_instance():
    t = CobwebTree()
    t.ifit({'x': 1})
    t.ifit({'x': 2})
    t.categorize({})

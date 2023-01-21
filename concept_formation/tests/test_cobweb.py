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
    examples = [{'a': 'a'}, {'b': 'b'}, {'c': 'c'}]
    tree.fit(examples)
    tree2.fit(examples)


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
    t.ifit({'x': '1'})
    t.ifit({'x': '2'})
    t.categorize({})

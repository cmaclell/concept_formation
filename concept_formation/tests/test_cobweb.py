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
    tree1 = CobwebTree()
    tree2 = CobwebTree()
    tree3 = CobwebTree()
    examples = []
    for i in range(10):
        data = {}
        data['a1'] = random.choice(['v%i' % i for i in range(4)])
        data['a2'] = random.choice(['v%i' % i for i in range(4)])
        examples.append(data)

    tree1.fit(examples, randomize_first=False)
    tree2.fit(examples, randomize_first=True)
    tree3.fit(examples, iterations=2)

    assert tree1.root.count == tree2.root.count
    assert tree1.root.count * 2 == tree3.root.count


def test_cobweb_cobweb():
    tree = CobwebTree()
    for i in range(40):
        data = {}
        data['a1'] = random.choice(['v%i' % i for i in range(4)])
        data['a2'] = random.choice(['v%i' % i for i in range(4)])
        tree.ifit(data)
    verify_counts(tree.root)


def test_cobweb_categorize():
    """This tests that categorize always goes to a leaf."""
    tree = CobwebTree()
    node = tree.categorize({})
    assert len(node.children) == 0

    for i in range(15):
        data = {}
        data['a1'] = random.choice(['v%i' % i for i in range(4)])
        data['a2'] = random.choice(['v%i' % i for i in range(4)])
        tree.ifit(data)

    node = tree.categorize({})
    assert len(node.children) == 0

    for i in range(10):
        data = {}
        data['a1'] = random.choice(['v%i' % i for i in range(4)])
        data['a2'] = random.choice(['v%i' % i for i in range(4)])
        node = tree.categorize(data)
        assert len(node.children) == 0


def test_cobweb_infer_missing():
    tree = CobwebTree()
    tree.ifit({'a': '1'})
    inst = tree.infer_missing({})
    assert inst == {'a': '1'}

    tree.ifit({'a': '1'})
    tree.ifit({'a': '2'})
    inst = tree.infer_missing({}, 'most likely')
    assert inst == {'a': '1'}

    vals = []
    for i in range(10):
        inst = tree.infer_missing({}, 'sampled')
        vals.append(inst['a'])
    assert len(set(vals)) == 2


def test_cobwebnode_init():
    node = CobwebNode()

    assert node.count == 0
    assert node.av_counts == {}
    assert node.children == []
    assert node.parent is None
    assert node.tree is None

    node.increment_counts({'a': '1'})

    node2 = CobwebNode()
    node2.increment_counts({'b': '2'})
    node.children.append(node2)

    node3 = CobwebNode(node)
    assert node3.count == 1
    assert 'a' in node3.av_counts
    assert '1' in node3.av_counts['a']
    assert len(node3.children) == 1
    assert 'b' in node3.children[0].av_counts
    assert '2' in node3.children[0].av_counts['b']


def test_cobwebnode_shallow_copy():
    node = CobwebNode()

    assert node.count == 0
    assert node.av_counts == {}
    assert node.children == []
    assert node.parent is None
    assert node.tree is None

    node.increment_counts({'a': '1'})

    node2 = CobwebNode()
    node2.increment_counts({'b': '2'})
    node.children.append(node2)

    node3 = node.shallow_copy()
    assert node3.count == 1
    assert 'a' in node3.av_counts
    assert '1' in node3.av_counts['a']
    assert len(node3.children) == 0


def test_cobwebnode_attrs():
    pass

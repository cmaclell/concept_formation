from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import random

import pytest

from concept_formation import utils
from concept_formation.cobweb3 import ContinuousValue


def test_most_likely_choice():
    n = 100
    some_list = [('a', 0.99), ('b', 0.01)]
    vals = [utils.most_likely_choice(some_list) for i in range(n)]
    a_count = vals.count('a')
    b_count = vals.count('b')
    assert a_count == n
    assert b_count == 0

    with pytest.raises(ValueError):
        utils.most_likely_choice([('a', -1)])

    with pytest.raises(ValueError):
        utils.most_likely_choice([])


def test_weighted_choice():
    n = 1000
    some_list = [('a', 0.9), ('b', 0.1)]
    vals = [utils.weighted_choice(some_list) for i in range(n)]
    a_count = vals.count('a')
    b_count = vals.count('b')
    assert a_count > b_count
    assert b_count > 0

    with pytest.raises(ValueError):
        utils.weighted_choice([('a', -1)])

    with pytest.raises(ValueError):
        utils.weighted_choice([])


def test_cv_mean():
    for i in range(10):
        values = [random.normalvariate(0, 1) for i in range(100)]
        cv = ContinuousValue()
        cv.update_batch(values)
        assert cv.mean - utils.mean(values) < 0.00000000001


def test_std():
    with pytest.raises(ValueError):
        utils.std([])

    vals = [random.normalvariate(0, 1) for i in range(1000)]
    assert utils.std(vals) - 1 < 0.1
    assert utils.std([1, 1, 1]) == 0


def test_mean():
    with pytest.raises(ValueError):
        utils.mean([])

    vals = [random.normalvariate(0, 1) for i in range(1000)]
    assert utils.mean(vals) - 0 < 0.1
    assert utils.mean([1, 1, 1]) == 1


def test_c4():
    with pytest.raises(ValueError):
        utils.c4(1)


def test_cv_std():
    for i in range(10):
        values = [random.normalvariate(0, 1) for i in range(100)]
        cv = ContinuousValue()
        cv.update_batch(values)
        assert cv.biased_std() - utils.std(values) < 0.00000000001


def test_cv_unbiased_std():
    for i in range(10):
        values = [random.normalvariate(0, 1) for i in range(10)]
        cv = ContinuousValue()
        cv.update_batch(values)
        assert (cv.unbiased_std() -
                ((len(values) * utils.std(values) / (len(values) - 1)) /
                 utils.c4(len(values))) < 0.00000000001)


def test_cv_update():
    for i in range(10):
        values = []
        cv = ContinuousValue()
        for i in range(20):
            x = random.normalvariate(0, 1)
            values.append(x)
            cv.update(x)
            assert cv.biased_std() - utils.std(values) < 0.00000000001


def test_cv_combine():
    for i in range(10):
        values1 = [random.normalvariate(0, 1) for i in range(50)]
        values2 = [random.normalvariate(0, 1) for i in range(50)]
        values = values1 + values2
        cv = ContinuousValue()
        cv2 = ContinuousValue()

        cv.update_batch(values1)
        assert cv.biased_std() - utils.std(values1) < 0.00000000001

        cv2.update_batch(values2)
        assert cv2.biased_std() - utils.std(values2) < 0.00000000001

        cv.combine(cv2)
        assert cv.biased_std() - utils.std(values) < 0.00000000001

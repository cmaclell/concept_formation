from random import random
from random import normalvariate

import pytest

from concept_formation.continuous_value import ContinuousValue


def test_cv_init():
    cv = ContinuousValue()
    assert cv.num == 0
    assert cv.mean == 0
    assert cv.meanSq == 0


def test_cv_len():
    cv = ContinuousValue()
    assert len(cv) == 1


def test_cv_copy():
    cv = ContinuousValue()
    for i in range(10):
        cv.update(random())

    cv2 = cv.copy()
    assert cv.num == cv2.num
    assert cv.mean == cv2.mean
    assert cv.meanSq == cv2.meanSq

    cv.update(1000)
    assert cv.num != cv2.num
    assert cv.mean != cv2.mean
    assert cv.meanSq != cv2.meanSq


def test_cv_unbiased_mean():
    nums = [random() for i in range(10)]
    cv = ContinuousValue()
    for n in nums:
        cv.update(n)
    assert cv.unbiased_mean() - sum(nums)/len(nums) <= 1e-6


def test_cv_scaled_unbiased_mean():
    nums = [random() for i in range(10)]
    cv = ContinuousValue()
    for n in nums:
        cv.update(n)
    assert cv.scaled_unbiased_mean(sum(nums)/len(nums), 1) <= 1e-6

    assert (cv.scaled_unbiased_mean(sum(nums)/len(nums), 1) ==
            cv.scaled_unbiased_mean(sum(nums)/len(nums), 0))

    assert (cv.scaled_unbiased_mean(sum(nums)/len(nums), 1) ==
            cv.scaled_unbiased_mean(sum(nums)/len(nums), -1))


def test_cv_biased_std():
    cv = ContinuousValue()
    for _ in range(1000):
        cv.update(normalvariate(0, 1))
    assert cv.biased_std() - 1 <= 0.1


def test_cv_unbiased_std():
    cv = ContinuousValue()
    assert cv.unbiased_std() == 0

    true_std = 10
    error_biased = []
    error_unbiased = []
    for _ in range(100):
        cv = ContinuousValue()
        for _ in range(4):
            cv.update(normalvariate(0, true_std))
        error_biased.append(cv.biased_std() - true_std)
        error_unbiased.append(cv.unbiased_std() - true_std)

    assert abs(sum(error_unbiased)) < abs(sum(error_biased))


def test_cv_scaled_biased_std():
    cv = ContinuousValue()
    for _ in range(1000):
        cv.update(normalvariate(0, 2))
    assert cv.scaled_biased_std(2) - 1 <= 0.1

    assert cv.scaled_biased_std(1) == cv.scaled_biased_std(0)
    assert cv.scaled_biased_std(1) == cv.scaled_biased_std(-1)


def test_cv_scaled_unbiased_std():
    cv = ContinuousValue()
    for _ in range(1000):
        cv.update(normalvariate(0, 2))
    assert cv.scaled_unbiased_std(2) - 1 <= 0.1

    assert cv.scaled_unbiased_std(1) == cv.scaled_unbiased_std(0)
    assert cv.scaled_unbiased_std(1) == cv.scaled_unbiased_std(-1)


def test_cv_hash():
    cv = ContinuousValue()
    cv2 = ContinuousValue()
    assert hash(cv) == hash("#ContinuousValue#")
    assert hash(cv) == hash(cv2)


def test_cv_repr():
    cv = ContinuousValue()
    assert repr(cv) == "0.0000 (0.0000) [0]"


def test_cv_update_batch():
    cv1 = ContinuousValue()
    cv2 = ContinuousValue()
    nums = [random() for i in range(10)]

    for n in nums:
        cv1.update(n)

    cv2.update_batch(nums)

    assert cv1.unbiased_mean() == cv2.unbiased_mean()
    assert cv1.biased_std() == cv2.biased_std()


def test_cv_update():
    cv = ContinuousValue()
    cv.update(1)
    assert cv.num == 1
    assert cv.mean == 1
    assert cv.meanSq == 0

    cv.update(2)
    assert cv.num == 2
    assert cv.mean == 1.5
    assert cv.meanSq == 0.5

    cv = ContinuousValue()
    samples = []
    for i in range(1000):
        s = normalvariate(0, 1)
        cv.update(s)
        samples.append(s)
    assert cv.num == 1000
    assert abs(cv.mean) <= 0.1
    assert abs(cv.meanSq - sum([(s - cv.mean)**2 for s in samples])) <= 1e-5


def test_cv_combine():
    cv1 = ContinuousValue()
    with pytest.raises(ValueError):
        cv1.combine(3)

    cv2 = ContinuousValue()
    cv3 = ContinuousValue()

    nums = [normalvariate(0, 1) for _ in range(1000)]

    cv1.update_batch(nums[:500])
    cv2.update_batch(nums[500:])
    cv3.update_batch(nums)

    cv1.combine(cv2)

    assert cv1.num == cv3.num
    assert abs(cv1.mean - cv3.mean) <= 1e-6
    assert abs(cv1.meanSq - cv3.meanSq) <= 1e-6


def test_cv_integral_of_gaussian_product():
    cv1 = ContinuousValue()
    cv2 = ContinuousValue()

    cv1.update(1)
    cv2.update(1)

    assert abs(cv1.integral_of_gaussian_product(cv2) - 1) <= 1e-6


def test_output_json():
    cv = ContinuousValue()
    d = cv.output_json()

    assert d['mean'] == 0
    assert d['std'] == 0
    assert d['n'] == 0

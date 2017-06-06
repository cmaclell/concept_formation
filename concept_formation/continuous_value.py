from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from math import sqrt
from math import exp
from math import pi


class ContinuousValue():
    """
    This class is used to store the number of samples, the mean of the samples,
    and the squared error of the samples for :ref:`Numeric Values<val-num>`.
    It can be used to perform incremental estimation of the attribute's mean,
    std, and unbiased std.

    Initially the number of values, the mean of the values, and the
    squared errors of the values are set to 0.
    """

    def __init__(self):
        """constructor"""
        self.num = 0.0
        self.mean = 0.0
        self.meansq = 0.0

    def __len__(self):
        return 1

    def copy(self):
        """
        Returns a deep copy of itself.

        :return: a deep copy of the continuous value
        :rtype: ContinuousValue
        """
        v = ContinuousValue()
        v.num = self.num
        v.mean = self.mean
        v.meansq = self.meansq
        return v

    def unbiased_mean(self):
        """
        Returns an unbiased estimate of the mean.

        :return: the unbiased mean
        :rtype: float
        """
        return self.mean

    def scaled_unbiased_mean(self, shift, scale):
        """
        Returns a shifted and scaled unbiased mean. This is equivelent to
        (self.unbiased_mean() - shift) / scale

        This is used as part of numerical value scaling.

        :param shift: the amount to shift the mean by
        :type shift: float
        :param scale: the amount to scale the returned mean by
        :type scale: float
        :return: ``(self.mean - shift) / scale``
        :rtype: float
        """
        if scale <= 0:
            scale = 1
        return (self.mean - shift) / scale

    def biased_std(self):
        """
        Returns a biased estimate of the std (i.e., the sample std)

        :return: biased estimate of the std (i.e., the sample std)
        :rtype: float
        """
        return sqrt(self.meansq / (self.num))

    def unbiased_std(self):
        """
        Returns an unbiased estimate of the std, but for n < 2 the std is
        estimated to be 0.0.

        This implementation uses the rule of thumb bias correction estimation:
        `<https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation#Rule_of_thumb_for_the_normal_distribution>`_

        :return: an unbiased estimate of the std
        :rtype: float

        .. seealso:: :meth:`concept_formation.utils.c4`
        """
        if self.num < 2:
            return 0.0
        return sqrt(self.meansq / (self.num - 1.5))

    def scaled_biased_std(self, scale):
        """
        Returns an biased estimate of the std (see:
        :meth:`ContinuousValue.biased_std`), but also adjusts the std given a
        scale parameter.

        This is used to return std values that have been normalized by some
        value. For edge cases, if scale is less than or equal to 0, then
        scaling is disabled (i.e., scale = 1.0).

        :param scale: an amount to scale biased std estimates by
        :type scale: float
        :return: A scaled unbiased estimate of std
        :rtype: float
        """
        if scale <= 0:
            scale = 1.0
        return self.biased_std() / scale

    def scaled_unbiased_std(self, scale):
        """
        Returns an unbiased estimate of the std (see:
        :meth:`ContinuousValue.unbiased_std`), but also adjusts the std given a
        scale parameter.

        This is used to return std values that have been normalized by some
        value. For edge cases, if scale is less than or equal to 0, then
        scaling is disabled (i.e., scale = 1.0).

        :param scale: an amount to scale unbiased std estimates by
        :type scale: float
        :return: A scaled unbiased estimate of std
        :rtype: float
        """
        if scale <= 0:
            scale = 1.0
        return self.unbiased_std() / scale

    def __hash__(self):
        """
        This hashing function returns the hash of a constant string, so that
        all lookups of a continuous value in a dictionary get mapped to the
        same entry.
        """
        return hash("#ContinuousValue#")

    def __repr__(self):
        """
        The textual representation of a continuous value."
        """
        return "%0.4f (%0.4f) [%i]" % (self.unbiased_mean(),
                                       self.unbiased_std(), self.num)

    def update_batch(self, data):
        """
        Calls the update function on every value in a given dataset

        :param data: A list of numberic values to add to the distribution
        :type data: [Number, Number, ...]
        """
        for x in data:
            self.update(x)

    def update(self, x):
        """
        Incrementally update the mean and squared mean error (meansq) values in
        an efficient and practical (no precision problems) way.

        This uses and algorithm by Knuth found here:
        `<https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_

        :param x: A new value to incorporate into the distribution
        :type x: Number
        """
        self.num += 1
        delta = x - self.mean
        self.mean += delta / self.num
        self.meansq += delta * (x - self.mean)

    def combine(self, other):
        """
        Combine another ContinuousValue's distribution into this one in
        an efficient and practical (no precision problems) way.

        This uses the parallel algorithm by Chan et al. found at:
        `<https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm>`_

        :param other: Another ContinuousValue distribution to be incorporated
            into this one.
        :type other: ContinuousValue
        """
        if not isinstance(other, ContinuousValue):
            raise ValueError("Can only merge 2 continuous values.")
        delta = other.mean - self.mean
        self.meansq = (self.meansq + other.meansq + delta * delta *
                       ((self.num * other.num) / (self.num + other.num)))
        self.mean = ((self.num * self.mean + other.num * other.mean) /
                     (self.num + other.num))
        self.num += other.num

    def integral_of_gaussian_product(self, other):
        """
        Computes the integral (from -inf to inf) of the product of two
        gaussians. It adds gaussian noise to both stds, so that the integral of
        their product never exceeds 1.

        Use formula computed here:
            `<http://www.tina-vision.net/docs/memos/2003-003.pdf>`_
        """
        mu1 = self.unbiased_mean()
        mu2 = other.unbiased_mean()
        sd1 = self.unbiased_std()
        sd2 = other.unbiased_std()

        noisy_sd_squared = 1 / (4 * pi)
        sd1 = sqrt(sd1 * sd1 + noisy_sd_squared)
        sd2 = sqrt(sd2 * sd2 + noisy_sd_squared)
        return ((1 / sqrt(2 * pi * (sd1 * sd1 + sd2 * sd2))) *
                exp(-1 * (mu1 - mu2) * (mu1 - mu2) /
                    (2 * (sd1 * sd1 + sd2 * sd2))))

    def output_json(self):
        return {
            'mean': self.unbiased_mean(),
            'std': self.unbiased_std(),
            'n': self.num
        }

"""
The Cobweb3 module contains the :class:`Cobweb3Tree` and :class:`Cobweb3Node`
classes, which extend the traditional Cobweb capabilities to support numeric
values on attributes.  See: :class:`NumericToNominal
<concept_formation.preprocessor.NumericToNominal>` as a way to convert numeric
values into strings, or :class:`NominalToNumeric
<concept_formation.preprocessor.NominalToNumeric>` for the reverse case.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from random import normalvariate
from math import sqrt
from math import pi
from math import exp
from math import log

import numpy as np

from concept_formation.cobweb import CobwebNode
from concept_formation.cobweb import CobwebTree
from concept_formation.continuous_value import ContinuousValue
from concept_formation.utils import is_number
from concept_formation.utils import weighted_choice
from concept_formation.utils import most_likely_choice

cv_key = "#ContinuousValue#"


class Cobweb3Tree(CobwebTree):
    """
    The Cobweb3Tree contains the knowledge base of a partiucluar instance of
    the Cobweb/3 algorithm and can be used to fit and categorize instances.
    Cobweb/3's main difference over Cobweb is the ability to handle numerical
    attributes by applying an assumption that they should follow a normal
    distribution. For the purposes of Cobweb/3's core algorithms a numeric
    attribute is any value where ``isinstance(instance[attr], Number)`` returns
    ``True``.

    The scaling parameter determines whether online normalization of continuous
    attributes is used, and to what standard deviation the values are scaled
    to. Scaling divides the std of each attribute by the std of the attribute
    in the root divided by the scaling constant (i.e., :math:`\\sigma_{root} /
    scaling` when making category utility calculations.  Scaling is useful to
    balance the weight of different numerical attributes, without scaling the
    magnitude of numerical attributes can affect category utility calculation
    meaning numbers that are naturally larger will recieve preference in the
    category utility calculation.

    :param scaling: The number of standard deviations numeric attributes
        are scaled to. By default this value is 0.5 (half a standard
        deviation), which is the max std of nominal values. If disabiling
        scaling is desirable, then it can be set to False or None.
    :type scaling: a float greater than 0.0, None, or False
    :param inner_attr_scaling: Whether to use the inner most attribute name
        when scaling numeric attributes. For example, if `('attr', '?o1')` was
        an attribute, then the inner most attribute would be 'attr'. When using
        inner most attributes, some objects might have multiple attributes
        (i.e., 'attr' for different objects) that contribute to the scaling.
    :param inner_attr_scaling: boolean
    """
    def __init__(self, scaling=0.5, inner_attr_scaling=True):
        """
        The tree constructor.
        """
        super(Cobweb3Tree, self).__init__()

        self.root = Cobweb3Node()
        self.root.tree = self

        self.hidden_numeric_key = {}
        self.numeric_key = {}
        self.hidden_numeric_count = 0
        self.numeric_count = 0

        self.scaling = scaling
        self.inner_attr_scaling = inner_attr_scaling
        self.attr_scales = Cobweb3Node()
        self.attr_scales.tree = self

    def clear(self):
        """
        Clears the concepts of the tree, but maintains the scaling parameter.
        """
        super(Cobweb3Tree, self).clear()

        self.root = Cobweb3Node()
        self.root.tree = self

        self.hidden_numeric_key = {}
        self.numeric_key = {}
        self.hidden_numeric_count = 0
        self.numeric_count = 0

        self.attr_scales = Cobweb3Node()
        self.attr_scales.tree = self

    def get_inner_attr(self, attr):
        """
        Extracts the inner most attribute name from the provided attribute, if
        the attribute is a tuple and inner_attr_scaling is on. Otherwise it
        just returns the attribute. This is used to for normalizing attributes.

        >>> t = Cobweb3Tree()
        >>> t.get_inner_attr(('a', '?object1'))
        'a'
        >>> t.get_inner_attr('a')
        'a'
        """
        if isinstance(attr, tuple) and self.inner_attr_scaling:
            return attr[0]
        else:
            return attr

    def update_scales(self, instance):
        """
        Reads through all the attributes in an instance and updates the
        tree scales object so that the attributes can be properly scaled.
        """
        # TODO for every attr with an inner attribute, update them all.
        self.attr_scales.increment_counts(instance)

    def update_keys(self, instance):
        """
        Updates the keys in the tree, so that it can be used to construct a new
        concept with the instance.
        """
        for attr in instance:
            val = instance[attr]

            if is_number(val):
                if attr[0] == "_":
                    key = self.hidden_numeric_key
                    if attr not in key:
                        key[attr] = self.hidden_numeric_count
                        self.hidden_numeric_count += 1
                else:
                    key = self.numeric_key
                    if attr not in key:
                        key[attr] = self.numeric_count
                        self.numeric_count += 1

            else:
                key = self.nominal_key
                if attr[0] == "_":
                    key = self.hidden_nominal_key
                if attr not in key:
                    key[attr] = {}

                if val not in key[attr]:
                    if attr[0] == "_":
                        key[attr][val] = self.hidden_nominal_count
                        self.hidden_nominal_count += 1
                    else:
                        key[attr][val] = self.nominal_count
                        self.nominal_count += 1

    def create_instance_concept(self, instance, allow_extra=False):
        concept = Cobweb3Node()
        concept.count = 1
        concept.hidden_counts = np.zeros(self.hidden_nominal_count)
        concept.counts = np.zeros(self.nominal_count)

        concept.hidden_num = np.zeros(self.hidden_numeric_count)
        concept.hidden_mean = np.zeros(self.hidden_numeric_count)
        concept.hidden_meansq = np.zeros(self.hidden_numeric_count)

        concept.num = np.zeros(self.numeric_count)
        concept.mean = np.zeros(self.numeric_count)
        concept.meansq = np.zeros(self.numeric_count)

        concept.attributes = set(instance)
        concept.tree = self

        extra_hidden_nominals = 0
        extra_hidden_numerics = 0
        extra_nominals = 0
        extra_numerics = 0

        for attr in instance:
            val = instance[attr]

            if is_number(val):
                if attr[0] == "_":
                    if attr not in self.hidden_numeric_key:
                        extra_hidden_numerics += 1
                        continue
                    idx = self.hidden_numeric_key[attr]
                    concept.hidden_num[idx] = 1
                    concept.hidden_mean[idx] = val
                    concept.hidden_meansq[idx] = 0.0
                else:
                    if attr not in self.numeric_key:
                        extra_numerics += 1
                        continue
                    idx = self.numeric_key[attr]
                    concept.num[idx] = 1
                    concept.mean[idx] = val
                    concept.meansq[idx] = 0.0
            else:
                if attr[0] == "_":
                    if (attr not in self.hidden_nominal_key or
                            val not in self.hidden_nominal_key[attr]):
                        extra_hidden_nominals += 1
                        continue
                    idx = self.hidden_nominal_key[attr][instance[attr]]
                    concept.hidden_counts[idx] = 1
                else:
                    if (attr not in self.nominal_key or
                            val not in self.nominal_key[attr]):
                        extra_nominals += 1
                        continue
                    idx = self.nominal_key[attr][instance[attr]]
                    concept.counts[idx] = 1

        if (allow_extra is False and
            (extra_hidden_nominals > 0 or extra_nominals > 0 or
             extra_hidden_numerics > 0 or extra_numerics > 0)):
            raise ValueError("Extra attribue values that do not exist"
                             " in the key. Call update_keys first.")

        concept.hidden_counts = np.append(concept.hidden_counts,
                                          np.ones(extra_hidden_nominals))
        concept.counts = np.append(concept.counts, np.ones(extra_nominals))

        concept.hidden_num = np.append(concept.hidden_num,
                                       np.ones(extra_hidden_numerics))
        concept.hidden_mean = np.append(concept.hidden_mean,
                                        np.zeros(extra_hidden_numerics))
        concept.hidden_meansq = np.append(concept.hidden_meansq,
                                          np.zeros(extra_hidden_numerics))
        concept.num = np.append(concept.num, np.ones(extra_numerics))
        concept.mean = np.append(concept.mean, np.zeros(extra_numerics))
        concept.meansq = np.append(concept.meansq, np.zeros(extra_numerics))

        return concept

    def ifit(self, instance):
        """
        Incrementally fit a new instance into the tree and return its resulting
        concept.

        The cobweb3 version of the :meth:`CobwebTree.ifit` function. This
        version keeps track of all of the continuous

        :param instance: An instance to be categorized into the tree.
        :type instance:  :ref:`Instance<instance-rep>`
        :return: A concept describing the instance
        :rtype: Cobweb3Node

        .. seealso:: :meth:`CobwebTree.cobweb`
        """
        self._sanity_check_instance(instance)
        self.update_keys(instance)
        instance = self.create_instance_concept(instance)
        self.update_scales(instance)
        # print(instance)
        return self.cobweb(instance)


class Cobweb3Node(CobwebNode):
    """
    A Cobweb3Node represents a concept within the knoweldge base of a
    particular :class:`Cobweb3Tree`. Each node contians a probability table
    that can be used to calculate the probability of different attributes given
    the concept that the node represents.

    In general the :meth:`Cobweb3Tree.ifit`, :meth:`Cobweb3Tree.categorize`
    functions should be used to initially interface with the Cobweb/3 knowledge
    base and then the returned concept can be used to calculate probabilities
    of certain attributes or determine concept labels.
    """

    def __init__(self, otherNode=None):
        """Create a new Cobweb3Node"""

        self.hidden_num = np.array([])
        self.hidden_mean = np.array([])
        self.hidden_meansq = np.array([])

        self.num = np.array([])
        self.mean = np.array([])
        self.meansq = np.array([])

        super(Cobweb3Node, self).__init__(otherNode)

    def unbiased_means(self):
        """
        Returns an unbiased estimate of the means.

        :return: the unbiased means
        :rtype: np.array
        """
        return self.mean

    def scaled_unbiased_mean(self, shift, scale):
        """
        Returns the shifted and scaled unbiased means. This is equivelent to
        (self.unbiased_mean() - shift) / scale. This will adjust the scale
        vector to replace any values <= 0 with 1. 

        :param shift: the amount to shift the mean by
        :type shift: np.array
        :param scale: the amount to scale the returned mean by
        :type scale: np.array
        :return: ``(self.mean - shift) / scale``
        :rtype: np.array
        """
        scale[scale <= 0] = 1.0
        return (self.mean - shift) / scale

    def biased_stds(self):
        """
        Returns a biased estimate of the std (i.e., the sample std)

        :return: biased estimate of the std (i.e., the sample std)
        :rtype: np.array
        """
        return np.sqrt(np.divide(self.meansq, self.num))

    def scaled_biased_std(self, scale):
        """
        Returns an biased estimate of the std (see: :meth:`biased_stds`), but
        also adjusts the std given a scale parameter.

        This is used to return std values that have been normalized by some
        value. For edge cases, if scale is less than or equal to 0, then
        scaling is disabled (i.e., scale = 1.0).

        :param scale: an amount to scale biased std estimates by
        :type scale: np.array
        :return: A scaled unbiased estimate of std
        :rtype: np.array
        """
        scale[scale <= 0] = 1.0
        return np.divide(self.biased_stds(), scale)

    def unbiased_stds(self):
        """
        Returns an biased-corrected estimate of the std, but for n < 2 the std
        is estimated to be 0.0.

        This implementation uses the rule of thumb bias correction estimation:
        `<https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation#Rule_of_thumb_for_the_normal_distribution>`_

        :return: an unbiased estimate of the std
        :rtype: np.float
        """
        if len(self.num) == 0:
            return np.array([])
        resp = np.sqrt(np.divide(self.meansq, self.num - 1.5))
        resp[self.num < 2] = 0.0
        return resp

    def scaled_unbiased_stds(self, scale):
        """
        Returns an unbiased estimate of the std (see: :meth:`unbiased_stds`),
        but also adjusts the stds given a scale parameter.

        This is used to return std values that have been normalized by some
        value. For edge cases, if scale is less than or equal to 0, then
        scaling is disabled (i.e., scale = 1.0).

        :param scale: an amount to scale unbiased std estimates by
        :type scale: np.array
        :return: A scaled unbiased estimate of std
        :rtype: np.array
        """
        scale[scale <= 0] = 1.0
        return np.divide(self.unbiased_stds(), scale)

    def increment_counts(self, instance):
        """
        Increment the counts at the current node according to the specified
        instance concept.

        Cobweb3Node uses a modified version of
        :meth:`CobwebNode.increment_counts
        <concept_formation.cobweb.CobwebNode.increment_counts>` that handles
        numerical attributes properly.

        :param instance: A new instance concept to incorporate into the node.
        :type instance: Cobweb3Node
        """
        self.update_counts_from_node(instance)

    def update_counts_from_node(self, node):
        """
        Increments the counts of the current node by the amount in the
        specified node, modified to handle numbers.

        :param node: Another node from the same Cobweb3Tree
        :type node: Cobweb3Node
        """
        super(Cobweb3Node, self).update_counts_from_node(node)

        self.ensure_vector_size(node)

        # Update numeric counts
        delta = node.hidden_mean - self.hidden_mean
        hidden_num = self.hidden_num + node.hidden_num
        self.hidden_meansq += (node.hidden_meansq +
                               np.multiply(np.multiply(delta, delta),
                                           (np.divide(
                                               np.multiply(self.hidden_num,
                                                           node.hidden_num),
                                               hidden_num))))
        self.hidden_mean = (np.divide((np.multiply(self.hidden_num,
                                                   self.hidden_mean) +
                                       np.multiply(node.hidden_num,
                                                   node.hidden_mean)),
                                      hidden_num))
        self.hidden_num = hidden_num
        self.hidden_mean[self.hidden_num == 0] = 0.0
        self.hidden_meansq[self.hidden_num == 0] = 0.0

        delta = node.mean - self.mean
        num = self.num + node.num
        self.meansq += (node.meansq +
                        np.multiply(np.multiply(delta, delta),
                                    (np.divide(np.multiply(self.num, node.num),
                                               num))))
        self.mean = (np.divide((np.multiply(self.num, self.mean) +
                                np.multiply(node.num, node.mean)),
                               num))
        self.num = num
        self.mean[self.num == 0] = 0.0
        self.meansq[self.num == 0] = 0.0

    def expected_correct_guesses(self):
        """
        Returns the number of attribute values that would be correctly guessed
        in the current concept. This extension supports both nominal and
        numeric attribute values.

        The typical Cobweb/3 calculation for correct guesses is:

        .. math::

            P(A_i = V_{ij})^2 = \\frac{1}{2 * \\sqrt{\\pi} * \\sigma}

        However, this does not take into account situations when
        :math:`P(A_i) < 1.0`. Additionally, the original formulation set
        :math:`\\sigma` to have a user specified minimum value. However, for
        small lower bounds, this lets cobweb achieve more than 1 expected
        correct guess per attribute, which is impossible for nominal attributes
        (and does not really make sense for continuous either). This causes
        problems when both nominal and continuous values are being used
        together; i.e., continuous attributes will get higher preference.

        To account for this we use a modified equation:

        .. math::

            P(A_i = V_{ij})^2 = P(A_i)^2 * \\frac{1}{2 * \\sqrt{\\pi} *
            \\sigma}

        The key change here is that we multiply by :math:`P(A_i)^2`.
        Further, instead of bounding :math:`\\sigma` by a user specified lower
        bound (often called acuity), we add some independent, normally
        distributed noise to sigma:
        :math:`\\sigma = \\sqrt{\\sigma^2 + \\sigma_{noise}^2}`, where
        :math:`\\sigma_{noise} = \\frac{1}{2 * \\sqrt{\\pi}}`.
        This ensures the expected correct guesses never exceeds 1. From a
        theoretical point of view, it basically is an assumption that there is
        some independent, normally distributed measurement error that is added
        to the estimated error of the attribute (`<https://en.wikipedia.org/wi
        ki/Sum_of_normally_distributed_random_variables>`_). It is possible
        that there is additional measurement error, but the value is chosen so
        as to yield a sensical upper bound on the expected correct guesses.

        :return: The number of attribute values that would be correctly guessed
            in the current concept.
        :rtype: float
        """
        ec = super(Cobweb3Node, self).expected_correct_guesses()

        self.ensure_vector_size()

        scale = np.zeros(self.num.shape)
        if self.tree is not None and self.tree.scaling:
            scale_std = self.tree.attr_scales.unbiased_stds()
            scale_std.resize(self.num.shape)
            scale += ((1/self.tree.scaling) *
                      scale_std)
            # self.tree.attr_scales.unbiased_stds())

        # we basically add noise to the std and adjust the normalizing constant
        # to ensure the probability of a particular value never exceeds 1.
        scaled_stds = self.scaled_unbiased_stds(scale)
        noisy_stds = np.sqrt(np.multiply(scaled_stds, scaled_stds) +
                             (1 / (4 * pi)))

        prob_attr = self.num / self.count
        ec += (np.dot(np.multiply(prob_attr, prob_attr),
                      (1/(2 * sqrt(pi) * noisy_stds))) /
               len(self.attributes))

        return ec

    def av_counts(self):
        """
        Generates an av_counts table that can be used for generating textual
        and json representations of a concept.
        """
        av_counts = super(Cobweb3Node, self).av_counts()

        self.ensure_vector_size()

        key = self.tree.hidden_numeric_key
        for attr in key:
            idx = key[attr]
            if self.hidden_num[idx] == 0:
                continue
            if attr not in av_counts:
                av_counts[attr] = {}
            if (self.hidden_num[idx] == 0):
                continue
            val = ContinuousValue()
            val.num = self.hidden_num[idx]
            val.mean = self.hidden_mean[idx]
            val.meansq = self.hidden_meansq[idx]
            av_counts[attr][cv_key] = val

        key = self.tree.numeric_key
        for attr in key:
            idx = key[attr]
            if self.num[idx] == 0:
                continue
            if attr not in av_counts:
                av_counts[attr] = {}
            if (self.num[idx] == 0):
                continue
            val = ContinuousValue()
            val.num = self.num[idx]
            val.mean = self.mean[idx]
            val.meansq = self.meansq[idx]
            av_counts[attr][cv_key] = val

        return av_counts

    def pretty_print(self, depth=0):
        """
        Print the categorization tree

        The string formatting inserts tab characters to align child nodes of
        the same depth. Numerical values are printed with their means and
        standard deviations.

        :param depth: The current depth in the print, intended to be called
            recursively
        :type depth: int
        :return: a formated string displaying the tree and its children
        :rtype: str
        """
        ret = str(('\t' * depth) + "|-")

        attributes = []

        av_counts = self.av_counts()

        for attr in av_counts:
            values = []
            for val in av_counts[attr]:
                values.append("'" + str(val) + "': " +
                              str(av_counts[attr][val]))

            attributes.append("'" + str(attr) + "': {" + ", ".join(values)
                              + "}")

        ret += "{" + ", ".join(attributes) + "}: " + str(self.count) + '\n'

        for c in self.children:
            ret += c.pretty_print(depth+1)

        return ret

    def get_weighted_values(self, attr, allow_none=True):
        """
        Return a list of weighted choices for an attribute based on the node's
        probability table.

        This calculation will include an option for the change that an
        attribute is missing from an instance all together. This is useful for
        probability and sampling calculations. If the attribute has never
        appeared in the tree then it will return a 100% chance of None.

        :param attr: an attribute of an instance
        :type attr: :ref:`Attribute<attributes>`
        :param allow_none: whether attributes in the nodes probability table
            can be inferred to be missing. If False, then None will not be
            cosidered as a possible value.
        :type allow_none: Boolean
        :return: a list of weighted choices for attr's value
        :rtype: [(:ref:`Value<values>`, float), (:ref:`Value<values>`, float),
            ...]
        """
        choices = []

        self.ensure_vector_size()

        if (attr not in self.tree.hidden_nominal_key and
                attr not in self.tree.hidden_numeric_key and
                attr not in self.tree.nominal_key and
                attr not in self.tree.numeric_key):
            choices.append((None, 1.0))
            return choices

        key = self.tree.nominal_key
        counts = self.counts
        if attr[0] == "_":
            key = self.tree.hidden_nominal_key
            counts = self.hidden_counts

        val_count = 0
        for val in key:
            idx = key[attr][val]
            count = counts[idx]
            choices.append((val, count / self.count))
            val_count += count

        key = self.tree.numeric_key
        num = self.num
        mean = self.mean
        meansq = self.meansq
        if attr[0] == "_":
            key = self.tree.hidden_numeric_key
            num = self.hidden_num
            mean = self.hidden_mean
            meansq = self.hidden_meansq

        if attr in key:
            idx = key[attr]
            count = num[idx]
            val = ContinuousValue()
            val.num = num[idx]
            val.mean = mean[idx]
            val.meansq = meansq[idx]
            choices.append((val, count / self.count))
            val_count += count

        if allow_none:
            choices.append((None, ((self.count - val_count) / self.count)))

        return choices

    def predict(self, attr, choice_fn="most likely", allow_none=True):
        """
        Predict the value of an attribute, using the provided strategy.

        If the attribute is a nominal then this function behaves the same as
        :meth:`CobwebNode.predict <concept_formation.cobweb.CobwebNode.predict>`.
        If the attribute is numeric then the mean value from the
        :class:`ContinuousValue<concept_formation.cv_key.ContinuousValue>` is chosen.

        :param attr: an attribute of an instance.
        :type attr: :ref:`Attribute<attributes>`
        :param allow_none: whether attributes not in the instance can be
            inferred to be missing. If False, then all attributes will be
            inferred with some value.
        :type allow_none: Boolean
        :return: The most likely value for the given attribute in the node's 
            probability table.
        :rtype: :ref:`Value<values>`

        .. seealso :meth:`Cobweb3Node.sample`
        """
        if choice_fn == "most likely" or choice_fn == "m":
            choose = most_likely_choice
        elif choice_fn == "sampled" or choice_fn == "s":
            choose = weighted_choice
        else:
            raise Exception("Unknown choice_fn")

        if (attr not in self.tree.hidden_nominal_key and
                attr not in self.tree.nominal_key and
                attr not in self.tree.hidden_numeric_key and
                attr not in self.tree.numeric_key):
            return None

        choices = self.get_weighted_values(attr, allow_none)
        val = choose(choices)

        if isinstance(val, ContinuousValue):
            if choice_fn == "most likely" or choice_fn == "m":
                val = val.unbiased_mean()
            elif choice_fn == "sampled" or choice_fn == "s":
                val = normalvariate(val.unbiased_mean(),
                                    val.unbiased_std())
            else:
                raise Exception("Unknown choice_fn")

        return val

    def ensure_vector_size(self, other=None):
        super(Cobweb3Node, self).ensure_vector_size(other)

        if other is None:
            if len(self.num) < self.tree.numeric_count:
                self.num.resize(self.tree.numeric_count)
                self.mean.resize(self.tree.numeric_count)
                self.meansq.resize(self.tree.numeric_count)
            if len(self.hidden_num) < self.tree.hidden_numeric_count:
                self.hidden_num.resize(self.tree.hidden_numeric_count)
                self.hidden_mean.resize(self.tree.hidden_numeric_count)
                self.hidden_meansq.resize(self.tree.hidden_numeric_count)
        else:
            if len(self.hidden_num) < len(other.hidden_num):
                self.hidden_num.resize(other.hidden_num.shape)
                self.hidden_mean.resize(other.hidden_mean.shape)
                self.hidden_meansq.resize(other.hidden_meansq.shape)
            if len(self.hidden_num) > len(other.hidden_num):
                other.hidden_num.resize(self.hidden_num.shape)
                other.hidden_mean.resize(self.hidden_mean.shape)
                other.hidden_meansq.resize(self.hidden_meansq.shape)

            if len(self.num) < len(other.num):
                self.num.resize(other.num.shape)
                self.mean.resize(other.mean.shape)
                self.meansq.resize(other.meansq.shape)
            if len(self.num) > len(other.num):
                other.num.resize(self.num.shape)
                other.mean.resize(self.mean.shape)
                other.meansq.resize(self.meansq.shape)

    def probability(self, attr, val):
        """
        Returns the probability of a particular attribute value at the current
        concept.

        This takes into account the possibilities that an attribute can take
        any of the values available at the root, or be missing.

        For numerical attributes it returns the integral of the product of two
        gaussians. One gaussian has :math:`\\mu = val` and :math:`\\sigma =
        \\sigma_{noise} = \\frac{1}{2 * \\sqrt{\\pi}}` (where
        :math:`\\sigma_{noise}` is from
        :meth:`Cobweb3Node.expected_correct_guesses
        <concept_formation.cobweb3.Cobweb3Node.expected_correct_guesses>` and
        ensures the probability or expected correct guesses never exceeds 1).
        The second gaussian has the mean and std values from the current concept
        with additional gaussian noise (independent and normally distributed
        noise with :math:`\\sigma_{noise} = \\frac{1}{2 * \\sqrt{\\pi}}`).

        The integral of this gaussian product is another gaussian with
        :math:`\\mu` equal to the concept attribute mean and :math:`\\sigma =
        \\sqrt{\\sigma_{attr}^2 + 2 * \\sigma_{noise}^2}` or, slightly
        simplified, :math:`\\sigma =
        \\sqrt{\\sigma_{attr}^2 + 2 * \\frac{1}{2 * \\pi}}`.

        :param attr: an attribute of an instance
        :type attr: :ref:`Attribute<attributes>`
        :param val: a value for the given attribute
        :type val: :ref:`Value<values>`
        :return: The probability of attr having the value val in the current
            concept.
        :rtype: float
        """
        # TODO Why is this important?
        assert self.tree is not None

        self.ensure_vector_size()

        nominal_key = self.tree.nominal_key
        counts = self.counts
        numeric_key = self.tree.numeric_key
        num = self.num
        if attr[0] == "_":
            nominal_key = self.tree.hidden_nominal_key
            counts = self.hidden_counts
            numeric_key = self.tree.hidden_numeric_key
            num = self.hidden_num

        if val is None:
            c = 0.0
            if attr in nominal_key:
                c = sum([counts[nominal_key[attr][v]] for v in
                         nominal_key[attr]])
            if attr in numeric_key:
                c += num[numeric_key[attr]]
            return (self.count - c) / self.count

        if not is_number(val):
            return super(Cobweb3Node, self).probability(attr, val)

        if attr not in numeric_key or num[numeric_key[attr]] == 0:
            return 0.0

        prob_attr = num[numeric_key[attr]] / self.count
        attr_scale = ContinuousValue()
        attr_scale.num = self.tree.attr_scales.num[numeric_key[attr]]
        attr_scale.mean = self.tree.attr_scales.mean[numeric_key[attr]]
        attr_scale.meansq = self.tree.attr_scales.meansq[numeric_key[attr]]

        cv = ContinuousValue()
        cv.num = self.num[numeric_key[attr]]
        cv.mean = self.mean[numeric_key[attr]]
        cv.meansq = self.meansq[numeric_key[attr]]

        if self.tree.scaling:
            scale = (1/self.tree.scaling) * attr_scale.unbiased_std()
            if scale == 0:
                scale = 1
            shift = attr_scale.unbiased_mean()
            val = (val - shift) / scale
        else:
            scale = 1.0
            shift = 0.0

        # mean = (av_counts[attr][cv_key].mean - shift) / scale
        mean = cv.scaled_unbiased_mean(shift, scale)
        std = cv.scaled_unbiased_std(scale)
        noisy_std = sqrt(std * std + (1 / (2 * pi)))
        p = (prob_attr *
             (1/(sqrt(2*pi) * noisy_std)) *
             exp(-((val - mean) * (val - mean)) /
                 (2.0 * noisy_std * noisy_std)))
        return p

    def log_likelihood(self, child_leaf):
        """
        Returns the log-likelihood of the concept. To compute the probability
        of one concept's numeric value generating another we integrate the
        product of their two gaussians.
        """
        ll = super(Cobweb3Node, self).log_likelihood(child_leaf)

        key = self.tree.numeric_key

        for attr in key:
            idx = key[attr]
            if self.num[idx] > 0 and child_leaf.num[idx] > 0:
                n1 = ContinuousValue()
                n1.num = self.num[idx]
                n1.mean = self.mean[idx]
                n1.meansq = self.meansq[idx]
                n2 = ContinuousValue()
                n2.num = child_leaf.num[idx]
                n2.mean = child_leaf.mean[idx]
                n2.meansq = child_leaf.meansq[idx]

                pn1 = n1.num / self.count
                pn2 = n2.num / child_leaf.count
                p = pn1 * pn2 * n1.integral_of_gaussian_product(n2)

                if p > 0:
                    ll += log(p)
                else:
                    raise Exception("p should be greater than 0")

        return ll

    def is_exact_match(self, instance):
        """
        Returns true if the concept exactly matches the instance.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: whether the instance perfectly matches the concept
        :rtype: boolean

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        self.ensure_vector_size(instance)

        if (not np.array_equal(self.mean,
                               instance.mean)):
            return False

        if 0.0 not in self.meansq:
            return False

        return super(Cobweb3Node, self).is_exact_match(instance)

    def output_json(self):
        """
        Outputs the categorization tree in JSON form.

        This is a modification of the :meth:`CobwebNode.output_json
        <concept_formation.cobweb.CobwebNode.output_json>` to handle numeric
        values.

        :return: an object that contains all of the structural information of
            the node and its children
        :rtype: obj
        """
        output = {}
        counts = self.av_counts()
        if "_guid" in counts:
            for guid in counts['_guid']:
                output['guid'] = guid
        output["name"] = "Concept" + str(self.concept_id)
        output["size"] = self.count
        output["children"] = []

        temp = {}
        for attr in counts:
            temp[str(attr)] = {}

            for val in counts[attr]:
                if val == cv_key:
                    temp[str(attr)][cv_key] = counts[attr][val].output_json()
                else:
                    temp[str(attr)][str(val)] = counts[attr][val]

        for child in self.children:
            output["children"].append(child.output_json())

        output["counts"] = temp

        return output

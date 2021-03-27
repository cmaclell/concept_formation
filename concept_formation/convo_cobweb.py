"""
The Trestle module contains the :class:`TrestleTree` class, which extends
Cobweb3 to support component and relational attributes.
"""
import numpy as np
from pprint import pprint
from collections import Counter
from math import sqrt
from math import pi

from concept_formation.utils import isNumber
from concept_formation.cobweb3 import Cobweb3Tree
from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.cobweb3 import cv_key
from concept_formation.continuous_value import ContinuousValue


def apply_filter(img_dict, filter_size, stride):
    ret = {}

    size = int(np.sqrt(len(img_dict.keys())))

    for row in range(size - filter_size + 1):
        for col in range(size - filter_size + 1):
            new = {}
            for fr in range(filter_size):
                for fc in range(filter_size):
                    new["{},{}".format(fr, fc)] = img_dict["{},{}".format(row+fr, col+fc)]
            ret["{},{}".format(row, col)] = new

    return ret


def convert_img_to_dict_tree(img, filter_size, stride):
    ret = {}

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # ret["{},{}".format(row, col)] = {'pixel_intensity': img[row, col]}
            ret["{},{}".format(row, col)] = img[row, col]

    while np.sqrt(len(ret.keys())) >= filter_size:
        ret = apply_filter(ret, filter_size, stride)

    return ret


def find_common_parent(node1, node2):

    node1_parents = set()
    curr = node1
    node1_parents.add(curr)
    while curr.parent:
        curr = curr.parent
        node1_parents.add(curr)

    curr = node2
    while curr:
        if curr in node1_parents:
            return curr
        curr = curr.parent

    raise ValueError("No common parent, nodes not in same tree.")

def is_parent(parent, child):
    curr = child
    while curr:
        if curr == parent:
            return True
        curr = curr.parent
    return False


class ConvoCobwebNode(Cobweb3Node):

    def __str__(self):
        return "Concept" + str(self.concept_id)

    def expected_correct_guesses(self):
        """
        handle concepts as values
        """
        correct_guesses = 0.0
        attr_count = 0

        for attr in self.attrs():
            attr_count += 1

            concept_avs = Counter()
            for val in self.av_counts[attr]:
                if val == cv_key:
                    scale = 1.0
                    if self.tree is not None and self.tree.scaling:
                        inner_attr = self.tree.get_inner_attr(attr)
                        if inner_attr in self.tree.attr_scales:
                            inner = self.tree.attr_scales[inner_attr]
                            scale = ((1/self.tree.scaling) *
                                     inner.unbiased_std())

                    # we basically add noise to the std and adjust the
                    # normalizing constant to ensure the probability of a
                    # particular value never exceeds 1.
                    cv = self.av_counts[attr][cv_key]
                    std = sqrt(cv.scaled_unbiased_std(scale) *
                               cv.scaled_unbiased_std(scale) +
                               (1 / (4 * pi)))
                    prob_attr = cv.num / self.count
                    correct_guesses += ((prob_attr * prob_attr) *
                                        (1/(2 * sqrt(pi) * std)))
                elif isinstance(val, ConvoCobwebNode):
                    curr = val
                    while curr:
                        concept_avs[curr] += self.av_counts[attr][val]
                        curr = curr.parent
                else:
                    prob = (self.av_counts[attr][val]) / self.count
                    correct_guesses += (prob * prob)

            for val in concept_avs:
                prob = concept_avs[val] / self.count
                correct_guesses += (prob * prob) / len(concept_avs)

        return correct_guesses / attr_count


class ConvoCobwebTree(Cobweb3Tree):
    """
    A new version of cobweb specificaially for image data. It accepts images
    and decomposes them into convolutional trees, which are then passed to a
    version of cobweb that operates similar to labyrinth. The algorithm does
    not do structure mapping, but does attribute generalization.

    The filter_size, and stride determies how the image is decomposed into a
    tree.

    The scaling parameter determines whether online normalization of continuous
    attributes is used, and to what standard deviation the values are scaled
    to. Scaling divides the std of each attribute by the std of the attribute
    in the root divided by the scaling constant (i.e.,
    :math:`\\sigma_{root} / scaling` when making category utility calculations.
    Scaling is useful to balance the weight of different numerical attributes,
    without scaling the magnitude of numerical attributes can affect category
    utility calculation meaning numbers that are naturally larger will recieve
    preference in the category utility calculation.

    :param filter_size: The 2d size of the pooling filter. By default this is
        3.
    :param stride: This determines how far the filter is shifted for each
        superpixel. By default this is 1.
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
    :param structure_map_internally: Determines whether structure mapping is
        used at each node during categorization (and when merging), this
        drastically reduces performance, but allows the category structure to
        influcence structure mapping.
    :type structure_map_internally: boolean
    """

    def __init__(self, filter_size=3, stride=1, scaling=0.5,
                 inner_attr_scaling=True):
        """
        The tree constructor.
        """
        self.gensym_counter = 0
        self.root = ConvoCobwebNode()
        self.root.tree = self
        self.scaling = scaling
        self.inner_attr_scaling = inner_attr_scaling
        self.attr_scales = {}
        self.filter_size = filter_size
        self.stride = stride

    def clear(self):
        """
        Clear the tree but keep initialization parameters
        """
        self.gensym_counter = 0
        self.root = ConvoCobwebNode()
        self.root.tree = self
        self.attr_scales = {}

    def ifit(self, instance):
        """
        Incrementally fit a new instance into the tree and return its resulting
        concept.

        The instance is passed down the tree and updates each node to
        incorporate the instance. **This modifies the tree's knowledge** for a
        non-modifying version see: :meth:`TrestleTree.categorize`.

        This version is modified from the normal :meth:`CobwebTree.ifit
        <concept_formation.cobweb.CobwebTree.ifit>` by first structure mapping
        the instance before fitting it into the knoweldge base.

        :param instance: an instance to be categorized into the tree.
        :type instance: :ref:`Instance<instance-rep>`
        :return: A concept describing the instance
        :rtype: Cobweb3Node

        .. seealso:: :meth:`TrestleTree.trestle`
        """
        instance = {attr: convert_img_to_dict_tree(
            instance[attr], filter_size=self.filter_size, stride=self.stride)
            if isinstance(instance[attr], np.ndarray) else instance[attr] for
            attr in instance}
        # pprint(instance)

        return self.convo_cobweb(instance)

    def convo_cobweb(self, instance):
        """
        The main labyrinth algorithm.
        """
        # new = {attr: self.convo_cobweb(instance[attr]) if
        #        isinstance(instance[attr], dict) else instance[attr] for attr in
        #        instance}

        new = {'sub-component for {}'.format(attr) if isinstance(instance[attr], dict) else
               attr: self.convo_cobweb(instance[attr]) if
               isinstance(instance[attr], dict) else instance[attr] for attr in
               instance}

        # new = {}
        # for attr in instance:
        #     if isinstance(instance[attr], dict):
        #         node = self.convo_cobweb(instance[attr])
        #         count = 0
        #         while node:
        #             count += 1
        #             new[(attr, "Concept" + str(node.concept_id))] = True
        #             node = node.parent
        #         print(count)
        #     else:
        #         new[attr] = instance[attr]

        return self.cobweb(new)

    def _convo_cobweb_categorize(self, instance):
        """
        The structure maps the instance, categorizes the matched instance, and
        returns the resulting concept.

        :param instance: an instance to be categorized into the tree.
        :type instance: {a1:v1, a2:v2, ...}
        :return: A concept describing the instance
        :rtype: concept
        """
        # instance = {attr: self.convo_cobweb_categorize(instance[attr]) if
        #             isinstance(instance[attr], dict) else instance[attr] for
        #             attr in instance}

        new = {'sub-component for {}'.format(attr) if
               isinstance(instance[attr], dict) else attr:
               self._convo_cobweb_categorize(instance[attr]) if
               isinstance(instance[attr], dict) else instance[attr] for attr in
               instance}

        # new = {}
        # for attr in instance:
        #     if isinstance(instance[attr], dict):
        #         node = self.convo_cobweb_categorize(instance[attr])
        #         while node:
        #             new[(str(node), attr)] = True
        #             node = node.parent
        #     else:
        #         new[attr] = instance[attr]

        return self._cobweb_categorize(new)

    def categorize(self, instance):
        """
        convo cobweb categorize without modifying.
        """
        instance = {attr: convert_img_to_dict_tree(
            instance[attr], filter_size=self.filter_size, stride=self.stride)
            if isinstance(instance[attr], np.ndarray) else instance[attr] for
            attr in instance}
        return self._convo_cobweb_categorize(instance)

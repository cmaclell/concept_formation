"""
The ConvoCobweb module contains the :class:`ConvoCobwebTree` class, which
extends Cobweb3 to support image attributes (2D ndarrays).
"""
from math import sqrt
from math import pi
import numpy as np

from concept_formation.cobweb3 import Cobweb3Tree
from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.cobweb3 import cv_key


class ConvoCobwebSubTreeNode(Cobweb3Node):
    def get_root_sub(self):
        curr = self
        while curr.parent:
            if curr.parent.parent is None:
                break
            curr = curr.parent
        return 'Concept-{}'.format(curr.concept_id)

    def expected_correct_guesses(self):
        """
        Modified to handle attribute values that are concepts.
        """
        correct_guesses = 0.0
        attr_count = 0

        for attr in self.attrs():
            attr_count += 1

            concept_vals = {}

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
                elif isinstance(val, ConvoCobwebSubTreeNode):
                    root_sub = val.get_root_sub()
                    if root_sub not in concept_vals:
                        concept_vals[val.get_root_sub()] = 0
                    concept_vals[
                        val.get_root_sub()] += self.av_counts[attr][val]

                else:
                    prob = (self.av_counts[attr][val]) / self.count
                    correct_guesses += (prob * prob)

            for val in concept_vals:
                prob = concept_vals[val] / self.count
                correct_guesses += (prob * prob)

        return correct_guesses / attr_count

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
        if "_guid" in self.av_counts:
            for guid in self.av_counts['_guid']:
                output['guid'] = guid
        output["name"] = "Concept" + str(self.concept_id)
        output["size"] = self.count
        output["children"] = []

        temp = {}
        for attr in self.attrs('all'):
            temp[str(attr)] = {}

            concept_vals = {}

            for val in self.av_counts[attr]:
                if val == cv_key:
                    json_str = self.av_counts[attr][val].output_json()
                    temp[str(attr)][cv_key] = json_str
                elif isinstance(val, ConvoCobwebSubTreeNode):
                    root_sub = val.get_root_sub()
                    if root_sub not in concept_vals:
                        concept_vals[val.get_root_sub()] = 0
                    concept_vals[
                        val.get_root_sub()] += self.av_counts[attr][val]
                else:
                    temp[str(attr)][str(val)] = self.av_counts[attr][val]

            for val in concept_vals:
                temp[str(attr)][val] = concept_vals[val]

        if self.children:
            temp['_category utility'] = {}
            temp['_category utility']["#ContinuousValue#"] = {
                'mean': self.category_utility(), 'std': 1, 'n': 1}

            temp['_expected_correct_guesses'] = {}
            temp['_expected_correct_guesses']["#ContinuousValue#"] = {
                'mean': self.expected_correct_guesses(), 'std': 1, 'n': 1}

        # temp['_corter_and_gluck_category utility'] = {}
        # temp['_corter_and_gluck_category utility']["#ContinuousValue#"] = {
        #     'mean': self.corter_and_gluck_category_utility(), 'std': 1, 'n':
        #     1}

        # temp['_binary_category utility'] = {}
        # temp['_binary_category utility']["#ContinuousValue#"] = {
        #     'mean': self.binary_category_utility(), 'std': 1, 'n': 1}

        for child in self.children:
            output["children"].append(child.output_json())

        output["counts"] = temp

        return output


class ConvoCobwebSubTree(Cobweb3Tree):

    def __init__(self, scaling=0.5, inner_attr_scaling=True):
        super().__init__(scaling=scaling,
                         inner_attr_scaling=inner_attr_scaling)
        self.root = ConvoCobwebSubTreeNode()

    def clear(self):
        super().clear()
        self.root = ConvoCobwebSubTreeNode()


class ConvoCobwebTree(ConvoCobwebSubTree):
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

    def __init__(self, filter_size=(3, 3), stride=(1, 2), scaling=0.5,
                 inner_attr_scaling=True):
        """
        The tree constructor.
        """
        super().__init__(scaling=0.5, inner_attr_scaling=True)
        self.filter_size = filter_size
        self.stride = stride
        self.trees = {}
        # self.tree_class = Cobweb3Tree
        self.tree_class = ConvoCobwebSubTree

    def clear(self):
        """
        Clear the tree but keep initialization parameters
        """
        super().clear()

        self.trees = {}

    def process_image(self, img, filter_size, stride, modifying):
        ret = {}
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                ret["{},{}".format(row, col)] = img[row, col]
        level = 0

        # while np.sqrt(len(ret.keys())) > i:
        for i, j in zip(filter_size, stride):
            ret = self.apply_filter(ret, filter_size=i, stride=j,
                                    level=level, modifying=modifying)
            level += 1
        return ret

    def apply_filter(self, img_dict, filter_size, stride, level, modifying):
        # TODO update to use stride appropriately

        ret = {}

        size = int(np.sqrt(len(img_dict.keys())))
        # print('applying filter to size {}'.format(size))
        # print(img_dict.keys())

        for row in range(0, size - filter_size + 1, stride):
            for col in range(0, size - filter_size + 1, stride):
                new = {}
                for fr in range(filter_size):
                    for fc in range(filter_size):
                        new["{},{}".format(fr, fc)] = img_dict[
                            "{},{}".format(row+fr, col+fc)]

                if level not in self.trees:
                    self.trees[level] = self.tree_class(
                        scaling=self.scaling,
                        inner_attr_scaling=self.inner_attr_scaling)

                if modifying:
                    curr = self.trees[level].ifit(new)
                else:
                    curr = self.trees[level].categorize(new)

                assert not curr.children

                # while curr.parent:
                #     if curr.parent.parent is None:
                #         break
                #     curr = curr.parent

                # ret["{},{}".format(
                #     row//stride, col//stride)] = "Concept-{}".format(
                #         curr.concept_id)

                ret["{},{}".format(row//stride, col//stride)] = curr

        return ret

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
        for attr in instance:
            if isinstance(instance[attr], np.ndarray):
                temp = self.process_image(instance[attr],
                                          filter_size=self.filter_size,
                                          stride=self.stride, modifying=True)
                break

        for attr in instance:
            if not isinstance(instance[attr], np.ndarray):
                temp[attr] = instance[attr]

        return super().ifit(temp)

    def categorize(self, instance):
        """
        convo cobweb categorize without modifying.
        """
        for attr in instance:
            if isinstance(instance[attr], np.ndarray):
                temp = self.process_image(instance[attr],
                                          filter_size=self.filter_size,
                                          stride=self.stride, modifying=False)
                break

        for attr in instance:
            if not isinstance(instance[attr], np.ndarray):
                temp[attr] = instance[attr]

        return super().categorize(temp)

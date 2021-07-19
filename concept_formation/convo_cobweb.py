"""
The ConvoCobweb module contains the :class:`ConvoCobwebTree` class, which
extends Cobweb3 to support image attributes (2D ndarrays).
"""
import numpy as np

from concept_formation.cobweb3 import Cobweb3Tree


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
        super().__init__(scaling=0.5, inner_attr_scaling=True)

        self.filter_size = filter_size
        self.stride = stride
        self.trees = {}

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
        while np.sqrt(len(ret.keys())) > filter_size:
            ret = self.apply_filter(ret, filter_size, stride,
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
                    self.trees[level] = Cobweb3Tree(
                        scaling=self.scaling,
                        inner_attr_scaling=self.inner_attr_scaling)

                if modifying:
                    curr = self.trees[level].ifit(new)
                else:
                    curr = self.trees[level].categorize(new)

                while curr.parent:
                    if curr.parent.parent is None:
                        break
                    curr = curr.parent

                ret["{},{}".format(
                    row//stride, col//stride)] = "Concept-{}".format(
                        curr.concept_id)

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

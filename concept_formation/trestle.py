from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from concept_formation.utils import weighted_choice
from concept_formation.cobweb3 import Cobweb3Tree
from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.structure_mapper import StructureMapper


class TrestleTree(Cobweb3Tree):
    """

    The TrestleTree contains the knoweldge base of a partiucluar instance of the
    Trestle algorithm and can be used to fit and categorize instances. Trestle
    adds the ability to handle component attributes as well as relations in
    addition to the numerical and nominal attributes of Cobweb and Cobweb/3.

    Attributes are interpreted in the following ways:

    * Component - ``isinstance(instance[attr],dict) == True``
    * Relation - ``isinstance(instance[attr],list) == True``
    * Numeric - ``isinstance(instance[attr],Number) == True``
    * Nominal - everything else, though the assumption is ``isinstance(instance[attr],str) == True``
    """

    def __init__(self, alpha=0.001, scaling=True):
        """The tree constructor. 

        The alpha parameter is the parameter used for laplacian smoothing. The
        higher the value, the higher the prior that all attributes/values are
        equally likely. By default a minor smoothing is used: 0.001.

        The scaling parameter determines whether online normalization of
        continuous attributes is used. By default scaling is used. Scaling
        divides the std of each attribute by the std of the attribute in the
        root node. Scaling is useful to balance the weight of different
        numerical attributes, without scaling the magnitude of numerical
        attributes can affect category utility calculation meaning numbers that
        are naturally larger will recieve extra weight in the calculation.

        :param alpha: constant to use for laplacian smoothing.
        :type alpha: float
        :param scaling: whether or not numerical values should be scaled in online normalization.
        :type scaling: bool
        """
        self.root = Cobweb3Node()
        self.root.tree = self
        self.alpha = alpha
        self.scaling = scaling

    def clear(self):
        """Clears the concepts of the tree, but maintains the alpha  and
        scaling parameters.
        """
        self.root = Cobweb3Node()
        self.root.tree = self

    def ifit(self, instance):
        """Incrementally fit a new instance into the tree and return its resulting
        concept

        The instance is passed down the tree and updates each node to
        incorporate the instance. This process modifies the trees knowledge for
        a non-modifying version of labeling use the categorize() function.

        This version is modified from the normal :meth:`CobwebTree.ifit
        <concept_formation.cobweb.CobwebTree.ifit>` by first structur mapping
        the instance before fitting it into the knoweldge base.
        
        :param instance: an instance to be categorized into the tree.
        :type instance: {a1:v1, a2:v2, ...}
        :return: A concept describing the instance
        :rtype: Cobweb3Node

        .. note:: this modifies the tree's knoweldge.
        .. seealso:: :meth:`TrestleTree.trestle`
        """
        return self.trestle(instance)

    def _trestle_categorize(self, instance):
        """
        The Trestle categorize function, this Trestle categorizes all the
        sub-components before categorizing itself.
        """
        structure_mapper = StructureMapper(self.root)
        temp_instance = structure_mapper.transform(instance)
        return self._cobweb_categorize(temp_instance)

    def complete_instance(self, instance, choice_fn=weighted_choice):
        """
        Given a tree and an instance, returns a new instance with attribute 
        values picked using hte choice_fn.

        :param instance: an instance to be completed.
        :type instance: {a1: v1, a2: v2, ...}
        :param choice_fn: A function for deciding which attribute/value to
            chose. The default is: concept_formation.utils.weighted_choice. The
            other option is: concept_formation.utils.most_likely_choice.
        :type choice_fn: a python function
        :type instance: {a1: v1, a2: v2, ...}
        :return: A completed instance
        :rtype: instance
        """
        structure_mapper = StructureMapper(self.root)
        temp_instance = structure_mapper.transform(instance)
        concept = self._cobweb_categorize(temp_instance)

        for attr in concept.av_counts:
            if attr in temp_instance:
                continue

            missing_prob = concept.get_probability_missing(attr)
            attr_choices = ((None, missing_prob), (attr, 1 - missing_prob))
            if choice_fn(attr_choices) == attr:
                temp_instance[attr] = choice_fn(concept.get_weighted_values(attr))

        return structure_mapper.undo_transform(temp_instance)

    def categorize(self, instance):
        """Sort an instance in the categorization tree and return its resulting
        concept.

        The instance is passed down the the categorization tree according to the
        normal cobweb algorithm except using only the new and best opperators
        and without modifying nodes' probability tables.

        This version differs fomr the normal :meth:`CobwebTree.categorize
        <concept_formation.cobweb.CobwebTree.categorize>` and
        :meth:`Cobweb3Tree.categorize
        <concept_formation.cobweb3.Cobweb3Tree.categorize>` by structure mapping
        instances before categorizing them.

        :param instance: an instance to be categorized into the tree.
        :type instance: {a1:v1, a2:v2, ...}
        :return: A concept describing the instance
        :rtype: CobwebNode

        .. note:: this does not modify the tree's knoweldge.
        .. seealso:: :meth:`TrestleTree.trestle`
        """
        return self._trestle_categorize(instance)

    def trestle(self, instance):
        """The core trestle algorithm used in fitting and categorization

        This function is similar to :meth:`Cobweb.cobweb
        <concept_formation.cobweb.CobwebTree.cobweb>` The key difference between
        trestle and cobweb is that trestle performs structure mapping (see:
        :meth:`structure_map
        <concept_formation.structure_mapper.structure_map>`) before proceeding
        through the normal cobweb algorithm.
        """
        structure_mapper = StructureMapper(self.root)
        temp_instance = structure_mapper.transform(instance)
        return self.cobweb(temp_instance)

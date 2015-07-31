from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from concept_formation.utils import weighted_choice
from concept_formation.cobweb3 import Cobweb3Tree
from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.cobweb3 import ContinuousValue
from concept_formation.structure_mapper import StructureMapper

class TrestleTree(Cobweb3Tree):
    """
    The TrestleTree instantiates the Trestle algorithm, which can
    be used to learn from and categorize instances. Trestle adds the ability to
    handle component attributes as well as relations in addition to the
    numerical and nominal attributes of Cobweb and Cobweb/3.

    The alpha parameter is the parameter used for laplacian smoothing of
    nominal values (or whether an attribute is present or not for both
    nominal and numeric attributes). The higher the value, the higher the
    prior that all attributes/values are equally likely. By default a minor
    smoothing is used: 0.001.
        
    The scaling parameter determines whether online normalization of
    continuous attributes is used. By default scaling is used. Scaling
    divides the std of each attribute by the std of the attribute in the
    parent node (no scaling is performed in the root). Scaling is useful to
    balance the weight of different numerical attributes, without scaling
    the magnitude of numerical attributes can affect category utility
    calculation meaning numbers that are naturally larger will recieve
    extra weight in the calculation.

    The beam width parameter detemines the inital beam width used by Beam
    Search to perform structure mapping.  A smaller beam width results in a
    faster search but is not gauranteed to find an optimal match. If beam width
    is set to ``float('inf')`` then A* search will be used, but typically this is
    prohibitively slow.

    The vars_only parameter determines whether the matcher should only allow
    variable attributes to match to other variable attributes or if variable
    attributes should also be allowed to match to constant attributes. This
    setting will generally depend on the domain of the data and whether
    variables mapping to constant attributes makes sense. Allowing the match to
    constant attributes also increases the search space taking more time to find
    matches.

    :param alpha: constant to use for laplacian smoothing.
    :type alpha: float
    :param scaling: whether or not numerical values should be scaled in
        online normalization.
    :type scaling: bool
    :param beam_width: the initial beam width to use in structure mapping's
        search step.
    :type beam_width: int
    :param vars_only: whether matching should be performed only between variable
        or if variables can be matched to constant values.
    :type vars_only: bool
    """

    def __init__(self, alpha=0.001, scaling=True, beam_width=2, vars_only=True):
        """
        The tree constructor. 

        .. todo:: Need to test scaling by 1 std vs. 2 std. It might be
        preferrable to standardize by 2 std because that gives it the same
        variance as a nominal value. 
        """
        self.root = Cobweb3Node()
        self.root.tree = self
        self.alpha = alpha
        self.scaling = scaling
        self.std_to_scale = 1.0
        self.beam_width = beam_width
        self.vars_only = vars_only

    def clear(self):
        """
        Clears the concepts stored in the tree, but maintains the alpha and
        scaling parameters.
        """
        self.root = Cobweb3Node()
        self.root.tree = self

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
        :type instance: {a1:v1, a2:v2, ...}
        :return: A concept describing the instance
        :rtype: Cobweb3Node

        .. seealso:: :meth:`TrestleTree.trestle`
        """
        return self.trestle(instance)

    def _trestle_categorize(self, instance):
        """
        The structure maps the instance, categorizes the matched instance, and
        returns the resulting Cobweb3Node.

        :param instance: an instance to be categorized into the tree.
        :type instance: {a1:v1, a2:v2, ...}
        :return: A concept describing the instance
        :rtype: Cobweb3Node
        """
        structure_mapper = StructureMapper(self.root,
                                           beam_width=self.beam_width,
                                           vars_only=self.vars_only)
        temp_instance = structure_mapper.transform(instance)
        return self._cobweb_categorize(temp_instance)

    def infer_missing(self, instance, choice_fn="most likely"):
        """
        Given a tree and an instance, returns a new instance with attribute 
        values picked using the specified choice function (either "most likely"
        or "sampled"). 

        :param instance: an instance to be completed.
        :type instance: {a1: v1, a2: v2, ...}
        :param choice_fn: a string specifying the choice function to use,
            either "most likely" or "sampled". 
        :type choice_fn: a string
        :return: A completed instance
        :rtype: instance
        """
        structure_mapper = StructureMapper(self.root,
                                           beam_width=self.beam_width,
                                           vars_only=self.vars_only)
        temp_instance = structure_mapper.transform(instance)
        temp_instance = super(TrestleTree, self).infer_missing(temp_instance, choice_fn)
        return structure_mapper.undo_transform(temp_instance)

    def categorize(self, instance):
        """
        Sort an instance in the categorization tree and return its resulting
        concept.

        The instance is passed down the the categorization tree according to the
        normal cobweb algorithm except using only the new and best opperators
        and without modifying nodes' probability tables. **This does not modify
        the tree's knowledge base** for a modifying version see
        :meth:`TrestleTree.ifit`

        This version differs fomr the normal :meth:`CobwebTree.categorize
        <concept_formation.cobweb.CobwebTree.categorize>` and
        :meth:`Cobweb3Tree.categorize
        <concept_formation.cobweb3.Cobweb3Tree.categorize>` by structure mapping
        instances before categorizing them.

        :param instance: an instance to be categorized into the tree.
        :type instance: {a1:v1, a2:v2, ...}
        :return: A concept describing the instance
        :rtype: CobwebNode

        .. seealso:: :meth:`TrestleTree.trestle`
        """
        return self._trestle_categorize(instance)

    def trestle(self, instance):
        """
        The core trestle algorithm used in fitting and categorization.

        This function is similar to :meth:`Cobweb.cobweb
        <concept_formation.cobweb.CobwebTree.cobweb>` The key difference
        between trestle and cobweb is that trestle performs structure mapping
        (see: :meth:`structure_map
        <concept_formation.structure_mapper.StructureMapper.transform>`) before proceeding
        through the normal cobweb algorithm.

        :param instance: an instance to be categorized into the tree.
        :type instance: {a1:v1, a2:v2, ...}
        :return: A concept describing the instance
        :rtype: CobwebNode
        """
        structure_mapper = StructureMapper(self.root,
                                           beam_width=self.beam_width,
                                           vars_only=self.vars_only)
        temp_instance = structure_mapper.transform(instance)
        return self.cobweb(temp_instance)

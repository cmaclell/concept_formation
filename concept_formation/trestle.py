from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from concept_formation.cobweb3 import Cobweb3Tree
from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.structure_mapper import structure_map

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
        self.r
        self.root = Cobweb3Node()
        self.root.root = self.root
        self.root.alpha = alpha
        self.root.scaling = scaling

    def ifit(self, instance):
        """
        A modification of ifit to call Trestle instead.
        """
        return self.trestle(instance)

    def _trestle_categorize(self, instance):
        """
        The Trestle categorize function, this Trestle categorizes all the
        sub-components before categorizing itself.
        """
        temp_instance = structure_map(self.root, instance)
        return self.cobweb_categorize(temp_instance)

    def categorize(self, instance):
        """
        A categorize function that can be used polymorphicaly without 
        having to worry about the type of the underlying object.

        In Trestle's case this calls _trestle_categorize()
        """
        return self._trestle_categorize(instance)

    def trestle(self, instance):
        """The code trestle algorithm used in fitting and categorization

        This function is similar to :meth:`Cobweb.cobweb
        <concept_formation.cobweb.CobwebTree.cobweb>` The key difference between
        trestle and cobweb is that trestle performs structure mapping (see:
        :meth:`structure_map
        <concept_formation.structure_mapper.structure_map>`) before proceeding
        through the normal cobweb algorithm.
        """
        temp_instance = structure_map(self.root, instance)
        return self.cobweb(temp_instance)

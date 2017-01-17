"""
The Trestle module contains the :class:`TrestleTree` class, which extends
Cobweb3 to support component and relational attributes.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from concept_formation.cobweb3 import Cobweb3Tree
from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.structure_mapper import StructureMapper
from concept_formation.preprocessor import SubComponentProcessor
from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Pipeline
from concept_formation.preprocessor import NameStandardizer


class TrestleTree(Cobweb3Tree):
    """
    The TrestleTree instantiates the Trestle algorithm, which can
    be used to learn from and categorize instances. Trestle adds the ability to
    handle component attributes as well as relations in addition to the
    numerical and nominal attributes of Cobweb and Cobweb/3.

    The scaling parameter determines whether online normalization of continuous
    attributes is used, and to what standard deviation the values are scaled
    to. Scaling divides the std of each attribute by the std of the attribute
    in the root divided by the scaling constant (i.e.,
    :math:`\\sigma_{root} / scaling` when making category utility calculations.
    Scaling is useful to balance the weight of different numerical attributes,
    without scaling the magnitude of numerical attributes can affect category
    utility calculation meaning numbers that are naturally larger will recieve
    preference in the category utility calculation.

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

    def __init__(self, scaling=0.5, inner_attr_scaling=True):
        """
        The tree constructor.
        """
        self.gensym_counter = 0
        self.root = Cobweb3Node()
        self.root.tree = self
        self.scaling = scaling
        self.inner_attr_scaling = inner_attr_scaling
        self.attr_scales = {}

    def clear(self):
        """
        Clear the tree but keep initialization parameters
        """
        self.gensym_counter = 0
        self.root = Cobweb3Node()
        self.root.tree = self
        self.attr_scales = {}

    def gensym(self):
        """
        Generates unique names for naming renaming apart objects.

        :return: a unique object name
        :rtype: '?o'+counter
        """
        self.gensym_counter += 1
        return '?o' + str(self.gensym_counter)

    def _sanity_check_instance(self, instance):
        """
        Checks the attributes of an instance to ensure they are properly
        subscriptable types and throws an excpetion if they are not.
        Lots of sub-processes in the structure mapper freak out if you have
        non-str non-tuple attributes so I decided it was best to do a one
        time check at the first call to transform.
        """
        for attr in instance:
            try:
                hash(attr)
                attr[0]
            except:
                raise ValueError('Invalid attribute: '+str(attr) +
                                 ' of type: ' + str(type(attr)) +
                                 ' in instance: ' + str(instance) +
                                 ',\n' + type(self).__name__ +
                                 ' only works with hashable and' +
                                 ' subscriptable attributes (e.g., strings).')
            if isinstance(attr, tuple):
                self._sanity_check_relation(attr, instance)
            if isinstance(instance[attr], dict):
                self._sanity_check_instance(instance[attr])
            else:
                try:
                    hash(instance[attr])
                except:
                    raise ValueError('Invalid value: ' + str(instance[attr]) +
                                     ' of type: ' + str(type(instance[attr])) +
                                     ' in instance: ' + str(instance) +
                                     ',\n' + type(self).__name__ +
                                     ' only works with hashable values.')

    def _sanity_check_relation(self, relation, instance):
        for v in relation:
            try:
                v[0]
            except:
                raise(ValueError('Invalid relation value: ' + str(v) +
                                 ' of type: ' + str(type(v)) +
                                 ' in instance: ' + str(instance) +
                                 ',\n' + type(self).__name__ +
                                 'requires that values inside relation' +
                                 ' tuples be of type str or tuple.'))
            if isinstance(v, tuple):
                self._sanity_check_relation(v, instance)

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
        return self.trestle(instance)

    def _trestle_categorize(self, instance):
        """
        The structure maps the instance, categorizes the matched instance, and
        returns the resulting concept.

        :param instance: an instance to be categorized into the tree.
        :type instance: {a1:v1, a2:v2, ...}
        :return: A concept describing the instance
        :rtype: concept
        """
        preprocessing = Pipeline(NameStandardizer(self.gensym),
                                 Flattener(), SubComponentProcessor(),
                                 StructureMapper(self.root))
        temp_instance = preprocessing.transform(instance)
        self._sanity_check_instance(temp_instance)
        return self._cobweb_categorize(temp_instance)

    def infer_missing(self, instance, choice_fn="most likely",
                      allow_none=True):
        """
        Given a tree and an instance, returns a new instance with attribute
        values picked using the specified choice function (either "most likely"
        or "sampled").

        .. todo:: write some kind of test for this.

        :param instance: an instance to be completed.
        :type instance: :ref:`Instance<instance-rep>`
        :param choice_fn: a string specifying the choice function to use,
            either "most likely" or "sampled".
        :type choice_fn: a string
        :param allow_none: whether attributes not in the instance can be
            inferred to be missing. If False, then all attributes will be
            inferred with some value.
        :type allow_none: Boolean
        :return: A completed instance
        :rtype: instance
        """
        preprocessing = Pipeline(NameStandardizer(self.gensym),
                                 Flattener(), SubComponentProcessor(),
                                 StructureMapper(self.root))

        temp_instance = preprocessing.transform(instance)
        concept = self._cobweb_categorize(temp_instance)

        for attr in concept.attrs('all'):
            if attr in temp_instance:
                continue
            val = concept.predict(attr, choice_fn, allow_none)
            if val is not None:
                temp_instance[attr] = val

        temp_instance = preprocessing.undo_transform(temp_instance)
        return temp_instance

    def categorize(self, instance):
        """
        Sort an instance in the categorization tree and return its resulting
        concept.

        The instance is passed down the the categorization tree according to
        the normal cobweb algorithm except using only the new and best
        opperators and without modifying nodes' probability tables. **This does
        not modify the tree's knowledge base** for a modifying version see
        :meth:`TrestleTree.ifit`

        This version differs fomr the normal :meth:`CobwebTree.categorize
        <concept_formation.cobweb.CobwebTree.categorize>` and
        :meth:`Cobweb3Tree.categorize
        <concept_formation.cobweb3.Cobweb3Tree.categorize>` by structure
        mapping instances before categorizing them.

        :param instance: an instance to be categorized into the tree.
        :type instance: :ref:`Instance<instance-rep>`
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
        <concept_formation.structure_mapper.StructureMapper.transform>`) before
        proceeding through the normal cobweb algorithm.

        :param instance: an instance to be categorized into the tree.
        :type instance: :ref:`Instance<instance-rep>`
        :return: A concept describing the instance
        :rtype: CobwebNode
        """
        preprocessing = Pipeline(NameStandardizer(self.gensym),
                                 Flattener(), SubComponentProcessor(),
                                 StructureMapper(self.root))
        temp_instance = preprocessing.transform(instance)
        self._sanity_check_instance(temp_instance)
        return self.cobweb(temp_instance)

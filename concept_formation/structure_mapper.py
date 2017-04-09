"""
This module contains the
:class:`StructureMapper<concept_formation.structure_mapper.StructureMapper>`
class which is used rename variable attributes it improve the category utility
on instances.

It is an instance of a
:class:`preprocessor<concept_formation.preprocessor.Preprocessor>` with a
:func:`transform` and :func:`undo_tranform` methods.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from random import choice
from random import random
from itertools import combinations

from munkres import Munkres
# from scipy.optimize import linear_sum_assignment

from py_search.base import Problem
from py_search.base import Node
from py_search.optimization import hill_climbing
from concept_formation.preprocessor import Preprocessor
from concept_formation.preprocessor import rename_relation
from concept_formation.preprocessor import get_attribute_components
from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.cobweb3 import cv_key


def get_component_names(instance, vars_only=True):
    """
    Given an instance or a concept's probability table return a list of all of
    the component names. If vars_only is false, than all constants and
    variables are returned.

    :param instance: An instance or a concept's probability table.
    :type instance: an instance
    :param vars_only: Whether or not to return only variables (i.e., strings
        with a names with a '?' at the beginning) or both variables and
        constants.
    :type vars_only: boolean
    :return: A frozenset of all of the component names present in the instance
    :rtype: frozenset

    >>> instance = {('a', ('sub1', 'c1')): 0, ('a', 'c2'): 0,
    ...             ('_', '_a', 'c3'): 0}
    >>> names = get_component_names(instance, False)
    >>> frozenset(names) == frozenset({'c3', 'c2', ('sub1', 'c1'), 'sub1', 'a',
    ...                               ('a', ('sub1', 'c1')), ('a', 'c2'),
    ...                                'c1'})
    True
    >>> names = get_component_names(instance, True)
    >>> frozenset(names) == frozenset()
    True

    >>> instance = {('relation1', ('sub1', 'c1'), 'o3'): True}
    >>> names = get_component_names(instance, False)
    >>> frozenset(names) == frozenset({'o3', ('relation1', ('sub1', 'c1'),
    ...                                       'o3'), 'sub1', ('sub1', 'c1'),
    ...                                'c1', 'relation1'})
    True
    """
    names = set()
    for attr in instance:
        for name in get_attribute_components(attr, vars_only):
            names.add(name)
    return names


def rename_flat(target, mapping):
    """
    Given an instance and a mapping rename the components and relations and
    return the renamed instance.

    :param instance: An instance to be renamed according to a mapping
    :type instance: instance
    :param mapping: :param mapping: A dictionary of mappings between component
        names
    :type mapping: dict
    :return: A copy of the instance with components and relations renamed
    :rtype: instance

    >>> import pprint
    >>> instance = {('a', '?c1'): 1, ('good', '?c1'): True}
    >>> mapping = {'?c1': '?o1'}
    >>> renamed = rename_flat(instance,mapping)
    >>> pprint.pprint(renamed)
    {('a', '?o1'): 1, ('good', '?o1'): True}
    """
    temp_instance = {}

    for attr in target:
        if attr in mapping:
            temp_instance[mapping[attr]] = target[attr]
        elif isinstance(attr, tuple):
            temp_instance[rename_relation(attr, mapping)] = target[attr]
        else:
            temp_instance[attr] = target[attr]

    return temp_instance


def bind_flat_attr(attr, mapping):
    """
    Renames an attribute given a mapping.

    :param attr: The attribute to be renamed
    :type attr: str or tuple
    :param mapping: A dictionary of mappings between component names
    :type mapping: dict
    :param unnamed: A list of components that are not yet mapped.
    :type unnamed: dict
    :return: The attribute's new name or ``None`` if the mapping is incomplete
    :rtype: str

    >>> attr = ('before', '?c1', '?c2')
    >>> mapping = {'?c1': '?o1', '?c2':'?o2'}
    >>> bind_flat_attr(attr, mapping)
    ('before', '?o1', '?o2')

    >>> attr = ('ordered-list', ('cells', '?obj12'), '?obj10', '?obj11')
    >>> mapping = {'?obj12': '?o1', '?obj10':'?o2', '?obj11': '?o3'}
    >>> bind_flat_attr(attr, mapping)
    ('ordered-list', ('cells', '?o1'), '?o2', '?o3')

    If the mapping is incomplete then returns partially mapped attributes

    >>> attr = ('before', '?c1', '?c2')
    >>> mapping = {'?c1': 'o1'}
    >>> bind_flat_attr(attr, mapping)
    ('before', 'o1', '?c2')

    >>> bind_flat_attr(('<', ('a', '?o2'), ('a', '?o1')), {'?o1': '?c1'})
    ('<', ('a', '?o2'), ('a', '?c1'))

    >>> bind_flat_attr(('<', ('a', '?o2'), ('a', '?o1')),
    ...                {'?o1': '?c1', '?o2': '?c2'})
    ('<', ('a', '?c2'), ('a', '?c1'))
    """
    return tuple([bind_flat_attr(ele, mapping) if isinstance(ele, tuple)
                  else mapping[ele] if ele in mapping else ele for ele in
                  attr])


def contains_component(component, attr):
    """
    Return ``True`` if the given component name is in the attribute, either as
    part of a hierarchical name or within a relations otherwise ``False``.

    :param component: A component name
    :type component: str
    :param attr: An attribute name
    :type atte: str
    :return: ``True`` if the component name exists inside the attribute name
             ``False`` otherwise
    :rtype: bool

    >>> contains_component('?c1', ('relation', '?c2', ('a', '?c1')))
    True
    >>> contains_component('?c3', ('before', '?c1', '?c2'))
    False
    """
    if isinstance(attr, tuple):
        for ele in attr:
            if contains_component(component, ele) is True:
                return True
    return attr == component


def flat_match(target, base, initial_mapping=None):
    """
    Given a base (usually concept) and target (instance or concept av table)
    this function returns a mapping that can be used to rename components in
    the target. Search is used to find a mapping that maximizes the expected
    number of correct guesses in the concept after incorporating the instance.

    The current approach is to refine the initially provided mapping using a
    local hill-climbing search. If no initial mapping is provided then one is
    generated using the Munkres / Hungarian matching on object-to-object
    assignment (no relations). This initialization approach is polynomial in
    the size of the base.

    :param target: An instance or concept.av_counts object to be mapped to the
        base concept.
    :type target: :ref:`Instance<instance-rep>` or av_counts obj from concept
    :param base: A concept to map the target to
    :type base: TrestleNode
    :param initial_mapping: An initial mapping to seed the local search
    :type initial_mapping: A mapping dict
    :return: a mapping for renaming components in the instance.
    :rtype: dict
    """
    inames = frozenset(get_component_names(target))
    cnames = frozenset(get_component_names(base.av_counts))

    if(len(inames) == 0 or len(cnames) == 0):
        return {}

    if len(inames.intersection(cnames)) > 0:
        raise Exception("Objects in target and base must not collide. "
                        "Consider running NameStandardizer first.")

    # TODO consider flipping target and base when one is larger than the other.
    if initial_mapping is None:
        initial_mapping = hungarian_mapping(inames, cnames, target, base)
    else:
        initial_mapping = frozenset([(a, v) for a, v in initial_mapping if a in
                                     inames and v in cnames])

    unmapped = cnames - frozenset(dict(initial_mapping).values())

    # print("MATCHING", initial_mapping, target, base)

    initial_cost = mapping_cost(initial_mapping, target, base)

    op_problem = StructureMappingOptimizationProblem((initial_mapping,
                                                      unmapped),
                                                     initial_cost=initial_cost,
                                                     extra=(target, base))

    solution = next(hill_climbing(op_problem))
    return dict(solution.state[0])


def hungarian_mapping(inames, cnames, target, base):
    """
    Utilizes the hungarian/munkres matching algorithm to compute an initial
    mapping of inames to cnames. The base cost is the expected correct guesses
    if each object is matched to itself (i.e., a new object). Then the cost of
    each object-object match is evaluated by setting each individual object and
    computing the expected correct guesses.

    :param inames: the target component names
    :type inames: collection
    :param cnames: the base component names
    :type cnames: collection
    :param target: An instance or concept.av_counts object to be mapped to the
        base concept.
    :type target: :ref:`Instance<instance-rep>` or av_counts obj from concept
    :param base: A concept to map the target to
    :type base: TrestleNode
    :return: a mapping for renaming components in the instance.
    :rtype: frozenset

    """
    cnames = list(cnames)
    inames = list(inames)

    cost_matrix = []
    for o in inames:
        row = []
        for c in cnames:
            nm = {}
            nm[o] = c
            cost = mapping_cost({o: c}, target, base)
            row.append(cost)
        unmapped_cost = mapping_cost({}, target, base)
        for other_o in inames:
            if other_o == o:
                row.append(unmapped_cost)
            else:
                row.append(float('inf'))
        cost_matrix.append(row)

    m = Munkres()
    indices = m.compute(cost_matrix)

    # comments for using scipy hungarian
    # indices = linear_sum_assignment(cost_matrix)

    mapping = {}

    # for i in range(len(indices[0])):
    #     row = indices[0][i]
    #     col = indices[1][i]

    for row, col in indices:
        if col >= len(cnames):
            mapping[inames[row]] = inames[row]
        else:
            mapping[inames[row]] = cnames[col]

    return frozenset(mapping.items())


def mapping_cost(mapping, target, base):
    """
    Used to evaluate a mapping between a target and a base. This is performed
    by renaming the target using the mapping, adding it to the base and
    evaluating the expected number of correct guesses in the newly updated
    concept.

    :param mapping: the mapping of target items to base items
    :type mapping: frozenset or dict
    :param target: the target
    :type target: an instance or concept.av_counts
    :param base: the base
    :type base: a concept
    """
    if isinstance(mapping, frozenset):
        mapping = dict(mapping)
    if not isinstance(mapping, dict):
        raise Exception("mapping must be dict or frozenset")
    renamed_target = rename_flat(target, mapping)

    # Need to ensure structure mapping is not used internally here.
    # (i.e., there is no infinite recrusion)
    temp_base = Cobweb3Node()
    temp_base.update_counts_from_node(base)
    temp_base.tree = base.tree

    # check if it is an av_counts table, then create concept to deal with it.
    if isinstance(next(iter(renamed_target.values())), dict):
        temp_target = Cobweb3Node()
        temp_target.av_counts = renamed_target
        temp_target.count = max([sum([renamed_target[attr][val].num if val ==
                                      cv_key else renamed_target[attr][val] for
                                      val in renamed_target[attr]]) for attr in
                                 renamed_target])
        temp_base.update_counts_from_node(temp_target)
    else:
        temp_base.increment_counts(renamed_target)

    return -temp_base.expected_correct_guesses()


class StructureMappingOptimizationProblem(Problem):
    """
    A class for describing a structure mapping problem to be solved using the
    `py_search <http://py-search.readthedocs.io/>`_ library. This class defines
    the node_value, the successor, and goal_test methods used by the search
    library.

    Unlike StructureMappingProblem, this class uses a local search approach;
    i.e., given an initial mapping it tries to improve the mapping by permuting
    it.
    """
    def goal_test(self, node):
        """
        This should always return False, so it never terminates early.
        """
        return False

    def node_value(self, node):
        """
        The value of a node (based on mapping_cost).
        """
        # return node.cost()
        mapping, unmapped_cnames = node.state
        target, base = node.extra
        return mapping_cost(mapping, target, base)

    def swap_two(self, o1, o2, mapping, unmapped_cnames, target, base, node):
        """
        returns the child node generated from swapping two mappings.
        """
        new_mapping = {a: mapping[a] for a in mapping}

        if mapping[o2] == o2:
            new_mapping[o1] = o1
        else:
            new_mapping[o1] = mapping[o2]

        if mapping[o1] == o1:
            new_mapping[o2] = o2
        else:
            new_mapping[o2] = mapping[o1]

        new_mapping = frozenset(new_mapping.items())
        return Node((new_mapping, unmapped_cnames), extra=node.extra)

    def swap_unnamed(self, o1, o2, mapping, unmapped_cnames, target, base,
                     node):
        """
        Returns the child node generated from assigning an unmapped component
        object to one of the instance objects.
        """
        new_mapping = {a: mapping[a] for a in mapping}
        new_unmapped_cnames = set(unmapped_cnames)
        new_unmapped_cnames.remove(o2)
        if mapping[o1] != o1:
            new_unmapped_cnames.add(new_mapping[o1])
        new_mapping[o1] = o2
        new_mapping = frozenset(new_mapping.items())

        return Node((new_mapping,
                    frozenset(new_unmapped_cnames)), extra=node.extra)

    def random_successor(self, node):
        """
        Similar to the successor function, but generates only a single random
        successor.
        """
        mapping, unmapped_cnames = node.state
        target, base = node.extra
        mapping = dict(mapping)

        o1 = choice(list(mapping))
        while mapping[o1] == o1 and len(unmapped_cnames) == 0:
            o1 = choice(list(mapping))

        possible_flips = [v for v in mapping if (v != o1 and
                                                 not (mapping[o1] == o1 or
                                                      mapping[v] == v))]

        if random() <= len(possible_flips) / (len(possible_flips) +
                                              len(unmapped_cnames)):
            o2 = choice(possible_flips)
            return self.swap_two(o1, o2, mapping, unmapped_cnames, target,
                                 base, node)
        else:
            o2 = choice(list(unmapped_cnames))
            return self.swap_unnamed(o1, o2, mapping, unmapped_cnames, target,
                                     base, node)

    def successors(self, node):
        """
        An iterator that returns all successors.
        """
        mapping, unmapped_cnames = node.state
        target, base = node.extra
        mapping = dict(mapping)

        for o1, o2 in combinations(mapping, 2):
            if o1 == o2 or (mapping[o1] == o1 and mapping[o2] == o2):
                continue

            yield self.swap_two(o1, o2, mapping, unmapped_cnames, target,
                                base, node)

        for o1 in mapping:
            for o2 in unmapped_cnames:
                yield self.swap_unnamed(o1, o2, mapping, unmapped_cnames,
                                        target, base, node)


def is_partial_match(iAttr, cAttr, mapping):
    """
    Returns True if the instance attribute (iAttr) partially matches the
    concept attribute (cAttr) given the mapping.

    :param iAttr: An attribute in an instance
    :type iAttr: str or tuple
    :param cAttr: An attribute in a concept
    :type cAttr: str or tuple
    :param mapping: A mapping between between attribute names
    :type mapping: dict
    :param unnamed: A list of components that are not yet mapped.
    :type unnamed: dict
    :return: ``True`` if the instance attribute matches the concept attribute
        in the mapping otherwise ``False``
    :rtype: bool

    >>> is_partial_match(('<', ('a', '?o2'), ('a', '?o1')),
    ...                  ('<', ('a', '?c2'), ('b', '?c1')), {'?o1': '?c1'})
    False

    >>> is_partial_match(('<', ('a', '?o2'), ('a', '?o1')),
    ...                  ('<', ('a', '?c2'), ('a', '?c1')), {'?o1': '?c1'})
    True

    >>> is_partial_match(('<', ('a', '?o2'), ('a', '?o1')),
    ...                  ('<', ('a', '?c2'), ('a', '?c1')),
    ...                  {'?o1': '?c1', '?o2': '?c2'})
    True
    """
    if type(iAttr) != type(cAttr):
        return False

    if isinstance(iAttr, tuple) and len(iAttr) != len(cAttr):
        return False

    if isinstance(iAttr, tuple):
        for i, v in enumerate(iAttr):
            if not is_partial_match(iAttr[i], cAttr[i], mapping):
                return False
        return True

    if iAttr[0] == '?' and iAttr in mapping:
        return mapping[iAttr] == cAttr

    if iAttr[0] == '?' and cAttr[0] == '?' and iAttr not in mapping:
        return True

    return iAttr == cAttr


class StructureMapper(Preprocessor):
    """
    Structure maps an instance that has been appropriately preprocessed (i.e.,
    standardized apart, flattened, subcomponent processed, and lists processed
    out). Transform renames the instance based on this structure mapping, and
    return the renamed instance.

    :param base: A concept to structure map the instance to
    :type base: TrestleNode
    :param gensym: a function that returns unique object names (str) on each
        call
    :type gensym: function
    :return: A flattened and mapped copy of the instance
    :rtype: instance
    """
    def __init__(self, base):
        self.base = base
        self.mapping = None
        self.reverse_mapping = None

    def get_mapping(self):
        """
        Returns the currently established mapping.

        :return: The current mapping.
        :rtype: dict
        """
        return {self.reverse_mapping[o]: o for o in self.reverse_mapping}

    def transform(self, target, initial_mapping=None):
        """
        Transforms a provided target (either an instance or an av_counts table
        from a CobwebNode or Cobweb3Node).

        :param target: An instance or av_counts table to rename to bring into
            alignment with the provided base.
        :type target: instance or av_counts table (from CobwebNode or
            Cobweb3Node).
        :return: The renamed instance or av_counts table
        :rtype: instance or av_counts table
        """
        self.mapping = flat_match(target, self.base, initial_mapping)
        self.reverse_mapping = {self.mapping[o]: o for o in self.mapping}
        return rename_flat(target, self.mapping)

    def undo_transform(self, target):
        """
        Takes a transformed target and reverses the structure mapping using the
        mapping discovered by transform.

        :param target: A previously renamed instance or av_counts table to
            reverse the structure mapping on.
        :type target: previously structure mapped instance or av_counts table
            (from CobwebNode or Cobweb3Node).
        :return: An instance or concept av_counts table with original object
            names
        :rtype: dict
        """
        if self.reverse_mapping is None:
            raise Exception("Must transform before undoing transform")
        return rename_flat(target, self.reverse_mapping)

"""
This module contains the
:class:`StructureMapper<concept_formation.structure_mapper.StructureMapper>`
class which is used to perform flattening (i.e., compilation of component
attributes into relational attributes) and structure mapping (i.e., renaming
variable attributes it improve the category utility) on instances.

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
from heapq import heappush
from heapq import heappop

from munkres import Munkres

from py_search.base import Problem
from py_search.base import Node
from py_search.optimization import hill_climbing
from concept_formation.preprocessor import NameStandardizer
from concept_formation.preprocessor import Preprocessor
from concept_formation.preprocessor import rename_relation
from concept_formation.preprocessor import get_attribute_components
from concept_formation.continuous_value import ContinuousValue

def get_component_names(instance, vars_only=True):
    """
    Given  an instance or a concept's probability table return a list of all of
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

    >>> instance = {('a', ('sub1', 'c1')): 0, ('a', 'c2'): 0, ('_', '_a', 'c3'): 0}
    >>> names = get_component_names(instance, False)
    >>> frozenset(names) == frozenset({'c3', 'c2', ('sub1', 'c1'), 'sub1', 'a', ('a', ('sub1', 'c1')), ('a', 'c2'), 'c1'})
    True
    >>> names = get_component_names(instance, True)
    >>> frozenset(names) == frozenset()
    True

    >>> instance = {('relation1', ('sub1', 'c1'), 'o3'): True}
    >>> names = get_component_names(instance, False)
    >>> frozenset(names) == frozenset({'o3', ('relation1', ('sub1', 'c1'), 'o3'), 'sub1', ('sub1', 'c1'), 'c1', 'relation1'})
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
    :param mapping: :param mapping: A dictionary of mappings between component names
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

    >>> bind_flat_attr(('<', ('a', '?o2'), ('a', '?o1')), {'?o1': '?c1', '?o2': '?c2'})
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
    else:
        if attr == component:
            return True
    
    return False

def build_index(onames, instance):
    """
    Given a set of object names and an instance (or av_count table), builds a
    an index (a dict) that has object names as key and a list of all relations
    that refer to that object as a value.

    :param onames: A set of objects in the instance.
    :type onames: an iterable
    :param instance: an instance or dictionary of attributes
    :type instance: :ref:`Instance<instance-rep>`
    """
    index = {}
    for o in onames:
        index[o] = []
    for attr in instance:
        for o in get_attribute_components(attr):
            index[o].append(attr)
    return index

def flat_match(target, base):
    """
    Given a concept and instance this function returns a mapping that can be
    used to rename components in the instance. Search is used to find a mapping
    that maximizes the expected number of correct guesses in the concept after
    incorporating the instance. 

    The current approach is to generate a solution via Munkres / Hungarian
    matching on object-to-object assignment (no relations). Then this
    assignment is refined by using a hill climbing search to account for the
    relations.

    :param target: An instance or concept.av_counts object to be mapped to the
        base concept.
    :type target: :ref:`Instance<instance-rep>` or av_counts obj from concept
    :param base: A concept to map the target to
    :type base: TrestleNode
    :return: a mapping for renaming components in the instance.
    :rtype: dict
    """
    inames = frozenset(get_component_names(target))
    cnames = frozenset(get_component_names(base.av_counts))

    if(len(inames) == 0 or len(cnames) == 0):
        return {}

    index = build_index(inames, target)
    initial_mapping = hungarian_mapping(inames, cnames, index, target, base)
    unmapped = cnames - frozenset(dict(initial_mapping).values())
    initial_cost = mapping_cost(initial_mapping, target, base, index)
    op_problem = StructureMappingOptimizationProblem((initial_mapping, unmapped),
                                                     initial_cost=initial_cost,
                                                     extra=(target, base,
                                                            index))
    solution = next(hill_climbing(op_problem))
    return dict(solution.state[0])

def eval_obj_mapping(target_o, mapping, target, base, index, partial=False):
    """
    Used to compute the value of the specified object (target_o) in the current
    mapping. Partial specifies whether partial relation matches should be
    computed. Including these is much more expensive.

    :param target_o: The current object that has been modified in the mapping.
    :type target_o: str
    :param mapping: the current mapping
    :type mapping: dict
    :param target: the target of the structure mapping
    :type target: :ref:`Instance<instance-rep>` or concept av_counts table
    :param base: the base of the structure mapping
    :type base: concept av_counts table
    :param index: an index mapping objects to the relations they appear in
    :type index: dict
    :param partial: whether to compute partial rewards for partially met
        relations.
    :type partial: Boolean
    """
    r = 0.0
    for attr in index[target_o]:
        new_attrs = []
        num_comps = len(get_attribute_components(attr))
        if partial and num_comps > 1:
            for cattr in base.av_counts:
                if is_partial_match(attr, cattr, mapping):
                    new_attrs.append(cattr)
        else:
            new_attrs.append(bind_flat_attr(attr, mapping))

        for new_attr in new_attrs:
            if isinstance(target[attr], dict):
                for val in target[attr]:
                    r += (base.attr_val_guess_gain(new_attr, val, 
                                                  target[attr][val]) /
                          num_comps)
            elif isinstance(target[attr], ContinuousValue):
                r += (base.attr_val_guess_gain(new_attr, target[attr],
                                             target[attr].num) / num_comps)
            else:
                r += (base.attr_val_guess_gain(new_attr, target[attr]) /
                      num_comps)

    return r

def hungarian_mapping(inames, cnames, index, target, base, partial=False):
    """
    Utilize the hungarian/munkres matching algorithm over the object to object
    features (ignoring the relations) to compute the initial assignment. Then
    utilize local search techniques to improve this matching by taking into
    account the relations
    """
    cnames = list(cnames)
    inames = list(inames)

    cost_matrix = []
    for o in inames:
        row = []
        for c in cnames:
            nm = {}
            nm[o] = c
            r = eval_obj_mapping(o, nm, target, base, index, partial=partial)
            row.append(-r)
        for o in inames:
            row.append(0.0)
        cost_matrix.append(row)

    m = Munkres()
    indices = m.compute(cost_matrix)

    mapping = {}
    for row,col in indices:
        if col >= len(cnames):
            mapping[inames[row]] = inames[row]
        else:
            mapping[inames[row]] = cnames[col]

    return frozenset(mapping.items())

def greedy_best_mapping(inames, cnames, index, target, base, partial=False):
    """
    A very greedy approach to finding an initial solution for the search.

    Currently, it computes a pairwaise match between each object and instance.
    It then iterates through the best matches and assigning legal matches and
    skipping illegal matches. When all objects have been assigned the resulting
    mapping is returned. 
    """
    mapping = {}
    cnames = set(cnames)
    inames = set(inames)
    possible = []

    for o in inames:
        heappush(possible, (0.0, o, o))
        for c in cnames:
            nm = {}
            nm[o] = c
            r = eval_obj_mapping(o, nm, target, base, index, partial=partial)
            heappush(possible, (-r, o, c))

    while len(inames) > 0:
        r, o, c = heappop(possible)
        if o not in inames:
            continue
        if c not in cnames and c != o:
            continue
        mapping[o] = c
        inames.remove(o)
        if c in cnames:
            cnames.remove(c)

    return frozenset(mapping.items())

def swap_two_mapping_cost(initial_cost, o1, o2, old_mapping, new_mapping, target, base, index):
    """
    A function for computing the cost of a successor in the optimization
    search. This function has O(n) time where n is the number of features in
    the object index.
    """
    cost = initial_cost

    if isinstance(old_mapping, frozenset):
        old_mapping = dict(old_mapping)
    if isinstance(new_mapping, frozenset):
        new_mapping = dict(new_mapping)

    # remove current assignments
    cost -= object_mapping_cost(o1, old_mapping, target, base, index, False)
    temp = old_mapping[o1]
    del old_mapping[o1]
    cost -= object_mapping_cost(o2, old_mapping, target, base, index, False)
    old_mapping[o1] = temp

    # add new assignments
    temp = new_mapping[o2]
    del new_mapping[o2]
    cost += object_mapping_cost(o1, new_mapping, target, base, index, False)
    new_mapping[o2] = temp
    cost += object_mapping_cost(o2, new_mapping, target, base, index, False)

    return cost

def swap_unnamed_mapping_cost(initial_cost, o1, old_mapping, new_mapping,
                              target, base, index):
    """
    A function for computing the cost of a successor in the optimization
    search. This function has O(n) time where n is the number of features in
    the object index.
    """
    cost = initial_cost

    if isinstance(old_mapping, frozenset):
        old_mapping = dict(old_mapping)
    if isinstance(new_mapping, frozenset):
        new_mapping = dict(new_mapping)

    # remove current assignments
    cost -= object_mapping_cost(o1, old_mapping, target, base, index, False)

    # add new assignments
    cost += object_mapping_cost(o1, new_mapping, target, base, index, False)

    return cost

def object_mapping_cost(o, mapping, target, base, index, div_relations=True):
    """
    A function that is used to compute the cost of a particular object in a
    mapping
    """
    cost = 0.0

    for attr in index[o]:
        num_objs = 1
        if div_relations:
            num_objs = len(get_attribute_components(attr))
        new_attr = bind_flat_attr(attr, mapping)

        if isinstance(target[attr], dict):
            for val in target[attr]:
                cost -= (base.attr_val_guess_gain(new_attr, val, 
                                                  target[attr][val]) / 
                         num_objs)
        elif isinstance(target[attr], ContinuousValue):
            cost -= (base.attr_val_guess_gain(new_attr, target[attr],
                                         target[attr].num) / num_objs)
        else:
            cost -= (base.attr_val_guess_gain(new_attr, target[attr]) /
                     num_objs)
    return cost


def mapping_cost(mapping, target, base, index):
    """
    An overall function for evaluating a mapping.
    """
    if isinstance(mapping, frozenset):
        mapping = dict(mapping)
    cost = 0
    for o in mapping:
        cost += object_mapping_cost(o, mapping, target, base, index)
    return cost

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
    def node_value(self, node):
        """
        The value is the precomputed cost.
        """
        return node.cost()

    def swap_two(self, o1, o2, mapping, unmapped_cnames, target, base, index,
                       node):
        """
        returns the child node generated from swapping two mappings.
        """
        new_mapping = {a:mapping[a] for a in mapping}

        if mapping[o2] == o2:
            new_mapping[o1] = o1
        else:
            new_mapping[o1] = mapping[o2]

        if mapping[o1] == o1:
            new_mapping[o2] = o2
        else:
            new_mapping[o2] = mapping[o1]

        new_mapping = frozenset(new_mapping.items())
        #path_cost = mapping_cost(new_mapping, target, base, index)
        path_cost = swap_two_mapping_cost(node.cost(), o1, o2, mapping,
                                          new_mapping, target, base, index)
        #if abs(path_cost - swap_cost) > 0.001:
        #    print(path_cost)
        #    print(swap_cost)
        #    assert False

        return Node((new_mapping, unmapped_cnames), node, 
                   ('swap', (o1, mapping[o1]), (o2, mapping[o2])), path_cost,
                    node.extra)

    def swap_unnamed(self, o1, o2, mapping, unmapped_cnames, target, base, index,
                       node):
        """
        Returns the child node generated from assigning an unmapped component
        object to one of the instance objects.
        """
        new_mapping = {a:mapping[a] for a in mapping}
        new_unmapped_cnames = set(unmapped_cnames)
        new_unmapped_cnames.remove(o2)
        if mapping[o1] != o1:
            new_unmapped_cnames.add(new_mapping[o1])
        new_mapping[o1] = o2
        new_mapping = frozenset(new_mapping.items())
        #path_cost = mapping_cost(new_mapping, target, base, index)
        path_cost = swap_unnamed_mapping_cost(node.cost(), o1, mapping,
                                          new_mapping, target, base, index)
        #if abs(path_cost - swap_cost) > 0.001:
        #    print(path_cost)
        #    print(swap_cost)
        #    assert False

        return Node((new_mapping,
                    frozenset(new_unmapped_cnames)), node,
                    ('swap', (o1, mapping[o1]), ('unmapped', o2)), path_cost, node.extra)

    def random_successor(self, node):
        """
        Similar to the successor function, but generates only a single random
        successor.
        """
        mapping, unmapped_cnames = node.state
        target, base, index = node.extra
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
                                 base, index, node)
        else:
            o2 = choice(list(unmapped_cnames)) 
            return self.swap_unnamed(o1, o2, mapping, unmapped_cnames, target,
                                     base, index, node)

    def successors(self, node):
        """
        An iterator that returns all successors.
        """
        mapping, unmapped_cnames = node.state
        target, base, index = node.extra
        mapping = dict(mapping)

        for o1 in mapping:
            # flip two non-self mappings
            for o2 in mapping:
                if o1 == o2 or (mapping[o1] == o1 and mapping[o2] == o2):
                    continue

                yield self.swap_two(o1, o2, mapping, unmapped_cnames, target,
                                      base, index, node)


            # flip mapped with some unused cname
            for o2 in unmapped_cnames:
                yield self.swap_unnamed(o1, o2, mapping, unmapped_cnames,
                                        target, base, index, node)
                
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
    :return: ``True`` if the instance attribute matches the concept attribute in the mapping otherwise ``False``
    :rtype: bool

    >>> is_partial_match(('<', ('a', '?o2'), ('a', '?o1')), ('<', ('a', '?c2'), ('b', '?c1')), {'?o1': '?c1'})
    False

    >>> is_partial_match(('<', ('a', '?o2'), ('a', '?o1')), ('<', ('a', '?c2'), ('a', '?c1')), {'?o1': '?c1'})
    True

    >>> is_partial_match(('<', ('a', '?o2'), ('a', '?o1')), ('<', ('a', '?c2'), ('a', '?c1')), {'?o1': '?c1', '?o2': '?c2'})
    True
    """
    if type(iAttr) != type(cAttr):
        return False

    if isinstance(iAttr, tuple) and len(iAttr) != len(cAttr):
        return False

    if isinstance(iAttr, tuple):
        for i,v in enumerate(iAttr):
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
    Flatten the instance, perform structure mapping to the concept, rename
    the instance based on this structure mapping, and return the renamed
    instance.

    :param base: A concept to structure map the instance to
    :type base: TrestleNode
    :param gensym: a function that returns unique object names (str) on each
        call
    :type gensym: function
    :return: A flattened and mapped copy of the instance
    :rtype: instance
    """
    def __init__(self, base, gensym):
        self.base = base
        self.reverse_mapping = None
        self.name_standardizer = NameStandardizer(gensym)

    def get_mapping(self):
        """
        Returns the currently established mapping.

        :return: The current mapping.
        :rtype: dict
        """
        return {self.reverse_mapping[o]: o for o in self.reverse_mapping}
    
    def transform(self, target):
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
        target = self.name_standardizer.transform(target)
        mapping = flat_match(target, self.base)
        self.reverse_mapping = {mapping[o]: o for o in mapping}
        return rename_flat(target, mapping)

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
        target = rename_flat(target, self.reverse_mapping)
        return self.name_standardizer.undo_transform(target)

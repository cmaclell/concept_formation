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

from itertools import permutations
from random import choice
from random import random
from random import shuffle
from heapq import heappush
from heapq import heappop

from py_search.search import Problem
from py_search.search import Node
from py_search.search import beam_search
from py_search.search import best_first_search
from py_search.search import simulated_annealing_optimization
from py_search.search import beam_optimization
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

def compute_rewards(names, target, base):
    """
    The function computes the rewards for renaming the target to match it to
    the base. All of the rewards for possible mappings are computed at once so
    that they don't have to be recomputed at each step in the search. 

    target here can be either an instance or an av_counts object from a
    concept.

    .. todo:: 
        Consider trying to speed this up
    """
    rewards = {}

    for attr in target:
        a_comps = get_attribute_components(attr)

        if len(a_comps) == 0:
            continue

        for c_comps in permutations(names, len(a_comps)):
            mapping = dict(zip(a_comps, c_comps))
            new_attr = bind_flat_attr(attr, mapping)

            if isinstance(target[attr], dict):
                r = 0
                for val in target[attr]:
                    r += base.attr_val_guess_gain(new_attr, val, 
                                                  target[attr][val])
            elif isinstance(target[attr], ContinuousValue):
                r = base.attr_val_guess_gain(new_attr, target[attr],
                                             target[attr].num)
            else:
                r = base.attr_val_guess_gain(new_attr, target[attr])

            if r != 0:
                items = sorted(mapping.items())
                keys = tuple(i[0] for i in items)
                values = tuple(i[1] for i in items)
                if keys not in rewards:
                    rewards[keys] = {}
                if values not in rewards[keys]:
                    rewards[keys][values] = 0
                rewards[keys][values] += r

    return rewards

def build_index(onames, instance):
    """
    Given a set of object names and an instance (or av_count table), builds a
    an index (a dict) that has object names as key and a list of all relations
    that refer to that object as a value.
    """
    index = {}
    for o in onames:
        index[o] = []
    for attr in instance:
        for o in get_attribute_components(attr):
            index[o].append(attr)
    return index

def flat_match(target, base, beam_width=1, vars_only=True):
    """
    Given a concept and instance this function returns a mapping that can be
    used to rename components in the instance. Search is used to find a mapping
    that maximizes the expected number of correct guesses in the concept after
    incorporating the instance. 

    Beam search is used to find a mapping between instance and concept.  The
    lower the beam width the more greedy (and faster) the search.  If the beam
    width is set to `float('inf')` then uses A* instead.

    :param target: An instance or concept.av_counts object to be mapped to the
        base concept.
    :type target: instance or av_counts obj from concept
    :param base: A concept to map the target to
    :type base: TrestleNode
    :param beam_width: The width of the beam used for Beam Search. Uses A* if
        the beam width is `float('inf')` 
    :type beam_width: int, or float('inf') for A* 
    :return: a mapping for renaming components in the instance.
    :rtype: dict
    """
    inames = frozenset(get_component_names(target))
    cnames = frozenset(get_component_names(base.av_counts))
    print("%i x %i Mapping" % (len(inames), len(cnames)))

    if(len(inames) == 0 or len(cnames) == 0):
        return {}

    index = build_index(inames, target)
    initial_mapping = greedy_best_mapping(inames, cnames, index, target, base)
    print(initial_mapping)
    unmapped = cnames - frozenset(dict(initial_mapping).values())
    initial_cost = mapping_cost(initial_mapping, target, base, index)
    print(initial_cost)
    op_problem = StructureMappingOptimizationProblem((initial_mapping, unmapped),
                                                     initial_cost=initial_cost,
                                                     extra=(target, base,
                                                            index))
    solution = next(beam_optimization(op_problem, beam_width=1))
    print("Beam Solution")
    print(solution.cost())
    #solution = next(simulated_annealing_optimization(op_problem,
    #                                                 limit=1000+10*len(inames)*len(cnames)))
    #print("Annealing Solution")
    #print(solution.cost())
    #print(1000+10*len(inames)*len(cnames))

    if len(set(dict(initial_mapping).keys())) != len(set(dict(initial_mapping).values())):
        from pprint import pprint
        pprint("INITIAL")
        pprint(initial_mapping)
        assert False

    if len(set(dict(solution.state[0]).keys())) != len(set(dict(solution.state[0]).values())):
        from pprint import pprint
        pprint(dict(solution.state[0]))
        print(solution.path())
        assert False

    return dict(solution.state[0])

def eval_obj_mapping(target_o, mapping, target, base, index):
    r = 0.0
    for attr in index[target_o]:
        new_attr = bind_flat_attr(attr, mapping)

        if isinstance(target[attr], dict):
            for val in target[attr]:
                r += base.attr_val_guess_gain(new_attr, val, 
                                              target[attr][val])
        elif isinstance(target[attr], ContinuousValue):
            r += base.attr_val_guess_gain(new_attr, target[attr],
                                         target[attr].num)
        else:
            r += base.attr_val_guess_gain(new_attr, target[attr])

    return r

def find_best_c(o, cnames, mapping, target, base, index):
    best = o
    best_reward = 0.0
    for c in cnames:
        nm = {a:mapping[a] for a in mapping}
        nm[o] = c
        r = eval_obj_mapping(o, nm, target, base, index)
        if r > best_reward:
            best = c
            best_reward = r
    return (best_reward, o, best)


def greedy_best_mapping(inames, cnames, index, target, base):
    """
    A very greedy approach to finding an initial solution for the search.

    Currently it computes all of the matches and chooses the best for each
    starting with the strongest match. If a conflict is encountered, then it 
    recomputes the best match for the conflict and continues. 

    While it does take into account relations, it might be better to recompute
    the best possible after each assignment, so that relations can be
    collected. Alternatively the find_best_c could be updated to collect
    partial match rewards (i.e., assume you can get all relations). 
    """
    mapping = {}
    cnames = set(cnames)
    inames = set(inames)
    possible = []

    for o in inames:
        heappush(possible, find_best_c(o, cnames, mapping, target, base,
                                       index))

    while len(inames) > 0:
        while len(possible) > 0:
            r, o, c = heappop(possible)
            if o not in inames:
                continue
            if c not in cnames and c != o:
                heappush(possible, find_best_c(o, cnames, mapping, target, base,
                                               index))
                continue
            mapping[o] = c
            inames.remove(o)
            if c in cnames:
                cnames.remove(c)

    return frozenset(mapping.items())

def swap_two_mapping_cost(initial_cost, o1, o2, old_mapping, new_mapping, target, base, index):
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
    if isinstance(mapping, frozenset):
        mapping = dict(mapping)
    cost = 0
    for o in mapping:
        cost += object_mapping_cost(o, mapping, target, base, index)
    return cost

class StructureMappingOptimizationProblem(Problem):

    def node_value(self, node):
        return node.cost()

    def swap_two(self, o1, o2, mapping, unmapped_cnames, target, base, index,
                       node):
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
                

class StructureMappingProblem(Problem):
    """
    A class for describing a structure mapping problem to be solved using the
    `py_search<http://py-search.readthedocs.org/>_` library. This class defines
    the node_value, the successor, and goal_test methods used by the search
    library.
    """
    def partial_match_heuristic(self, node):
        """
        Given a node, considers all partial matches for each unbound attribute
        and assumes that you get the highest guess_gain match. This provides an
        over estimation of the possible reward (i.e., is admissible).

        This heuristic is used by the :func:`node_value` method to
        estimate how promising the state is. 
        """
        mapping, unnamed, availableNames = node.state
        rewards = node.extra

        h = 0
        for iattr in rewards:
            values = rewards[iattr].values()
            if len(values) > 0:
                h -= max(rewards[iattr].values())

        return h

    def node_value(self, node):
        """
        The value of a node. Uses cost + heuristic to achieve A* and other
        greedy variants.

        See the `py_search<http://py-search.readthedocs.org/>_` library for
        more details of how this function is used in search.
        """
        return node.cost() #+ self.partial_match_heuristic(node)

    def reward(self, new, mapping, rewards):
        reward = 0
        new_rewards = {}

        mapped = frozenset(mapping.keys())
        reverse_mapping = {mapping[a]:a for a in mapping}

        for iattr in rewards:
            if frozenset(iattr).issubset(mapped):
                bindings = tuple(mapping[o] for o in iattr)
                if bindings in rewards[iattr]:
                    reward += rewards[iattr][bindings]
            else:
                nrw = {}
                for vals in rewards[iattr]:
                    partial_match = True
                    for i,v in enumerate(vals):
                        if iattr[i] in mapping and mapping[iattr[i]] != v:
                            partial_match = False
                            break
                        if (v in reverse_mapping and reverse_mapping[v] !=
                            iattr[i]):
                            partial_match = False
                            break
                    if partial_match:
                        nrw[vals] = rewards[iattr][vals]
                if len(nrw) > 0:
                    new_rewards[iattr] = nrw

            #elif new[0] in iattr:
            #    # TODO double check we cannot knock out more possible matches
            #    idx = iattr.index(new[0])
            #    new_rewards[iattr] = {vals: rewards[iattr][vals] for vals in
            #                          rewards[iattr] if vals[idx] == new[1]}
            #else:
            #    new_rewards[iattr] = {val: rewards[iattr][val] for val in 
            #                          rewards[iattr] if len(val) > 1 or 
            #                          val[0] != new[1]}

        return reward, new_rewards

    def successors(self, node):
        """
        Given a search node (contains mapping, instance, concept), this
        function computes the successor nodes where an additional mapping has
        been added for each possible additional mapping. 

        See the `py_search<http://py-search.readthedocs.org/>_` library for
        more details of how this function is used in search.
        """
        mapping, inames, availableNames = node.state
        rewards = node.extra

        for a in inames:
            for b in frozenset([a]).union(availableNames):
                m = {a:v for a,v in mapping}
                m[a] = b
                state = (mapping.union(frozenset([(a, b)])), inames -
                            frozenset([a]), availableNames - frozenset([b]))
                reward, new_rewards = self.reward((a,b), m, rewards)
                path_cost = node.cost() - reward

                yield Node(state, node, (a, b), path_cost, extra=new_rewards)

    def goal_test(self, node):
        """
        Given a search node, this returns True if every component in the
        original instance has been renamed in the given node.

        See the `py_search<http://py-search.readthedocs.org/>_` library for
        more details of how this function is used in search.
        """
        mapping, unnamed, availableNames = node.state
        #print(unnamed)
        #print(node.extra)
        return len(unnamed) == 0

def is_partial_match(iAttr, cAttr, mapping, unnamed):
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

    >>> is_partial_match(('<', ('a', '?o2'), ('a', '?o1')), ('<', ('a', '?c2'), ('b', '?c1')), {'?o1': '?c1'}, {'?o2'})
    False

    >>> is_partial_match(('<', ('a', '?o2'), ('a', '?o1')), ('<', ('a', '?c2'), ('a', '?c1')), {'?o1': '?c1'}, {'?o2'})
    True

    >>> is_partial_match(('<', ('a', '?o2'), ('a', '?o1')), ('<', ('a', '?c2'), ('a', '?c1')), {'?o1': '?c1', '?o2': '?c2'}, {})
    True
    """
    if type(iAttr) != type(cAttr):
        return False

    if isinstance(iAttr, tuple) and len(iAttr) != len(cAttr):
        return False

    if isinstance(iAttr, tuple):
        for i,v in enumerate(iAttr):
            if not is_partial_match(iAttr[i], cAttr[i], mapping, unnamed):
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

    :param concept: A concept to structure map the instance to
    :type concept: TrestleNode
    :param beam_width: The width of the beam used for Beam Search. If set to
        float('inf'), then A* will be used.
    :type beam_width: int (or float('inf') for optimal) 
    :param vars_only: Determines whether or not variables in the instance can
        be matched only to variables in the concept or if they can also be bound to
        constants. 
    :type vars_only: boolean
    :param pipeline: A preprocessing pipeline to apply before structure mapping
        and to undo when undoing the structure mapping. If ``None`` then the
        default pipeline of
        :class:`Tuplizer<concept_formation.preprocessor.Tuplizer>` ->
        :class:`NameStandardizer<concept_formation.preprocessor.NameStandardizer>`
        ->
        :class:`SubComponentProcessor<concept_formation.preprocessor.SubComponentProcessor>`
        -> :class:`Flattener<concept_formation.preprocessor.Flattener>` is
        applied
    :return: A flattened and mapped copy of the instance
    :rtype: instance
    """
    def __init__(self, base, gensym, beam_width=1, vars_only=True):
        self.base = base
        self.reverse_mapping = None
        self.beam_width = beam_width
        self.vars_only = vars_only
        self.name_standardizer = NameStandardizer(gensym)

    def get_mapping(self):
        return {self.reverse_mapping[o]: o for o in self.reverse_mapping}
    
    def transform(self, target):
        target = self.name_standardizer.transform(target)
        mapping = flat_match(target, self.base,
                             beam_width=self.beam_width,
                             vars_only=self.vars_only)
        self.reverse_mapping = {mapping[o]: o for o in mapping}
        return rename_flat(target, mapping)

    def undo_transform(self, target):
        if self.reverse_mapping is None:
            raise Exception("Must transform before undoing transform")
        target = rename_flat(target, self.reverse_mapping)
        return self.name_standardizer.undo_transform(target)

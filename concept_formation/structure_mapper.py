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

from py_search.search import Problem
from py_search.search import Node
from py_search.search import beam_search
from concept_formation.preprocessor import Tuplizer
from concept_formation.preprocessor import NameStandardizer
from concept_formation.preprocessor import SubComponentProcessor
from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Pipeline
from concept_formation.preprocessor import Preprocessor
from concept_formation.preprocessor import rename_relation

def get_relation_components(relation, vars_only=True):
    """
    Gets component names out of a relation.
    """
    names = set()

    if vars_only is not True and relation[0] != '_':
        names.add(relation)

    for ele in relation:
        if isinstance(ele, tuple):
            for name in get_relation_components(ele, vars_only):
                names.add(name)
        else:
            if ((vars_only is not True or ele[0] == '?') and ele != '_' and
                ele[0] != '_'):
                names.add(ele)

    return names

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
        if isinstance(attr, tuple):
            for name in get_relation_components(attr, vars_only):
                names.add(name)
        elif (vars_only is not True and attr[0] != '_') or attr[0] == '?':
            names.add(attr)

    return names

def rename_flat(instance, mapping):
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

    for attr in instance:
        if attr in mapping:
            temp_instance[mapping[attr]] = instance[attr]
        elif isinstance(attr, tuple):
            temp_instance[rename_relation(attr, mapping)] = instance[attr]
        else:
            temp_instance[attr] = instance[attr]

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

    If the mapping is incomplete then returns ``None`` (nothing) 

    >>> attr = ('before', '?c1', '?c2')
    >>> mapping = {'?c1': 'o1'}
    >>> bind_flat_attr(attr, mapping) is None
    True

    >>> bind_flat_attr(('<', ('a', '?o2'), ('a', '?o1')), {'?o1': '?c1'}) is None
    True

    >>> bind_flat_attr(('<', ('a', '?o2'), ('a', '?o1')), {'?o1': '?c1', '?o2': '?c2'}) is None
    False
    """
    if not isinstance(attr, tuple) and attr in mapping:
        return mapping[attr]

    if not isinstance(attr, tuple):
        if attr[0] == '?':
            return None
        else:
            return attr

    if isinstance(attr, tuple):
        new_attr = []
        for ele in attr:
            new_ele = bind_flat_attr(ele, mapping)
            if new_ele is None:
                return None
            else:
                new_attr.append(new_ele)
        return tuple(new_attr)

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

def flat_match(concept, instance, beam_width=1, vars_only=True):
    """
    Given a concept and instance this function returns a mapping  that can be
    used to rename components in the instance. The mapping returned maximizes
    similarity between the instance and the concept.

    Beam search is used to find a mapping between instance and concept. If
    `beam_width==float('inf')` then searh will be optimal (i.e., A* search).
    The lower the beam width the more greedy (and faster) the search. The
    default beam width is 1, which is basically greedy hill climbing search. In
    situations where an instance contains no relational attributes and no
    sub-component attributes (i.e., components that contain other components)
    than a beam width of 1 will return the optimal result. 

    :param concept: A concept to map the instance to
    :type concept: TrestleNode
    :param instance: An instance to be mapped to the concept
    :type instance: instance
    :param beam_width: The width of the beam used for Beam Search. If set to
        float('inf'), then A* will be used.
    :type beam_width: int (or float('inf') for optimal) 
    :param vars_only: Determines whether or not variables in the instance can
        be matched only to variables in the concept or if they can also be bound to
        constants. 
    :type vars_only: boolean
    :return: a mapping for renaming components in the instance.
    :rtype: dict
    """
    inames = frozenset(get_component_names(instance))
    cnames = frozenset(get_component_names(concept.av_counts, vars_only))

    if(len(inames) == 0 or len(cnames) == 0):
        return {}

    problem = StructureMappingProblem((frozenset(), inames, cnames),
                                      extra=(concept, instance))
    solution = next(beam_search(problem, beam_width=beam_width))

    if solution:
        mapping, unnamed, availableNames = solution.state
        return {a:v for a,v in mapping}
    else:
        return None

class StructureMappingProblem(Problem):
    """
    A class for describing a structure mapping problem to be solved using the
    `py_search<http://py-search.readthedocs.org/>_` library. This class defines
    the heuristic, the successor, and goal_test methods used by the search
    library.
    """

    def partial_match_heuristic(self, node):
        """
        Given a node, considers all partial matches for each unbound attribute
        and assumes that you get the highest guess_gain match. This provides an
        over estimation of the possible reward (i.e., is admissible).

        This heuristic is used by the :func:`node_value` method to compute the
        value of each state in the search.
        """
        mapping, unnamed, availableNames = node.state
        concept, instance = node.extra

        h = 0
        m = {a:v for a,v in mapping}
        for attr in instance:
            new_attr = bind_flat_attr(attr, m)
            if not new_attr:
                best_attr_h = [concept.attr_val_guess_gain(cAttr, instance[attr]) for
                                   cAttr in concept.av_counts if
                                   is_partial_match(attr, cAttr, m, unnamed)]

                if len(best_attr_h) > 0:
                    h -= max(best_attr_h)

        return h

    def node_value(self, node):
        """
        The value of a node. Uses cost + heuristic to achieve A* and other
        greedy variants.
        """
        return node.cost() + self.partial_match_heuristic(node)

    def successor(self, node):
        """
        Given a search node (contains mapping, instance, concept), this
        function computes the successor nodes where an additional mapping has
        been added for each possible additional mapping. 

        See the `py_search<http://py-search.readthedocs.org/>_` library for
        more details of how this function is used in search.
        """
        mapping, inames, availableNames = node.state
        concept, instance = node.extra

        for n in inames:
            reward = 0
            m = {a:v for a,v in mapping}
            m[n] = n
            for attr in instance:
                if not contains_component(n, attr):
                    continue
                new_attr = bind_flat_attr(attr, m)
                if new_attr:
                    reward += concept.attr_val_guess_gain(new_attr, instance[attr])

            state = (mapping.union(frozenset([(n, n)])), inames -
                        frozenset([n]), availableNames)
            path_cost = node.cost() - reward
            yield Node(state, node, n + ":" + n, path_cost, extra=node.extra)

            for new in availableNames:
                reward = 0
                m = {a:v for a,v in mapping}
                m[n] = new
                for attr in instance:
                    if not contains_component(n, attr):
                        continue
                    new_attr = bind_flat_attr(attr, m)
                    if new_attr:
                        reward += concept.attr_val_guess_gain(new_attr,
                                                              instance[attr])
                state = (mapping.union(frozenset([(n, new)])), inames -
                                          frozenset([n]), availableNames -
                                          frozenset([new]))
                path_cost = node.cost() - reward
                yield Node(state, node, n + ":" + new, path_cost,
                           extra=node.extra)

    def goal_test(self, node):
        """
        Given a search node, this returns True if every component in the
        original instance has been renamed in the given node.

        See the `py_search<http://py-search.readthedocs.org/>_` library for
        more details of how this function is used in search.
        """
        mapping, unnamed, availableNames = node.state
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
    def __init__(self, concept, beam_width=1, vars_only=True, pipeline=None):
        self.concept = concept        
        self.reverse_mapping = None
        self.beam_width = beam_width
        self.vars_only = vars_only

        if pipeline is None:
        	self.pipeline = Pipeline(Tuplizer(), NameStandardizer(),
                                 SubComponentProcessor(), Flattener())
        else :
        	self.pipeline = pipeline

    def get_mapping(self):
        return {self.reverse_mapping[o]: o for o in self.reverse_mapping}
    
    def transform(self, instance):
        instance = self.pipeline.transform(instance)
        mapping = flat_match(self.concept, instance,
                             beam_width=self.beam_width,
                             vars_only=self.vars_only)
        self.reverse_mapping = {mapping[o]: o for o in mapping}
        return rename_flat(instance, mapping)

    def undo_transform(self, instance):
        if self.reverse_mapping is None:
            raise Exception("Must transform before undoing transform")
        instance = rename_flat(instance, self.reverse_mapping)
        return self.pipeline.undo_transform(instance)



                


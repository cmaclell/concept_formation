"""
This module contains classes for preprocessing and structure mapping instances
for use with Trestle. A preprocessor transforms an instance so that it can be
integrated into Trestle's knoweldge base. 

.. todo:: Most of this module's header comment is no longer true. We need to update it.

This module contains all of the core functions for Trestle's structure mapping
and flattening procedures. The core function in this process is
:meth:`structure_map` with most of the remaning functions being sub-procedures of
the core process.

Throughout this module we refer to instances in several different stages of the
structure mapping process. Here is a description of what each stage means:

.. _raw-instance:

* **raw instance** - The original state of the instance. Conventionally this is assumed to be
  the result of ``json.load()``). All component atributes have their original names
  and refer to dictionaries with their own attribute values and all relations are full lists.

.. _standard-instance:

* **standardized instance** - An instance where all components have been
  renamed, both in the instance and in relations, to have unique names to prevent
  collisions in the mapping process. (e.g. ``{a:{b:2}, c:{d:{x:1,y:2}}}`` -> ``{o1:{b:2}, o2:{o3:{x:1,y:2}}}``)

.. _flattened-instance:

* **flattened instance** - An instance where component attributes are flattened
  using a dot notation (e.g. ``o1:{b:1}`` -> ``o1.b:1``) and relations have been
  turned into tuples (e.g. ``rel:['before', 'o1', 'o2']`` -> ``('before', 'o1',
  'o2'):True``)

.. _fully-mapped:

* **mapped instance** - A fully structure mapped instance with component
  attributes renamed, both in the instance and its relations. And components
  flattened using dot notation. This is the final result of the overall process.

"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from concept_formation import search
from concept_formation.preprocessor import Tuplizer
from concept_formation.preprocessor import ListProcessor
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
    Given  a :ref:`flattened instance <flattened-instance>` or a concept's
    probability table return a list of all of the component names.

    :param instance: An instance or a concept's probability table.
    :type instance: :ref:`raw instance <raw-instance>` or dict
    :return: A list of all of the component names present in the instance
    :rtype: [str, str, ...]

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
    Given a :ref:`flattened instance <flattened-instance>` and a mapping (type =
    dict) rename the components and relations and return the renamed instance.

    :param instance: An instance to be renamed according to a mapping
    :type instance: :ref:`flattened instance <flattened-instance>`
    :param mapping: :param mapping: A dictionary of mappings between component names
    :type mapping: dict
    :return: A copy of the instance with components and relations renamed
    :rtype: :ref:`mapped instance <fully-mapped>`

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



def _traverseStructure(path, instance):
    """Given an instance dict to the given subobject for the given path.
    Creates subobjects if they do not exist.

    Essentially this ensures that subdictionaries exist to accept values from a
    flat instance.

    >>> x = {}
    >>> _traverseStructure(['c1', 'c2'], x)
    {}
    >>> x
    {'c1': {'c2': {}}}
    """
    curr = instance
    for obj in path:
        if obj not in curr:
            curr[obj] = {}
        curr = curr[obj]
    return curr

def structurizeJSON(instance):
    """Takes a :ref:`flattened instance <flattened-instance>` and adds the
    structure back in. 

    This essentially "undoes" the flattening process, however any renaming that
    may have been performed is not undone.

    :param instance: A instance to be re-structured
    :type instance: :ref:`flattened instance <flattened-instance>`
    :return: the instance with structure reproduced from the flattened information
    :rtype: :ref:`standardized instance <standard-instance>`

    >>> instance = {'c1.c2.a': 1}
    >>> structurizeJSON(instance)
    {'c1': {'c2': {'a': 1}}}
    """
    temp = {}
    for attr in instance:
        if isinstance(attr, tuple):
            relation = []
            path = []
            for i,v in enumerate(attr):
                if i == 0:
                    relation.append(v)
                elif v in instance:
                    path = v.split('.')
                    relation.append('.'.join(path[-2:]))
                    path = path[:-2]
                else:
                    path = v.split('.')
                    relation.append(path[-1])
                    path = path[:-1]
            obj = _traverseStructure(path, temp)
            obj[tuple(relation)] = True

        elif "." in attr:
            path = [p[1:] if p[0] == "_" else p for p in attr.split('.')]
            subatt = path[-1]
            path = path[:-1]
            curr = _traverseStructure(path, temp)
            if attr[0] == "_":
                curr["_" + subatt] = instance[attr]
            else:
                curr[subatt] = instance[attr]

        else:
            temp[attr] = instance[attr]

    return temp

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

    The mapping search can be performed in an optimal way (using A* search) to
    guarantee the best possible mapping at the expense of performance or in an
    greedy way (using Beam search) to find a sufficient solution in less time.
    
    .. note:: If the instance contains no relational attributes then the optimal
       and greedy searches will be identical.

    :param concept: A concept to map the instance to
    :type concept: TrestleNode
    :param instance: An instance to be mapped to the concept
    :type instance: :ref:`flattened instance <flattened-instance>`
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

    initial = search.Node((frozenset(), inames, cnames), extra=(concept,
                                                                instance))
    if beam_width == float('inf'):
        solution = next(search.BestFGS(initial, _flat_match_successor_fn,
                                       _flat_match_goal_test_fn,
                                       _flat_match_heuristic_fn))
    else:
        solution = next(search.BeamGS(initial, _flat_match_successor_fn,
                                      _flat_match_goal_test_fn,
                                      _flat_match_heuristic_fn,
                                      initialBeamWidth=beam_width))
    #print(solution.cost)

    if solution:
        mapping, unnamed, availableNames = solution.state
        return {a:v for a,v in mapping}
    else:
        return None

def _flat_match_successor_fn(node):
    """
    Given a node (mapping, instance, concept), this function computes the
    successor nodes where an additional mapping has been added for each
    possible additional mapping. 

    See the :mod:`concept_formation.search` library for more details.
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

        yield search.Node((mapping.union(frozenset([(n, n)])), inames -
                    frozenset([n]), availableNames), node, n + ":" + n,
                   node.cost - reward, node.depth + 1, node.extra)

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

            yield search.Node((mapping.union(frozenset([(n, new)])), inames -
                                      frozenset([n]), availableNames -
                                      frozenset([new])), node, n + ":" + new,
                        node.cost - reward, node.depth + 1, node.extra)


def _flat_match_heuristic_fn(node):
    """
    Considers all partial matches for each unbound attribute and assumes that
    you get the highest guess_gain match. This provides an over estimation of
    the possible reward (i.e., is admissible).

    See the :mod:`concept_formation.search` library for more details.
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

def _flat_match_goal_test_fn(node):
    """
    Returns True if every component in the original instance has been renamed
    in the given node.

    See the :mod:`concept_formation.search` library for more details.
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

def pre_process(instance):
    """
    Runs all of the pre-processing functions

    >>> from concept_formation.preprocessor import _reset_gensym
    >>> _reset_gensym()
    >>> import pprint
    >>> instance = {"noma":"a","num3":3,"compa":{"nomb":"b","num4":4,"sub":{"nomc":"c","num5":5}},"compb":{"nomd":"d","nome":"e"},"(related compa.num4 compb.nome)":True,"list1":["a","b",{"i":1,"j":12.3,"k":"test"}]}
    >>> pprint.pprint(instance)
    {'(related compa.num4 compb.nome)': True,
     'compa': {'nomb': 'b', 'num4': 4, 'sub': {'nomc': 'c', 'num5': 5}},
     'compb': {'nomd': 'd', 'nome': 'e'},
     'list1': ['a', 'b', {'i': 1, 'j': 12.3, 'k': 'test'}],
     'noma': 'a',
     'num3': 3}

    >>> instance = pre_process(instance)
    >>> pprint.pprint(instance)
    {'noma': 'a',
     'num3': 3,
     ('has-component', 'compa', 'sub'): True,
     ('has-element', 'list1', '?o4'): True,
     ('has-element', 'list1', '?o5'): True,
     ('has-element', 'list1', '?o6'): True,
     ('i', '?o6'): 1,
     ('j', '?o6'): 12.3,
     ('k', '?o6'): 'test',
     ('nomb', 'compa'): 'b',
     ('nomc', 'sub'): 'c',
     ('nomd', 'compb'): 'd',
     ('nome', 'compb'): 'e',
     ('num4', 'compa'): 4,
     ('num5', 'sub'): 5,
     ('ordered-list', 'list1', '?o4', '?o5'): True,
     ('ordered-list', 'list1', '?o5', '?o6'): True,
     ('related', 'compa.num4', 'compb.nome'): True,
     ('val', '?o4'): 'a',
     ('val', '?o5'): 'b'}

    >>> instance = pre_process(instance)
    >>> pprint.pprint(instance)
    {'noma': 'a',
     'num3': 3,
     ('has-component', 'compa', 'sub'): True,
     ('has-element', 'list1', '?o4'): True,
     ('has-element', 'list1', '?o5'): True,
     ('has-element', 'list1', '?o6'): True,
     ('i', '?o6'): 1,
     ('j', '?o6'): 12.3,
     ('k', '?o6'): 'test',
     ('nomb', 'compa'): 'b',
     ('nomc', 'sub'): 'c',
     ('nomd', 'compb'): 'd',
     ('nome', 'compb'): 'e',
     ('num4', 'compa'): 4,
     ('num5', 'sub'): 5,
     ('ordered-list', 'list1', '?o4', '?o5'): True,
     ('ordered-list', 'list1', '?o5', '?o6'): True,
     ('related', 'compa.num4', 'compb.nome'): True,
     ('val', '?o4'): 'a',
     ('val', '?o5'): 'b'}
    
    """
    tuplizer = Tuplizer()
    instance = tuplizer.transform(instance)

    list_processor = ListProcessor()
    instance = list_processor.transform(instance)

    standardizer = NameStandardizer()
    instance = standardizer.transform(instance)
    
    sub_component_processor = SubComponentProcessor()
    instance = sub_component_processor.transform(instance)

    flattener = Flattener()
    instance = flattener.transform(instance)

    return instance

class StructureMapper(Preprocessor):
    """
    Flatten the instance, perform structure mapping to the concept, rename
    the instance based on this structure mapping, and return the renamed
    instance.

    :param concept: A concept to structure map the instance to
    :type concept: TrestleNode
    :param instance: An instance to map to the concept
    :type instance: :ref:`raw instance <raw-instance>`
    :return: A fully mapped and flattend copy of the instance
    :rtype: :ref:`mapped instance <fully-mapped>`
    """
    def __init__(self, concept, pipeline=None):
        self.concept = concept        
        self.reverse_mapping = None

        if pipeline is None:
        	self.pipeline = Pipeline(Tuplizer(), NameStandardizer(),
                                 SubComponentProcessor(), Flattener())
        else :
        	self.pipeline = pipeline

    def get_mapping(self):
        return {self.reverse_mapping[o]: o for o in self.reverse_mapping}
    
    def transform(self, instance):
        instance = self.pipeline.transform(instance)
        mapping = flat_match(self.concept, instance)
        self.reverse_mapping = {mapping[o]: o for o in mapping}
        return rename_flat(instance, mapping)

    def undo_transform(self, instance):
        if self.reverse_mapping is None:
            raise Exception("Must transform before undoing transform")
        instance = rename_flat(instance, self.reverse_mapping)
        return self.pipeline.undo_transform(instance)



                


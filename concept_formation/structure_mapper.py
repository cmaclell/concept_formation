"""

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

_gensym_counter = 0;

def gensym():
    """
    Generates unique names for naming renaming apart objects.

    :return: a unique object name
    :rtype: 'o'+counter
    """
    global _gensym_counter
    _gensym_counter += 1
    return '?o' + str(_gensym_counter)

def _reset_gensym():
    """
    Resets the gensym counter to 0, which is useful for doctesting. Do not call
    this function during normal operation.
    """
    global _gensym_counter
    _gensym_counter = 0

class Preprocessor(object):
    """
    A template class that defines the functions a preprocessor class should
    implement. In particular, a preprocessor should tranform an instance and
    implement a function for undoing this transformation.
    """
    def transform(self, instance):
        """
        Transforms an instance.
        """
        raise NotImplementedError("Class must implement transform function")

    def undo_transform(self, instance):
        """
        Undoes a transformation to an instance.
        """
        raise NotImplementedError("Class must implement undo_transform function")

class Pipeline(Preprocessor):
    """
    A special preprocessor class used to chain together many preprocessors.
    Supports the same same transform and undo_transform functions as a regular
    preprocessor.
    """
    def __init__(self, *preprocessors):
        self.preprocessors = preprocessors
        print(preprocessors)

    def transform(self, instance):
        for pp in self.preprocessors:
            instance = pp.transform(instance)
        return instance

    def undo_transform(self, instance):
        for pp in reversed(self.preprocessors):
            instance = pp.undo_transform(instance)
        return instance

class Tuplizer(Preprocessor):
    """
    Converts all string versions of relations into tuples

    >>> tuplizer = Tuplizer()
    >>> instance = {'(foo1 o1 (foo2 o2 o3))': True}
    >>> print(tuplizer.transform(instance))
    {('foo1', 'o1', ('foo2', 'o2', 'o3')): True}
    >>> print(tuplizer.undo_transform(tuplizer.transform(instance)))
    {'(foo1 o1 (foo2 o2 o3))': True}
    """
    def transform(self, instance):
        return {self.tuplize_relation(attr): instance[attr] for attr in instance}

    def undo_transform(self, instance):
        return {self.stringify_relation(attr): instance[attr] for attr in instance}

    def tuplize_relation(self, relation):
        """
        Converts a string formatted relation into a tuplized relation. 

        :param attr: The relational attribute formatted as a string
        :type attr: string
        :param mapping: A dictionary of mappings with component names as keys. Just
        the keys are used (i.e., as a set) to determine if elements in the relation
        are objects.
        :type mapping: dict
        :return: A new relational attribute in tuple format
        :rtype: tuple

        >>> relation = '(foo1 o1 (foo2 o2 o3))'
        >>> tuplizer = Tuplizer()
        >>> tuplizer.tuplize_relation(relation)
        ('foo1', 'o1', ('foo2', 'o2', 'o3'))
        """
        if relation[0] != '(':
            return relation

        stack = [[]]

        for val in relation.split(' '):
            end = 0

            if val[0] == '(':
                stack.append([])
                val = val[1:]

            while val[-1] == ')':
                end += 1
                val = val[:-1]
            
            current = stack[-1]
            current.append(val)
            
            while end > 0:
                last = tuple(stack.pop())
                current = stack[-1]
                current.append(last)
                end -= 1

        final = tuple(stack[-1][-1])
        #final = self.tuplize_elements(final)
        return final

    def stringify_relation(self, relation):
        """
        Converts a tupleized relation into a string formated relation.

        >>> relation = ('foo1', 'o1', ('foo2', 'o2', 'o3'))
        >>> tuplizer = Tuplizer()
        >>> tuplizer.stringify_relation(relation)
        '(foo1 o1 (foo2 o2 o3))'
        """
        #relation = convert_unary_to_dot(relation)
        if isinstance(relation, tuple):
            relation = [self.stringify_relation(ele) if isinstance(ele, tuple)
                        else ele for ele in relation]
            return "(" + " ".join(relation) + ")"
        else:
            return relation

class StandardizeApartNames(Preprocessor):
    """
    A preprocessor that standardizes apart object names.

    >>> _reset_gensym()
    >>> import pprint
    >>> instance = {'nominal': 'v1', 'numeric': 2.3, 'c1': {'a1': 'v1'}, '?c2': {'a2': 'v2', '?c3': {'a3': 'v3'}}, '(relation1 c1 ?c2)': True, 'lists': [{'c1': {'inner': 'val'}}, 's2', 's3'], '(relation2 (a1 c1) (relation3 (a3 (?c3 ?c2))))': 4.3}
    >>> tuplizer = Tuplizer()
    >>> instance = tuplizer.transform(instance)
    >>> std = StandardizeApartNames()
    >>> std.undo_transform(instance)
    Traceback (most recent call last):
        ...
    Exception: Must call transform before undo_transform!
    >>> new_i = std.transform(instance)

    #>>> new_i['?o1']['?o2'][('a4', '?o2')] = 'v4'

    >>> old_i = std.undo_transform(new_i)
    >>> pprint.pprint(instance)
    {'?c2': {'?c3': {'a3': 'v3'}, 'a2': 'v2'},
     'c1': {'a1': 'v1'},
     'lists': [{'c1': {'inner': 'val'}}, 's2', 's3'],
     'nominal': 'v1',
     'numeric': 2.3,
     ('relation1', 'c1', '?c2'): True,
     ('relation2', ('a1', 'c1'), ('relation3', ('a3', ('?c3', '?c2')))): 4.3}
    >>> pprint.pprint(new_i)
    {'?o1': {'?o2': {'a3': 'v3'}, 'a2': 'v2'},
     'c1': {'a1': 'v1'},
     'lists': [{'c1': {'inner': 'val'}}, 's2', 's3'],
     'nominal': 'v1',
     'numeric': 2.3,
     ('relation1', 'c1', '?o1'): True,
     ('relation2', ('a1', 'c1'), ('relation3', ('a3', '?o2'))): 4.3}
    >>> pprint.pprint(old_i)
    {'?c2': {'?c3': {'a3': 'v3'}, 'a2': 'v2'},
     'c1': {'a1': 'v1'},
     'lists': [{'c1': {'inner': 'val'}}, 's2', 's3'],
     'nominal': 'v1',
     'numeric': 2.3,
     ('relation1', 'c1', '?c2'): True,
     ('relation2', ('a1', 'c1'), ('relation3', ('a3', ('?c3', '?c2')))): 4.3}
    """

    def __init__(self):
        self.reverse_mapping = None

    def transform(self, instance):
        """
        Performs the standardize apart tranformation.
        """
        mapping = {}
        new_instance = self.standardize(instance, mapping)
        self.reverse_mapping = {mapping[o]: o for o in mapping}
        return new_instance

    def undo_transform(self, instance):
        """
        Undoes the standardize apart tranformation.
        """
        if self.reverse_mapping is None:
            raise Exception("Must call transform before undo_transform!")

        return self.undo_standardize(instance)

    def undo_standardize(self, instance):
        new_instance = {}

        for attr in instance:
            
            name = attr
            if attr in self.reverse_mapping:
                name = self.reverse_mapping[attr]
                if isinstance(name, tuple):
                    name = name[0]

            if isinstance(instance[attr], dict):
                new_instance[name] = self.undo_standardize(instance[attr])
            elif isinstance(instance[attr], list):
                new_instance[name] = [self.undo_standardize(ele) if
                                      isinstance(ele, dict) else ele for ele in
                                      instance[attr]]
            elif isinstance(attr, tuple):
                temp_rel = rename_relation(attr, self.reverse_mapping)
                new_instance[temp_rel] = instance[attr]
            else:
                new_instance[attr] = instance[attr]
        
        return new_instance

    def standardize(self, instance, mapping={}, prefix=None):
        """
        Given a :ref:`raw instance <raw-instance>` rename all the components so they
        have unique names.

        :warning: relations cannot have dictionaries as values (i.e., cannot be
        subojects).
        :warning: relations can only exist at the top level, not in sub-objects.

        This will rename component attirbutes as well as any occurance of the
        component's name within relation attributes. This renaming is necessary to
        allow for a search between possible mappings without collisions.

        :param instance: An instance to be named apart.
        :param mapping: An existing mapping to add new mappings to; used for
        recursive calls.
        :type instance: :ref:`raw instance <raw-instance>`
        :return: an instance with component attributes renamed
        :rtype: :ref:`standardized instance <standard-instance>`

        >>> _reset_gensym()
        >>> import pprint
        >>> instance = {'nominal': 'v1', 'numeric': 2.3, '?c1': {'a1': 'v1'}, 'c2': {'a2': 'v2', 'c3': {'a3': 'v3'}}, '(relation1 ?c1 c2)': True, 'lists': ['s1', 's2', 's3'], '(relation2 (a1 ?c1) (relation3 (a3 (c2 c3))))': 4.3}
        >>> tuplizer = Tuplizer()
        >>> instance = tuplizer.transform(instance)
        >>> std = StandardizeApartNames()
        >>> standard = std.transform(instance)
        >>> pprint.pprint(standard)
        {'?o1': {'a1': 'v1'},
         'c2': {'a2': 'v2', 'c3': {'a3': 'v3'}},
         'lists': ['s1', 's2', 's3'],
         'nominal': 'v1',
         'numeric': 2.3,
         ('relation1', '?o1', 'c2'): True,
         ('relation2', ('a1', '?o1'), ('relation3', ('a3', ('c2', 'c3')))): 4.3}
        """
        new_instance = {}
        relations = []

        # I had to add the key function to the sort because python apparently can't
        # naturally sort strings and tuples
        #for attr in instance:
        for attr in sorted(instance, key=lambda at: str(at)):

            if prefix is None:
                new_a = attr
            else:
                new_a = (attr, prefix)

            if attr[0] == '?':
                mapping[new_a] = gensym()

            if isinstance(attr, tuple):
                relations.append((attr, instance[attr]))

            elif isinstance(instance[attr], dict):
                name = attr
                if attr[0] == '?':
                    name = mapping[new_a]
                new_instance[name] = self.standardize(instance[attr],
                                                       mapping, new_a)
            elif isinstance(instance[attr], list):
                name = attr
                if attr[0] == '?':
                    name = mapping[new_a]
                new_instance[name] = [self.standardize(ele, mapping, new_a) 
                                       if isinstance(ele, dict) else ele for
                                       ele in instance[attr]]
            else:
                new_instance[attr] = instance[attr]

        for relation, val in relations:
            temp_rel = rename_relation(relation, mapping)
            new_instance[temp_rel] = val

        return new_instance

def rename_relation(relation, mapping):
    """
    Takes a tuplized relational attribute (e.g., ('before', 'o1', 'o2')) and
    a mapping and renames the components based on mapping. This function
    contains a special edge case for handling dot notation which is used in
    standardize apart.

    :param attr: The relational attribute containing components to be renamed
    :type attr: tuple
    :param mapping: A dictionary of mappings between component names
    :type mapping: dict
    :return: A new relational attribute with components renamed
    :rtype: tuple

    >>> relation = ('foo1', 'o1', ('foo2', 'o2', 'o3'))
    >>> mapping = {'o1': 'o100', 'o2': 'o200', 'o3': 'o300'}
    >>> rename_relation(relation, mapping)
    ('foo1', 'o100', ('foo2', 'o200', 'o300'))

    >>> relation = ('foo1', ('o1', ('o2', 'o3')))
    >>> mapping = {('o1', ('o2', 'o3')): 'o100'}
    >>> rename_relation(relation, mapping)
    ('foo1', 'o100')
    """
    new_relation = []

    for v in relation:
        if v in mapping:
            new_relation.append(mapping[v])
        elif isinstance(v, tuple):
            new_relation.append(rename_relation(v, mapping))
        else:
            new_relation.append(v)

    return tuple(new_relation)

def get_component_names(instance):
    """
    Given  a :ref:`flattened instance <flattened-instance>` or a concept's
    probability table return a list of all of the component names.

    :param instance: An instance or a concept's probability table.
    :type instance: :ref:`raw instance <raw-instance>` or dict
    :return: A list of all of the component names present in the instance
    :rtype: [str, str, ...]

    >>> instance = {('a', ('c1',)): 0, ('a', ('c2',)): 0, ('_', '_a', ('c3',)): 0}
    >>> names = get_component_names(instance)
    >>> sorted(list(names))
    ['c1', 'c2', 'c3']
    """
    names = set()
    for attr in instance:
        if not isinstance(attr, tuple):
            continue

        for ele in attr:
            if isinstance(ele, tuple) and len(ele) == 1:
                names.add(ele[0])
            elif isinstance(ele, tuple):
                for name in get_component_names(ele):
                    names.add(name)

    return names

def renameComponent(attr, mapping):
    """Takes a component attribute (e.g., o1.o2) and renames the 
    components given a mapping.

    :param attr: The attribute to be renamed
    :type attr: str
    :param mapping: A dictionary of mappings between component names
    :type mapping: dict
    :return: The attribute's new name
    :rtype: str

    >>> attr = "c1.c2.a"
    >>> mapping = {'c1': 'o1', 'c2': 'o2'}
    >>> renameComponent(attr, mapping)
    'o1.o2.a'

    >>> attr = "_c1._c2._a"
    >>> mapping = {'c1': 'o1', 'c2': 'o2'}
    >>> renameComponent(attr, mapping)
    '_o1._o2._a'
    """
    new_attr = []
    att_split = attr.split('.')
    for name in att_split[:-1]:
        if name[0] == "_":
            new_attr.append(mapping[name[1:]])
        else:
            new_attr.append(mapping[name])

    if att_split[-1][0] == "_":
        new_attr.append(att_split[-1][1:])
    else:
        new_attr.append(att_split[-1])

    if attr[0] == "_":
        return "_" + "._".join(new_attr)
    else:
        return ".".join(new_attr)

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
    >>> instance = {('a', ('c1',)): 1, ('good', ('c1',)): True}
    >>> mapping = {'c1': 'o1'}
    >>> renamed = rename_flat(instance,mapping)
    >>> pprint.pprint(renamed)
    {('a', ('o1',)): 1, ('good', ('o1',)): True}
    """
    temp_instance = {}

    for attr in instance:
        if isinstance(attr, tuple):
            temp_instance[rename_relation(attr, mapping)] = instance[attr]
        else:
            temp_instance[attr] = instance[attr]

    return temp_instance

def flatten_json(instance):
    """
    Takes a :ref:`raw instance <raw-instance>` that has already been
    standardized apart and flattens it.

    :warning: important to note that relations can only exist at the top level,
    not within subobjects. If they do exist than this function will return
    incorrect results.

    Hierarchy is represented with periods between variable names in the
    flattened attributes. However, this process converts the attributes with
    periods in them into a tuple of objects with an attribute as the last
    element, this is more efficient for later processing.

    :param instance: An instance to be flattened.
    :type instance: :ref:`raw instance <raw-instance>`
    :return: A copy of the instance flattend
    :rtype: :ref:`flattened instance <flattened-instance>`

    >>> import pprint
    >>> instance = {'a': 1, 'c1': {'b': 1, '_c': 2}}
    >>> flat = flatten_json(instance)
    >>> pprint.pprint(flat)
    {'a': 1, ('_', ('_c', ('c1',))): 2, ('b', ('c1',)): 1}
    """
    temp = {}
    for attr in instance:
        if isinstance(instance[attr], dict):
            for so_attr in instance[attr]:
                if so_attr[0] == '_':
                    new_attr = ('_', (so_attr, (attr,)))
                else:
                    new_attr = (so_attr, (attr,))
                temp[new_attr] = instance[attr][so_attr]
        else:
            temp[attr] = instance[attr]
    return temp

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

    >>> attr = ('before', ('c1',), ('c2',))
    >>> mapping = {'c1': 'o1', 'c2':'o2'}
    >>> bind_flat_attr(attr, mapping)
    ('before', ('o1',), ('o2',))

    If the mapping is incomplete then returns ``None`` (nothing) 

    >>> attr = ('before', ('c1',), ('c2',))
    >>> mapping = {'c1': 'o1'}
    >>> bind_flat_attr(attr, mapping) is None
    True

    >>> bind_flat_attr(('<', ('a', ('o2',)), ('a', ('o1',))), {'o1': 'c1'}) is None
    True

    >>> bind_flat_attr(('<', ('a', ('o2',)), ('a', ('o1',))), {'o1': 'c1', 'o2': 'c2'}) is None
    False
    """
    if not isinstance(attr, tuple):
        return attr

    if isinstance(attr, tuple) and len(attr) == 1:
        if attr[0] not in mapping:
            return None
        else:
            return (mapping[attr[0]],)

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

    >>> contains_component('c1', ('relation', ('c2',), ('a', ('c1',))))
    True
    >>> contains_component('c3', ('before', ('c1',), ('c2',)))
    False
    """
    if isinstance(attr, tuple) and len(attr) == 1:
        return component == attr[0]

    elif isinstance(attr, tuple):
        for ele in attr:
            if contains_component(component, ele) is True:
                return True
    
    return False

def flat_match(concept, instance, optimal=False):
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
    :param optimal: If True the mapping will be optimal (using A* Search)
    otherwise it will be greedy (using Beam Search).
    :type optimal: bool
    :return: a mapping for renaming components in the instance.
    :rtype: dict

    """
    inames = frozenset(get_component_names(instance))
    cnames = frozenset(get_component_names(concept.av_counts))

    if(len(inames) == 0 or len(cnames) == 0):
        return {}

    initial = search.Node((frozenset(), inames, cnames), extra=(concept,
                                                                instance))
    if optimal:
        solution = next(search.BestFGS(initial, _flat_match_successor_fn,
                                       _flat_match_goal_test_fn,
                                       _flat_match_heuristic_fn))
    else:
        solution = next(search.BeamGS(initial, _flat_match_successor_fn,
                                      _flat_match_goal_test_fn,
                                      _flat_match_heuristic_fn,
                                      initialBeamWidth=1))
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

    >>> is_partial_match(('<', ('a', ('o2',)), ('a', ('o1',))), ('<', ('a', ('c2',)), ('b', ('c1',))), {'o1': 'c1'}, {'o2'})
    False

    >>> is_partial_match(('<', ('a', ('o2',)), ('a', ('o1',))), ('<', ('a', ('c2',)), ('a', ('c1',))), {'o1': 'c1'}, {'o2'})
    True

    >>> is_partial_match(('<', ('a', ('o2',)), ('a', ('o1',))), ('<', ('a', ('c2',)), ('a', ('c1',))), {'o1': 'c1', 'o2': 'c2'}, {})
    True
    """
    if type(iAttr) != type(cAttr):
        return False

    if isinstance(iAttr, tuple) and len(iAttr) != len(cAttr):
        return False

    if isinstance(iAttr, tuple) and len(iAttr) == 1:
        if iAttr[0] in unnamed:
            return True
        elif iAttr[0] in mapping:
            return mapping[iAttr[0]] == cAttr[0]

    if isinstance(iAttr, tuple):
        for i,v in enumerate(iAttr):
            if not is_partial_match(iAttr[i], cAttr[i], mapping, unnamed):
                return False
        return True

    return iAttr == cAttr

class ListProcessor(Preprocessor):
    """
    Preprocesses out the lists, converting them into objects and relations.
    """
    def __init__(self):
        self.processor = Pipeline(ExtractListElements(), ListsToRelations())

    def transform(self, instance):
        return self.processor.transform(instance)        

    def undo_transform(self, instance):
        self.processor.undo_transform(instance)

class ExtractListElements(Preprocessor):
    """
    A pre-processor that extracts the elements of lists into their own objects

    >>> _reset_gensym()
    >>> import pprint
    >>> instance = {"att1":"V1","list1":["a","b","c",{"B":"C","D":"E"}]}
    >>> pprint.pprint(instance)
    {'att1': 'V1', 'list1': ['a', 'b', 'c', {'B': 'C', 'D': 'E'}]}
    >>> pp = ExtractListElements()
    >>> instance = pp.transform(instance)
    >>> pprint.pprint(instance)
    {'?o1': {'val': 'a'},
     '?o2': {'val': 'b'},
     '?o3': {'val': 'c'},
     '?o4': {'B': 'C', 'D': 'E'},
     'att1': 'V1',
     'list1': ['?o1', '?o2', '?o3', '?o4']}
    >>> instance = pp.undo_transform(instance)
    >>> pprint.pprint(instance)
    {'att1': 'V1', 'list1': ['a', 'b', 'c', {'B': 'C', 'D': 'E'}]}

    """
    def transform(self, instance):
        """
        Peforms the list element extraction operation.
        """
        new_instance = self.extract(instance)
        return new_instance

    def undo_transform(self, instance):
        """
        Undoes the list element extraction operation.
        """
        return self.undo_extract(instance)

    def undo_extract(self,instance):
        """
        Reverses the list element extraction process
        """
        new_instance = {}
        lists = {}
        elements = {}

        for a in instance:
            if isinstance(instance[a],list):
                lists[a] = True
                new_list = []
                for i in range(len(instance[a])):
                    elements[instance[a][i]] = True
                    obj = instance[instance[a][i]]

                    if len(obj) > 1 and "val" not in obj:
                        new_list.append(obj)
                    else :
                        new_list.append(obj["val"])
                new_instance[a] = new_list

        for a in instance:
            if isinstance(instance[a],list) or a in elements:
                continue
            new_instance[a] = instance[a]

        return new_instance

    def extract(self,instance):
        """
        Find all lists in an instance and extract their elements into their own
        subjects of the main instance.

        Unlike the utils.extract_components function this one will extract ALL
        elements into their own objects not just object literals

        >>> _reset_gensym()
        >>> import pprint
        >>> instance = {"a":"n","list1":["test",{"p":"q","j":"k"},{"n":"m"}]}
        >>> pp = ExtractListElements()
        >>> instance = pp.extract(instance)
        >>> pprint.pprint(instance)
        {'?o1': {'val': 'test'},
         '?o2': {'j': 'k', 'p': 'q'},
         '?o3': {'n': 'm'},
         'a': 'n',
         'list1': ['?o1', '?o2', '?o3']}
        """
        new_instance = {}
        for a in instance.keys():
            if isinstance(instance[a],list):

                if a[0] == '_':
                    new_instance[a] = str(instance[a])
                    continue

                new_list = []
                for el in instance[a]:
                    
                    # TODO do we want to deep copy in the case we find a dict?
                    if isinstance(el,dict):
                        new_obj = el
                    else :
                        new_obj = {"val": el}

                    new_att = gensym()
                    new_instance[new_att] = self.extract(new_obj)
                    new_list.append(new_att)

                new_instance[a] = new_list

            elif isinstance(instance[a],dict):
                new_instance[a] = self.extract(instance[a])
            else :
                new_instance[a] = instance[a]

        return new_instance

class ListsToRelations(Preprocessor):
    """
    Converts an object with lists into an object with sub-objects and list
    relations.

    >>> _reset_gensym()
    >>> ltr = ListsToRelations()
    >>> import pprint
    >>> instance = {"list1":['c','b','a']}
    >>> instance = ltr.transform(instance)
    >>> pprint.pprint(instance)
    {'list1': {},
     ('has-element', 'list1', 'a'): True,
     ('has-element', 'list1', 'b'): True,
     ('has-element', 'list1', 'c'): True,
     ('ordered-list', 'list1', 'b', 'a'): True,
     ('ordered-list', 'list1', 'c', 'b'): True}

    >>> instance = ltr.undo_transform(instance)
    >>> pprint.pprint(instance)
    {'list1': ['c', 'b', 'a']}
    """
    def transform(self, instance):
        return self.lists_to_relations(instance)

    def undo_transform(self, instance):
        return self.relations_to_lists(instance)

    def relations_to_lists(self, instance, path=None):
        """
        Traverse the instance and turn any list relations into a list. If there
        is a total ordering with no cycles than this should work perfectly. In
        the event of a cycle or a partial ordering than this might return
        incorrect results, just be careful.
        """
        new_instance = {}

        elements = {}
        order = {}

        for attr in instance:
            if isinstance(attr, tuple) and (attr[0] == 'has-element'):
                if attr[1] not in elements:
                    elements[attr[1]] = []
                elements[attr[1]].append(attr[2])
            
            elif isinstance(attr, tuple) and attr[0] == 'ordered-list':
                rel, lname, ele1, ele2 = attr

                if lname not in order:
                    order[lname] = []

                order[lname].append((ele1, ele2))

            else:
                new_instance[attr] = instance[attr]

        for l in elements:
            all_lists = []

            while len(elements[l]) > 0:
                new_list = [elements[l].pop(0)]
            
                # chain to front
                change = True
                while change is not None:
                    change = None
                    for a,b in order[l]:
                        if b == new_list[0]:
                            change = (a,b)
                            elements[l].remove(a)
                            new_list.insert(0, a)
                            break
                    if change is not None:
                        order[l].remove(change)
                
                # chain to end
                change = True
                while change is not None:
                    change = None
                    for a,b in order[l]:
                        if a == new_list[-1]:
                            change = (a,b)
                            elements[l].remove(b)
                            new_list.append(b)
                            break
                    if change is not None:
                        order[l].remove(change)

                all_lists.append(new_list)

            new_list = [ele for sub_list in all_lists for ele in sub_list]
            
            path = self.get_path(l)
            current = new_instance
            while len(path) > 1:
                current = new_instance[path.pop(0)]
            current[path[0]] = new_list

        return new_instance

    def remove_cycles(self, list_order):
        pass

    def get_path(self, path):
        if isinstance(path, tuple):
            return self.get_path(path[1]).append(path[0])
        else:
            return [path]

    def lists_to_relations(self, instance, current=None, top_level=None):
        """
        Travese the instance and turn any list elements into 
        a series of relations.

        >>> _reset_gensym()
        >>> ltr = ListsToRelations()
        >>> import pprint
        >>> instance = {"list1":['a','b','c']}
        >>> instance = ltr.lists_to_relations(instance)
        >>> pprint.pprint(instance)
        {'list1': {},
         ('has-element', 'list1', 'a'): True,
         ('has-element', 'list1', 'b'): True,
         ('has-element', 'list1', 'c'): True,
         ('ordered-list', 'list1', 'a', 'b'): True,
         ('ordered-list', 'list1', 'b', 'c'): True}
        
        >>> instance = {"list1":['a','b','c'],"list2":['w','x','y','z']}
        >>> instance = ltr.lists_to_relations(instance)
        >>> pprint.pprint(instance)
        {'list1': {},
         'list2': {},
         ('has-element', 'list1', 'a'): True,
         ('has-element', 'list1', 'b'): True,
         ('has-element', 'list1', 'c'): True,
         ('has-element', 'list2', 'w'): True,
         ('has-element', 'list2', 'x'): True,
         ('has-element', 'list2', 'y'): True,
         ('has-element', 'list2', 'z'): True,
         ('ordered-list', 'list1', 'a', 'b'): True,
         ('ordered-list', 'list1', 'b', 'c'): True,
         ('ordered-list', 'list2', 'w', 'x'): True,
         ('ordered-list', 'list2', 'x', 'y'): True,
         ('ordered-list', 'list2', 'y', 'z'): True}

        >>> instance = {"stack":[{"a":1, "b":2, "c":3}, {"x":1, "y":2, "z":3}, {"i":1, "j":2, "k":3}]}
        >>> ele = ExtractListElements()
        >>> instance = ele.extract(instance)
        >>> instance = ltr.lists_to_relations(instance)
        >>> pprint.pprint(instance)
        {'?o1': {'a': 1, 'b': 2, 'c': 3},
         '?o2': {'x': 1, 'y': 2, 'z': 3},
         '?o3': {'i': 1, 'j': 2, 'k': 3},
         'stack': {},
         ('has-element', 'stack', '?o1'): True,
         ('has-element', 'stack', '?o2'): True,
         ('has-element', 'stack', '?o3'): True,
         ('ordered-list', 'stack', '?o1', '?o2'): True,
         ('ordered-list', 'stack', '?o2', '?o3'): True}

        >>> instance = {'subobj': {'list1': ['a', 'b', 'c']}}
        >>> instance = ltr.lists_to_relations(instance)
        >>> pprint.pprint(instance)
        {'subobj': {'list1': {}},
         ('has-component', 'subobj', ('list1', 'subobj')): True,
         ('has-element', ('list1', 'subobj'), 'a'): True,
         ('has-element', ('list1', 'subobj'), 'b'): True,
         ('has-element', ('list1', 'subobj'), 'c'): True,
         ('ordered-list', ('list1', 'subobj'), 'a', 'b'): True,
         ('ordered-list', ('list1', 'subobj'), 'b', 'c'): True}

        >>> _reset_gensym()
        >>> instance = {'tta':'alpha','ttb':{'tlist':['a','b',{'sub-a':'c','sub-sub':{'s':'d','sslist':['w','x','y',{'issue':'here'}]}},'g']}}
        >>> ele = ExtractListElements()
        >>> instance = ele.extract(instance)
        >>> pprint.pprint(instance)
        {'tta': 'alpha',
         'ttb': {'?o1': {'val': 'a'},
                 '?o2': {'val': 'b'},
                 '?o3': {'sub-a': 'c',
                         'sub-sub': {'?o4': {'val': 'w'},
                                     '?o5': {'val': 'x'},
                                     '?o6': {'val': 'y'},
                                     '?o7': {'issue': 'here'},
                                     's': 'd',
                                     'sslist': ['?o4', '?o5', '?o6', '?o7']}},
                 '?o8': {'val': 'g'},
                 'tlist': ['?o1', '?o2', '?o3', '?o8']}}
                
        >> instance = ltr.lists_to_relations(instance)
        >> pprint.pprint(instance)
        {'tta': 'alpha',
         'ttb': {'?o1': {'val': 'a'},
                 '?o2': {'val': 'b'},
                 '?o3': {'sub-a': 'c',
                         'sub-sub': {'?o4': {'val': 'w'},
                                     '?o5': {'val': 'x'},
                                     '?o6': {'val': 'y'},
                                     '?o7': {'issue': 'here'},
                                     's': 'd',
                                     'sslist': {}}},
                 '?o8': {'val': 'g'},
                 'tlist': {}},
         ('has-component', 'ttb', ('tlist', 'ttb')): True,
         ('has-component', ('sub-sub', ('?o3', 'ttb')), ('sslist', ('sub-sub', ('?o3', 'ttb')))): True,
         ('has-element', ('sslist', ('sub-sub', ('?o3', 'ttb'))), '?o4'): True,
         ('has-element', ('sslist', ('sub-sub', ('?o3', 'ttb'))), '?o5'): True,
         ('has-element', ('sslist', ('sub-sub', ('?o3', 'ttb'))), '?o6'): True,
         ('has-element', ('sslist', ('sub-sub', ('?o3', 'ttb'))), '?o7'): True,
         ('has-element', ('tlist', 'ttb'), '?o1'): True,
         ('has-element', ('tlist', 'ttb'), '?o2'): True,
         ('has-element', ('tlist', 'ttb'), '?o3'): True,
         ('has-element', ('tlist', 'ttb'), '?o8'): True,
         ('ordered-list', ('sslist', ('sub-sub', ('?o3', 'ttb'))), '?o4', '?o5'): True,
         ('ordered-list', ('sslist', ('sub-sub', ('?o3', 'ttb'))), '?o5', '?o6'): True,
         ('ordered-list', ('sslist', ('sub-sub', ('?o3', 'ttb'))), '?o6', '?o7'): True,
         ('ordered-list', ('tlist', 'ttb'), '?o1', '?o2'): True,
         ('ordered-list', ('tlist', 'ttb'), '?o2', '?o3'): True,
         ('ordered-list', ('tlist', 'ttb'), '?o3', '?o8'): True}

        >> instance = {"Function Defintion":{"body":[{"Return":{"value":{"Compare":{"left":{"Number":{"n":2 } }, "ops":[{"<":{}},{"<=":{}}],"comparators":[{"Name":{"id":"daysPassed","ctx":{"Load":{}}}},{"Number":{"n":9}}]}}}}]}}
        >> ele = ExtractListElements()
        >> instance = ele.extract(instance)
        >> pprint.pprint(instance)
        {'Function Defintion': {'?o9': {'Return': {'value': {'Compare': {'?o10': {'<': {}},
                                                                         '?o11': {'<=': {}},
                                                                         '?o12': {'Name': {'ctx': {'Load': {}},
                                                                                           'id': 'daysPassed'}},
                                                                         '?o13': {'Number': {'n': 9}},
                                                                         'comparators': ['?o12',
                                                                                         '?o13'],
                                                                         'left': {'Number': {'n': 2}},
                                                                         'ops': ['?o10',
                                                                                 '?o11']}}}},
                                'body': ['?o9']}}
        >> instance = ltr.lists_to_relations(instance)
        >> pprint.pprint(instance)
        {'Function Defintion': {'?o9': {'Return': {'value': {'Compare': {'?o10': {'<': {}},
                                                                         '?o11': {'<=': {}},
                                                                         '?o12': {'Name': {'ctx': {'Load': {}},
                                                                                           'id': 'daysPassed'}},
                                                                         '?o13': {'Number': {'n': 9}},
                                                                         'comparators': {},
                                                                         'left': {'Number': {'n': 2}},
                                                                         'ops': {}}}}},
                                'body': {}},
         ('has-component', 'Function Defintion', ('body', 'Function Defintion')): True,
         ('has-component', ('Compare', ('value', ('Return', ('?o9', 'Function Defintion')))), ('comparators', ('Compare', ('value', ('Return', ('?o9', 'Function Defintion')))))): True,
         ('has-component', ('Compare', ('value', ('Return', ('?o9', 'Function Defintion')))), ('ops', ('Compare', ('value', ('Return', ('?o9', 'Function Defintion')))))): True,
         ('has-element', ('body', 'Function Defintion'), '?o9'): True,
         ('has-element', ('comparators', ('Compare', ('value', ('Return', ('?o9', 'Function Defintion'))))), '?o12'): True,
         ('has-element', ('comparators', ('Compare', ('value', ('Return', ('?o9', 'Function Defintion'))))), '?o13'): True,
         ('has-element', ('ops', ('Compare', ('value', ('Return', ('?o9', 'Function Defintion'))))), '?o10'): True,
         ('has-element', ('ops', ('Compare', ('value', ('Return', ('?o9', 'Function Defintion'))))), '?o11'): True,
         ('ordered-list', ('comparators', ('Compare', ('value', ('Return', ('?o9', 'Function Defintion'))))), '?o12', '?o13'): True,
         ('ordered-list', ('ops', ('Compare', ('value', ('Return', ('?o9', 'Function Defintion'))))), '?o10', '?o11'): True}
        """
        new_instance = {}
        if top_level is None:
            top_level = new_instance

        for attr in instance.keys():
            if current is None:
                lname = attr
            else:
                lname = (attr, current)

            if isinstance(instance[attr], list):
                new_instance[attr] = {}

                for i in range(len(instance[attr])-1):
                    rel = ("ordered-list", lname, str(instance[attr][i]),
                           str(instance[attr][i+1]))
                    top_level[rel] = True

                    rel = ("has-element", lname, instance[attr][i])
                    top_level[rel] = True

                if len(instance[attr]) > 0:
                    rel = ('has-element', lname, instance[attr][-1])
                    top_level[rel] = True

                if isinstance(lname, tuple):
                    rel = ('has-component', current, lname)
                    top_level[rel] = True

            elif isinstance(instance[attr],dict):
                new_instance[attr] = self.lists_to_relations(instance[attr],
                                                        lname,
                                                        top_level)
            else:
                new_instance[attr] = instance[attr]
        
        return new_instance

def hoist_sub_objects(instance):
    """
    Travese the instance for objects that contain subobjects and hoists the
    subobjects to be their own objects at the top level of the instance. 
    
    >>> _reset_gensym()
    >>> import pprint
    >>> instance = {"a1":"v1","sub1":{"a2":"v2","a3":3},"sub2":{"a4":"v4","subsub1":{"a5":"v5","a6":"v6"},"subsub2":{"subsubsub":{"a8":"V8"},"a7":7}}}
    >>> pprint.pprint(instance)
    {'a1': 'v1',
     'sub1': {'a2': 'v2', 'a3': 3},
     'sub2': {'a4': 'v4',
              'subsub1': {'a5': 'v5', 'a6': 'v6'},
              'subsub2': {'a7': 7, 'subsubsub': {'a8': 'V8'}}}}

    >>> instance = hoist_sub_objects(instance)
    >>> pprint.pprint(instance)
    {'a1': 'v1',
     'sub1': {'a2': 'v2', 'a3': 3},
     'sub2': {'a4': 'v4'},
     'subsub1': {'a5': 'v5', 'a6': 'v6'},
     'subsub2': {'a7': 7},
     'subsubsub': {'a8': 'V8'},
     ('has-component', ('sub2',), ('subsub1',)): True,
     ('has-component', ('sub2',), ('subsub2',)): True,
     ('has-component', ('subsub2',), ('subsubsub',)): True}
    """
    new_instance = {}
    
    for a in instance.keys() :
        # this is a subobject
        if isinstance(instance[a],dict):
            new_instance[a] = _hoist_sub_objects_rec(instance[a],a,new_instance)
        else :
            new_instance[a] = instance[a]

    return new_instance

def _hoist_sub_objects_rec(sub,attr,top_level):
    """
    The recursive version of subobject hoisting.
    """
    new_sub = {}
    for a in sub.keys():
        # this is a sub-sub object
        if isinstance(sub[a],dict):
            top_level[a] = _hoist_sub_objects_rec(sub[a],a,top_level)
            rel = tuplize_elements(
                ("has-component",
                   str(attr),
                   str(a)))
                #,{str(attr), str(a)})
            top_level[rel] = True
        else :
            new_sub[a] = sub[a]
    return new_sub


def pre_process(instance):
    """
    Runs all of the pre-processing functions

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
     ('has-component', ('o7',), ('o8',)): True,
     ('i', ('o12',)): 1,
     ('j', ('o12',)): 12.3,
     ('k', ('o12',)): 'test',
     ('nomb', ('o7',)): 'b',
     ('nomc', ('o8',)): 'c',
     ('nomd', ('o9',)): 'd',
     ('nome', ('o9',)): 'e',
     ('num4', ('o7',)): 4,
     ('num5', ('o8',)): 5,
     ('ordered-list', 'list1', ('o10',), ('o11',)): True,
     ('ordered-list', 'list1', ('o11',), ('o12',)): True,
     ('related', ('num4', ('o7',)), ('nome', ('o9',))): True,
     ('val', ('o10',)): 'a',
     ('val', ('o11',)): 'b'}

    >>> instance = pre_process(instance)
    >>> pprint.pprint(instance)
    {'noma': 'a',
     'num3': 3,
     ('has-component', ('o7',), ('o8',)): True,
     ('i', ('o12',)): 1,
     ('j', ('o12',)): 12.3,
     ('k', ('o12',)): 'test',
     ('nomb', ('o7',)): 'b',
     ('nomc', ('o8',)): 'c',
     ('nomd', ('o9',)): 'd',
     ('nome', ('o9',)): 'e',
     ('num4', ('o7',)): 4,
     ('num5', ('o8',)): 5,
     ('ordered-list', 'list1', ('o10',), ('o11',)): True,
     ('ordered-list', 'list1', ('o11',), ('o12',)): True,
     ('related', ('num4', ('o7',)), ('nome', ('o9',))): True,
     ('val', ('o10',)): 'a',
     ('val', ('o11',)): 'b'}
    
    """
    tuplizer = Tuplizer()
    instance = tuplizer.transform(instance)

    list_processor = ListProcessor()
    instance = list_processor.transform(instance)

    standardizer = StandardizeApartNames()
    instance = standardizer.transform(instance)
    
    instance = hoist_sub_objects(instance)
    instance = flatten_json(instance)
    return instance

def structure_map(concept, instance):
    """Flatten the instance, perform structure mapping to the concept, rename
    the instance based on this structure mapping, and return the renamed
    instance.

    :param concept: A concept to structure map the instance to
    :type concept: TrestleNode
    :param instance: An instance to map to the concept
    :type instance: :ref:`raw instance <raw-instance>`
    :return: A fully mapped and flattend copy of the instance
    :rtype: :ref:`mapped instance <fully-mapped>`

    """
    instance = pre_process(instance)
    mapping = flat_match(concept, instance)
    instance = rename_flat(instance, mapping)
    return instance

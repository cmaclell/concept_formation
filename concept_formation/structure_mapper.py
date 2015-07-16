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
    """Generates unique names for naming renaming apart objects.

    :return: a unique object name
    :rtype: 'o'+counter
    """
    global _gensym_counter
    _gensym_counter += 1
    return 'o' + str(_gensym_counter)

def standardize_apart_names(instance, mapping = {}):
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

    >>> instance = {'nominal': 'v1', 'numeric': 2.3, 'c1': {'a1': 'v1'}, 'c2': {'a2': 'v2'}, '(relation1 c1 c2)': True, 'lists': ['s1', 's2', 's3'], '(relation2 c1 (relation3 c2))': 4.3}
    >>> standard = standardize_apart_names(instance)
    >>> doctest_print(standard)
    {'lists': [
        's1',
        's2',
        's3'
    ],
    'nominal': 'v1',
    'numeric': 2.3,
    'o13': {'a1': 'v1'},
    'o14': {'a2': 'v2'},
    (('relation1',), ('o13', ''), ('o14', '')): True,
    (('relation2',), ('o13', ''), (('relation3',), ('o14', ''))): 4.3}
    """
    new_instance = {}
    relations = []

    # I had to add the key function to the sort because python apparently can't
    # naturally sort strings nad tuples
    for attr in sorted(instance, key=lambda at: str(at)):
        if attr[0] == '(':
            relations.append((attr, instance[attr]))
        elif isinstance(instance[attr], dict):
            mapping[attr] = gensym()
            new_instance[mapping[attr]] = standardize_apart_names(instance[attr], mapping)
        else:
            new_instance[attr] = instance[attr]

    for relation, val in relations:
        new_instance[rename_relation(tuplize_relation(relation, mapping), mapping)] = val

    return new_instance

def tuplize_relation_elements(elements, mapping):
    """
    Converts a relation element into a tuple for efficient
    processing.
    
    >>> mapping = {'o1': 'c1', 'o2': 'c2'}
    >>> ele1 = 'o1'
    >>> tuplize_relation_elements(ele1, mapping)
    ('o1', '')
    >>> ele2 = "o1.o2.a"
    >>> tuplize_relation_elements(ele2, mapping)
    ('o1', 'o2', 'a')
    >>> ele3 = ('o1', ('o1.o2.a',))
    >>> tuplize_relation_elements(ele3, mapping)
    (('o1', ''), (('o1', 'o2', 'a'),))
    """
    if isinstance(elements, tuple):
        return tuple([tuplize_relation_elements(ele, mapping) for ele in elements])
    
    elements = elements.split('.')
    if elements[-1] in mapping:
        elements.append('')

    return tuple(elements)

def tuplize_relation(relation, mapping={}) :
    """
    Converts a string formatted relation into a tuplized relation. It requires
    the mapping so that it can convert the period separated object references
    correctly into presplit tuples for efficient processing.

    :param attr: The relational attribute formatted as a string
    :type attr: string
    :param mapping: A dictionary of mappings with component names as keys. Just
    the keys are used (i.e., as a set) to determine if elements in the relation
    are objects.
    :type mapping: dict
    :return: A new relational attribute in tuple format
    :rtype: tuple

    >>> relation = '(foo1 o1 (foo2 o2 o3))'
    >>> mapping = {'o1': 'sk1', 'o2': 'sk2', 'o3': 'sk3'}
    >>> tuplize_relation(relation, mapping)
    (('foo1',), ('o1', ''), (('foo2',), ('o2', ''), ('o3', '')))
    """
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
    final = tuplize_relation_elements(final, mapping)
    return final

def stringify_relation(relation):
    """
    Converts a tupleized relation into a string formated relation.

    >>> relation = ('foo1', 'o1', ('foo2', 'o2', 'o3'))
    >>> stringify_relation(relation)
    '(foo1 o1 (foo2 o2 o3))'
    """
    temp = [stringify_relation(ele) if isinstance(ele, tuple) else ele for ele in relation]
    return "(" + " ".join(temp) + ")"

def rename_relation(relation, mapping):
    """
    Takes a tuplized relational attribute (e.g., ('before', 'o1', 'o2')) and
    a mapping and renames the components based on mapping.

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
    """
    new_relation = []

    for v in relation:
        if isinstance(v, tuple):
            new_relation.append(rename_relation(v, mapping))
        #elif "." in v:
        #    new_v = []
        #    for ele in v.split("."):
        #        if ele in mapping:
        #            new_v.append(mapping[ele])
        #        else:
        #            new_v.append(ele)
        #    new_relation.append(".".join(new_v))
        else:
            if v in mapping:
                new_relation.append(mapping[v])
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

    >>> instance = {('c1', 'a'): 0, ('c2','a'): 0, ('_c3', '_a'): 0}
    >>> names = get_component_names(instance)
    >>> sorted(names)
    ['c1', 'c2', 'c3']
    """
    names = set()
    for attr in instance:
        if isinstance(attr, tuple) and isinstance(attr[0], tuple):
            continue
        if isinstance(attr, tuple):
            for ele in attr[:-1]:
                if ele[0] == '_':
                    names.add(ele[1:])
                else:
                    names.add(ele)

    return list(names)

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

def renameRelation(attr, mapping):
    """Takes a relational attribute (e.g., (before o1 o2)) and renames
    the components based on mapping.

    :param attr: The relational attribute containing components to be renamed
    :type attr: tuple
    :param mapping: A dictionary of mappings between component names
    :type mapping: dict
    :return: A new relational attribute with components renamed
    :rtype: tuple

    >>> attr = ('before', 'c1', 'c2')
    >>> mapping =  {'c1': 'o1', 'c2': 'o2'}
    >>> renameRelation(attr, mapping)
    ('before', 'o1', 'o2')
    """
    temp = []
    for idx, val in enumerate(attr):
        if idx == 0:
            temp.append(val)
        else:
            new_attr = [mapping[name] if name in mapping else name for name in
                        val.split(".")]
            temp.append(".".join(new_attr))
    return tuple(temp)

def renameFlat(instance, mapping):
    """Given a :ref:`flattened instance <flattened-instance>` and a mapping (type =
    dict) rename the components and relations and return the renamed instance.

    :param instance: An instance to be renamed according to a mapping
    :type instance: :ref:`flattened instance <flattened-instance>`
    :param mapping: :param mapping: A dictionary of mappings between component names
    :type mapping: dict
    :return: A copy of the instance with components and relations renamed
    :rtype: :ref:`mapped instance <fully-mapped>`

    >>> instance = {'c1.a': 1, ('good', 'c1'): True}
    >>> mapping = {'c1': 'o1'}
    >>> renamed = renameFlat(instance,mapping)
    >>> doctest_print(renamed)
    {'o1.a': 1,
    ('good', 'o1'): True}
    """
    for attr in instance:
        if isinstance(attr, tuple):
            continue
        for name in attr.split('.')[:-1]:
            if name[0] == "_":
                name = name[1:]
            if name not in mapping:
                mapping[name] = name

    temp_instance = {}

    for attr in instance:
        if isinstance(attr, tuple):
            temp_instance[renameRelation(attr, mapping)] = instance[attr]
        elif "." in attr:
            temp_instance[renameComponent(attr, mapping)] = instance[attr]
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

    >>> instance = {'a': 1, 'c1': {'b': 1, '_c': 2}}
    >>> flat = flatten_json(instance)
    >>> doctest_print(flat)
    {'a': 1,
    ('_c1', '_c'): 2,
    ('c1', 'b'): 1}
    """
    temp = {}
    for attr in instance:
        if isinstance(instance[attr], dict):
            subobject = flatten_json(instance[attr])
            for so_attr in subobject:
                if isinstance(so_attr, tuple):
                    if so_attr[0][0] == '_':
                        new_attr = ('_' + attr) + so_attr
                    else:
                        new_attr = (attr) + so_attr
                elif so_attr[0] == '_':
                    new_attr = ('_' + attr, so_attr)
                else:
                    new_attr = (attr, so_attr)
                temp[new_attr] = subobject[so_attr]
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

def bind_flat_attr(attr, mapping, unnamed):
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

    >>> attr = (('before',), ('c1',''), ('c2',''))
    >>> mapping = {'c1': 'o1', 'c2':'o2'}
    >>> bind_flat_attr(attr, mapping, {})
    (('before',), ('o1', ''), ('o2', ''))

    If the mapping is incomplete then returns ``None`` (nothing) 

    >>> attr = (('before',), ('c1',''), ('c2',''))
    >>> mapping = {'c1': 'o1'}
    >>> bind_flat_attr(attr, mapping, {'c2'}) is None
    True

    >>> bind_flat_attr((('<',), ('o2','a'), ('o1','a')), {'o1': 'c1'}, {'o2'}) is None
    True

    >>> bind_flat_attr((('<',), ('o2','a'), ('o1','a')), {'o1': 'c1', 'o2': 'c2'}, {}) is None
    False
    """
    for o in unnamed:
        if contains_component(o, attr):
            return None

    if isinstance(attr, tuple):
        return rename_relation(attr, mapping)
    else:
        return attr

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

    >>> contains_component('c1', ('c2', 'c1', 'a'))
    True
    >>> contains_component('c3', ('before', 'c1', 'c2'))
    False
    """
    if isinstance(attr, tuple) and isinstance(attr[0], tuple):
        for ele in attr:
            if contains_component(component, ele) is True:
                return True

    elif isinstance(attr, tuple):
        for ele in attr[:-1]:
            if contains_component(component, ele) is True:
                return True

    elif component == attr:
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
    :param optimal: If True the mapping will be optimal (using A* Search) otherwise it will be greedy (using Beam Search).
    :type optimal: bool
    :return: a mapping for renaming components in the instance.
    :rtype: dict

    """
    inames = frozenset(get_component_names(instance))
    cnames = frozenset(get_component_names(concept.av_counts))

    if(len(inames) == 0 or
       len(cnames) == 0):
        return {}

    initial = search.Node((frozenset(), inames, cnames), extra=(concept,
                                                                instance))
    if optimal:
        solution = next(search.BestFGS(initial, _flat_match_successor_fn, _flatMatchGoalTestFn,
                                _flatMatchHeuristicFn))
    else:
        solution = next(search.BeamGS(initial, _flat_match_successor_fn, _flatMatchGoalTestFn,
                           _flatMatchHeuristicFn, initialBeamWidth=1))
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
            new_attr = bindFlatAttr(attr, m, inames)
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
                if not containsComponent(n, attr):
                    continue
                new_attr = bindFlatAttr(attr, m, inames)
                if new_attr:
                    reward += concept.attr_val_guess_gain(new_attr,
                                                          instance[attr])
            yield search.Node((mapping.union(frozenset([(n, new)])), inames -
                                      frozenset([n]), availableNames -
                                      frozenset([new])), node, n + ":" + new,
                        node.cost - reward, node.depth + 1, node.extra)


def _flatMatchHeuristicFn(node):
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
        new_attr = bindFlatAttr(attr, m, unnamed)
        if not new_attr:
            best_attr_h = [concept.attr_val_guess_gain(cAttr, instance[attr]) for
                               cAttr in concept.av_counts if
                               isPartialMatch(attr, cAttr, m, unnamed)]

            if len(best_attr_h) > 0:
                h -= max(best_attr_h)

    return h

def _flatMatchGoalTestFn(node):
    """
    Returns True if every component in the original instance has been renamed
    in the given node.

    See the :mod:`concept_formation.search` library for more details.
    """
    mapping, unnamed, availableNames = node.state
    return len(unnamed) == 0

def isPartialMatch(iAttr, cAttr, mapping, unnamed):
    """Returns True if the instance attribute (iAttr) partially matches the
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

    >>> isPartialMatch(('<', 'o2.a', 'o1.a'), ('<', 'c2.a', 'c1.b'), {'o1': 'c1'}, {'o2'})
    False

    >>> isPartialMatch(('<', 'o2.a', 'o1.a'), ('<', 'c2.a', 'c1.a'), {'o1': 'c1'}, {'o2'})
    True

    >>> isPartialMatch(('<', 'o2.a', 'o1.a'), ('<', 'c2.a', 'c1.a'), {'o1': 'c1', 'o2': 'c2'}, {})
    True
    """
    if type(iAttr) != type(cAttr):
        return False
    if isinstance(iAttr, tuple) and len(iAttr) != len(cAttr):
        return False

    if isinstance(iAttr, tuple):
        if iAttr[0] != cAttr[0]:
            return False
        for i,v in enumerate(iAttr):
            if i == 0:
                continue

            # TODO handle nested relations here
            # Chris MacLellan

            iSplit = v.split('.')
            cSplit = cAttr[i].split('.')
            if len(iSplit) != len(cSplit):
                return False
            for j,v2 in enumerate(iSplit):
                if v2 in mapping and mapping[v2] != cSplit[j]:
                    return False
                if v2 not in mapping and v2 not in unnamed and v2 != cSplit[j]:
                    return False
    elif "." not in cAttr:
        return False
    else:
        iSplit = iAttr.split('.')
        cSplit = cAttr.split('.')
        if len(iSplit) != len(cSplit):
            return False
        if iSplit[-1] != cSplit[-1]:
            return False
        for i,v in enumerate(iSplit[:-1]):
            if v in mapping and mapping[v] != cSplit[i]:
                return False

    return True

def extract_list_elements(instance):
    """
    Find all lists in an instance and extract their elements into their own
    subjects of the main instance.

    Unlike the utils.extract_components function this one will extract ALL
    elements into their own objects not just object literals

    >>> instance = {"a":"n","list1":["test",{"p":"q","j":"k"},{"n":"m"}]}
    >>> instance = extract_list_elements(instance)
    >>> doctest_print(instance)
    {'a': 'n',
    'list1': [
        'o1',
        'o2',
        'o3'
    ],
    'o1': {'val': 'test'},
    'o2': {'j': 'k',
        'p': 'q'},
    'o3': {'n': 'm'}}
    """

    new_instance = {}
    for a in instance.keys():
        if isinstance(instance[a],list):
            new_list = []
            for el in instance[a]:
                
                # TODO do we want to deep copy in the case we find a dict?
                if isinstance(el,dict):
                    new_obj = el
                else :
                    new_obj = {"val": el}

                new_att = gensym()
                new_instance[new_att] = extract_list_elements(new_obj)
                new_list.append(new_att)

            new_instance[a] = new_list

        elif isinstance(instance[a],dict):
            new_instance[a] = extract_list_elements(instance[a])
        else :
            new_instance[a] = instance[a]

    return new_instance

def lists_to_relations(instance):
    """
    Travese the instance and turn any list elements into 
    a series of relations.

    >>> instance = {"list1":['a','b','c']}
    >>> instance = lists_to_relations(instance)
    >>> doctest_print(instance)
    {(('ordered-list',), ('list1',), ('a', ''), ('b', '')): True,
    (('ordered-list',), ('list1',), ('b', ''), ('c', '')): True}
    
    >>> instance = {"list1":['a','b','c'],"list2":['w','x','y','z']}
    >>> instance = lists_to_relations(instance)
    >>> doctest_print(instance)
    {(('ordered-list',), ('list1',), ('a', ''), ('b', '')): True,
    (('ordered-list',), ('list1',), ('b', ''), ('c', '')): True,
    (('ordered-list',), ('list2',), ('w', ''), ('x', '')): True,
    (('ordered-list',), ('list2',), ('x', ''), ('y', '')): True,
    (('ordered-list',), ('list2',), ('y', ''), ('z', '')): True}

    >>> instance = {"stack":[{"a":1, "b":2, "c":3}, {"x":1, "y":2, "z":3}, {"i":1, "j":2, "k":3}]}
    >>> instance = extract_list_elements(instance)
    >>> doctest_print(instance)
    {'o4': {'a': 1,
        'b': 2,
        'c': 3},
    'o5': {'x': 1,
        'y': 2,
        'z': 3},
    'o6': {'i': 1,
        'j': 2,
        'k': 3},
    'stack': [
        'o4',
        'o5',
        'o6'
    ]}

    >>> instance = lists_to_relations(instance)
    >>> doctest_print(instance)
    {'o4': {'a': 1,
        'b': 2,
        'c': 3},
    'o5': {'x': 1,
        'y': 2,
        'z': 3},
    'o6': {'i': 1,
        'j': 2,
        'k': 3},
    (('ordered-list',), ('stack',), ('o4', ''), ('o5', '')): True,
    (('ordered-list',), ('stack',), ('o5', ''), ('o6', '')): True}
    """
    new_instance = {}
    for attr in instance.keys():
        if isinstance(instance[attr], list):
            for i in range(len(instance[attr])-1):
                
                rel = tuplize_relation_elements(
                    ("ordered-list",
                        attr,
                        str(instance[attr][i]),
                        str(instance[attr][i+1])),
                    {str(instance[attr][i]),
                        str(instance[attr][i+1])})

                new_instance[rel] = True

        elif isinstance(instance[attr],dict):
            new_instance[attr] = lists_to_relations(instance[attr])
        else:
            new_instance[attr] = instance[attr]
    
    return new_instance

def hoist_sub_objects(instance) :
    """
    Travese the instance for objects that contain subobjects and hoists the
    subobjects to be their own objects at the top level of the instance. 
    
    >>> instance = {"a1":"v1","sub1":{"a2":"v2","a3":3},"sub2":{"a4":"v4","subsub1":{"a5":"v5","a6":"v6"},"subsub2":{"subsubsub":{"a8":"V8"},"a7":7}}}
    >>> doctest_print(instance)
    {'a1': 'v1',
    'sub1': {'a2': 'v2',
        'a3': 3},
    'sub2': {'a4': 'v4',
        'subsub1': {'a5': 'v5',
            'a6': 'v6'},
        'subsub2': {'a7': 7,
            'subsubsub': {'a8': 'V8'}}}}
            
    >>> instance = hoist_sub_objects(instance)
    >>> doctest_print(instance)
    {'a1': 'v1',
    'sub1': {'a2': 'v2',
        'a3': 3},
    'sub2': {'a4': 'v4'},
    'subsub1': {'a5': 'v5',
        'a6': 'v6'},
    'subsub2': {'a7': 7},
    'subsubsub': {'a8': 'V8'},
    (('has-component',), ('sub2', ''), ('subsub1', '')): True,
    (('has-component',), ('sub2', ''), ('subsub2', '')): True,
    (('has-component',), ('subsub2', ''), ('subsubsub', '')): True}
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
            rel = tuplize_relation_elements(
                ("has-component",
                   str(attr),
                   str(a)),{str(attr),
                   str(a)})
            top_level[rel] = True
        else :
            new_sub[a] = sub[a]
    return new_sub


def pre_process(instance):
    """
    Runs all of the pre-processing functions

    >>> instance = {"noma":"a","num3":3,"compa":{"nomb":"b","num4":4,"sub":{"nomc":"c","num5":5}},"compb":{"nomd":"d","nome":"e"},"(related compa.num4 comb.nome)":True,"list1":["a","b",{"i":1,"j":12.3,"k":"test"}]}
    >>> doctest_print(instance)
    {'(related compa.num4 comb.nome)': True,
    'compa': {'nomb': 'b',
        'num4': 4,
        'sub': {'nomc': 'c',
            'num5': 5}},
    'compb': {'nomd': 'd',
        'nome': 'e'},
    'list1': [
        'a',
        'b',
        {'i': 1,
            'j': 12.3,
            'k': 'test'}
    ],
    'noma': 'a',
    'num3': 3}

    >>> instance = pre_process(instance)
    >>> doctest_print(instance)
    {'noma': 'a',
    'num3': 3,
    ('o10', 'val'): 'a',
    ('o11', 'val'): 'b',
    ('o12', 'i'): 1,
    ('o12', 'j'): 12.3,
    ('o12', 'k'): 'test',
    ('o7', 'nomb'): 'b',
    ('o7', 'num4'): 4,
    ('o8', 'nomc'): 'c',
    ('o8', 'num5'): 5,
    ('o9', 'nomd'): 'd',
    ('o9', 'nome'): 'e',
    (('has-component',), ('o7', ''), ('o8', '')): True,
    (('ordered-list',), ('list1',), ('o10', ''), ('o11', '')): True,
    (('ordered-list',), ('list1',), ('o11', ''), ('o12', '')): True,
    (('related',), ('o7', 'num4'), ('comb', 'nome')): True}

    >>> instance = pre_process(instance)
    >>> doctest_print(instance)
    {'noma': 'a',
    'num3': 3,
    ('o10', 'val'): 'a',
    ('o11', 'val'): 'b',
    ('o12', 'i'): 1,
    ('o12', 'j'): 12.3,
    ('o12', 'k'): 'test',
    ('o7', 'nomb'): 'b',
    ('o7', 'num4'): 4,
    ('o8', 'nomc'): 'c',
    ('o8', 'num5'): 5,
    ('o9', 'nomd'): 'd',
    ('o9', 'nome'): 'e',
    (('has-component',), ('o7', ''), ('o8', '')): True,
    (('ordered-list',), ('list1',), ('o10', ''), ('o11', '')): True,
    (('ordered-list',), ('list1',), ('o11', ''), ('o12', '')): True,
    (('related',), ('o7', 'num4'), ('comb', 'nome')): True}
    
    """
    instance = standardize_apart_names(instance)
    instance = extract_list_elements(instance)
    instance = lists_to_relations(instance)
    instance = hoist_sub_objects(instance)
    #instance = tuplize(instance) # flatten should just tuplize. 
    instance = flatten_json(instance)
    return instance

def doctest_print(instance, depth=0):
    """Take and instance pretty print it with a deterministic key ordering. 

    We can't use the default pprint operation because it doesn't
    deterministically order tuple keys in a dictionary.
    """
    def cmp(a, b):
        return (a > b) - (a < b) 

    def t_depth(tup,depth=0):
        if isinstance(tup[0],tuple):
            return t_depth(tup[0],depth+1)
        else:
            return (depth,str(tup))

    def compare(x,y):
        if isinstance(x,tuple) and isinstance(y,tuple):
            xd = t_depth(x)
            yd = t_depth(y)
            if xd[0] == yd[0]:
                return cmp(str(x),str(y))
            else:
                return cmp(xd[0],yd[0])
        elif isinstance(x,tuple):
            return 1
        elif isinstance(y,tuple):
            return -1
        else:
            return cmp(x,y)

    def cmp_to_key(mycmp):
        'Convert a cmp= function into a key= function'
        class K(object):
            def __init__(self, obj, *args):
                self.obj = obj
            def __lt__(self, other):
                return mycmp(self.obj, other.obj) < 0
            def __gt__(self, other):
                return mycmp(self.obj, other.obj) > 0
            def __eq__(self, other):
                return mycmp(self.obj, other.obj) == 0
            def __le__(self, other):
                return mycmp(self.obj, other.obj) <= 0
            def __ge__(self, other):
                return mycmp(self.obj, other.obj) >= 0
            def __ne__(self, other):
                return mycmp(self.obj, other.obj) != 0
        return K

    def str2(val):
        if isinstance(val,str):
            return "'"+val+"'"
        else:
            return str(val)

    str_to_print = '{'
    first = True

    for k in sorted(instance, key=cmp_to_key(compare)):
        if first:
            str_to_print += str2(k) + ': '
            first=False 
        else:
            str_to_print += (' '*4*depth) + str2(k) + ': '
        if isinstance(instance[k],dict):
            str_to_print += doctest_print(instance[k],depth+1)
        elif isinstance(instance[k],list):
            str_to_print += '[\n'
            for i in instance[k]:
                if isinstance(i,dict):
                    str_to_print += (' '*4*(depth+1))+doctest_print(i,depth+2)
                else:
                    str_to_print += (' '*4*(depth+1))+str2(i)
                str_to_print += ',\n'
            str_to_print = str_to_print[:-2] + '\n' +(' '*4*depth)+']'
        else:
            str_to_print += str2(instance[k])
        str_to_print +=',\n'
    str_to_print = str_to_print[:-2] + '}'
    if depth == 0:
        print(str_to_print)
    else:
        return str_to_print

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
    instance = renameFlat(instance, mapping)
    return instance
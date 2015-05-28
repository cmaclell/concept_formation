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

def standardizeApartNames(instance):
    """Given a :ref:`raw instance <raw-instance>` rename all the components so they
    have unique names.

    This will rename component attirbutes as well as any occurance of the
    component's name within relation attributes. This renaming is necessary to
    allow for a search between possible mappings without collisions.

    :param instance: An instance to be named apart.
    :type instance: :ref:`raw instance <raw-instance>`
    :return: an instance with component attributes renamed
    :rtype: :ref:`standardized instance <standard-instance>`

    >>> import pprint
    >>> instance = {'a': {}, 'r1': ['is-good', 'a']}
    >>> standard = standardizeApartNames(instance)
    >>> pprint.pprint(standard)
    {'o2': {}, 'r1': ['is-good', 'o2']}
    """
    new_instance = {}
    relations = []
    mapping = {}

    for attr in instance:
        if isinstance(instance[attr], list):
            relations.append((attr, instance[attr]))
        elif isinstance(instance[attr], dict):
            mapping[attr] = gensym()
            new_instance[mapping[attr]] = standardizeApartNames(instance[attr])
        else:
            new_instance[attr] = instance[attr]

    for name, r in relations:
        new_relation = []
        for i,v in enumerate(r):
            if i == 0:
                new_relation.append(v)
            else:
                new_relation.append(mapping[v])
        new_instance[name] = new_relation

    return new_instance

def getComponentNames(instance):
    """Given  a :ref:`flattened instance <flattened-instance>` or a concept's
    probability table return a list of all of the component names.


    :param instance: An instance or a concept's probability table.
    :type instance: :ref:`raw instance <raw-instance>` or dict
    :return: A list of all of the component names present in the instance
    :rtype: [str, str, ...]

    >>> instance = {'c1.a': 0, 'c2.a': 0, '_c3._a': 0}
    >>> names = getComponentNames(instance)
    >>> sorted(names)
    ['c1', 'c2', 'c3']
    """
    names = set()
    for attr in instance:
        if isinstance(attr, tuple):
            continue

        for name in attr.split(".")[:-1]:
            if name[0] == "_":
                names.add(name[1:])
            else:
                names.add(name)

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
            new_attr = []
            for name in val.split("."):
                new_attr.append(mapping[name])
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

    >>> import pprint
    >>> instance = {'c1.a': 1, ('good', 'c1'): True}
    >>> mapping = {'c1': 'o1'}
    >>> renamed = renameFlat(instance,mapping)
    >>> pprint.pprint(renamed)
    {'o1.a': 1, ('good', 'o1'): True}
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

def flattenJSON(instance):
    """Takes a :ref:`raw instance <raw-instance>`, standardizes apart the component
    names, and flattens it. 

    Hierarchy is represented with periods between variable names. This process
    also converts the relations into tuples with values of ``True``.

    :param instance: An instance to be flattened.
    :type instance: :ref:`raw instance <raw-instance>`
    :return: A copy of the instance flattend
    :rtype: :ref:`flattened instance <flattened-instance>`

    >>> import pprint
    >>> instance = {'a': 1, 'c1': {'b': 1}}
    >>> flat = flattenJSON(instance)
    >>> pprint.pprint(flat)
    {'a': 1, 'o1.b': 1}
    """
    instance = standardizeApartNames(instance)
    temp = {}
    for attr in instance:
        if isinstance(instance[attr], dict):
            subobject = flattenJSON(instance[attr])
            for so_attr in subobject:
                if isinstance(so_attr, tuple):
                    relation = []
                    for idx, val in enumerate(so_attr):
                        if idx == 0:
                            relation.append(val)
                        else:
                            relation.append(attr + "." + val)
                    temp[tuple(relation)] = True
                elif so_attr[0] == "_":
                    temp["_" + attr + "." + so_attr] = subobject[so_attr]
                else:
                    temp[attr + "." + so_attr] = subobject[so_attr]
        else:
            if isinstance(instance[attr], list):
                temp[tuple(instance[attr])] = True
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

def bindFlatAttr(attr, mapping):
    """Renames an attribute given a mapping.

    :param attr: The attribute to be renamed
    :type attr: str
    :param mapping: A dictionary of mappings between component names
    :type mapping: dict
    :return: The attribute's new name or ``None`` if the mapping is incomplete
    :rtype: str

    >>> attr = ('before', 'c1', 'c2')
    >>> mapping = {'c1': 'o1', 'c2':'o2'}
    >>> bindFlatAttr(attr, mapping)
    ('before', 'o1', 'o2')

    If the mapping is incomplete then returns ``None`` (nothing) 

    >>> attr = ('before', 'c1', 'c2')
    >>> mapping = {'c1': 'o1'}
    >>> bindFlatAttr(attr, mapping) is None
    True
    """
    if isinstance(attr, tuple):
        for i,v in enumerate(attr):
            if i == 0:
                continue
            for o in v.split('.'):
                if o not in mapping:
                    return None
        return renameRelation(attr, mapping)
    elif '.' in attr:
        path = attr.split('.')[:-1]
        for o in path:
            if o not in mapping:
                return None
        return renameComponent(attr, mapping)
    else:
        return attr

def containsComponent(component, attr):
    """Return ``True`` if the given component name is in the attribute, either as part of a
    hierarchical name or within a relations otherwise ``False``.

    :param component: A component name
    :type component: str
    :param attr: An attribute name
    :type atte: str
    :return: ``True`` if the component name exists inside the attribute name ``False`` otherwise
    :rtype: bool

    >>> containsComponent('c1', 'c2.c1.a')
    True

    >>> containsComponent('c3', ('before', 'c1', 'c2'))
    False
    """
    if isinstance(attr, tuple):
        for i,v in enumerate(attr):
            if i == 0:
                continue
            for o in v.split('.'):
                if o == component:
                    return True
    elif "." in attr:
        for o in attr.split('.')[:-1]:
            if o == component:
                return True

    return False

def flatMatch(concept, instance, optimal=False):
    """Given a concept and instance this function returns a mapping  that can be
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
    inames = frozenset(getComponentNames(instance))
    cnames = frozenset(getComponentNames(concept.av_counts))

    if(len(inames) == 0 or
       len(cnames) == 0):
        return {}
     
    initial = search.Node((frozenset(), inames, cnames), extra=(concept, instance))
    if optimal:
        solution = next(search.BestFGS(initial, _flatMatchSuccessorFn, _flatMatchGoalTestFn,
                                _flatMatchHeuristicFn))
    else:
        solution = next(search.BeamGS(initial, _flatMatchSuccessorFn, _flatMatchGoalTestFn,
                           _flatMatchHeuristicFn, initialBeamWidth=3))
    #print(solution.cost)

    if solution:
        mapping, unnamed, availableNames = solution.state
        return {a:v for a,v in mapping}
    else:
        return None

def _flatMatchSuccessorFn(node):
    """Given a node (mapping, instance, concept), this function computes the
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
            if not containsComponent(n, attr):
                continue
            new_attr = bindFlatAttr(attr, m)
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
                new_attr = bindFlatAttr(attr, m)
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
        new_attr = bindFlatAttr(attr, m)
        if not new_attr:
            best_attr_h = [concept.attr_val_guess_gain(cAttr, instance[attr]) for
                               cAttr in concept.av_counts if
                               isPartialMatch(attr, cAttr, m)]

            if len(best_attr_h) != 0:
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

def isPartialMatch(iAttr, cAttr, mapping):
    """Returns True if the instance attribute (iAttr) partially matches the
    concept attribute (cAttr) given the mapping.

    :param iAttr: An attribute in an instance
    :type iAttr: str
    :param cAttr: An attribute in a concept
    :type cAttr: str
    :param mapping: A mapping between between attribute names
    :type mapping: dict
    :return: ``True`` if the instance attribute matches the concept attribute in the mapping otherwise ``False``
    :rtype: bool
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
            iSplit = v.split('.')
            cSplit = cAttr[i].split('.')
            if len(iSplit) != len(cSplit):
                return False
            for j,v2 in enumerate(iSplit):
                if v2 in mapping and mapping[v2] != cSplit[j]:
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

def structure_map(concept, instance):
    """Flatten the instance, perform structure mapping to the concept, rename the instance
    based on this structure mapping, and return the renamed instance.

    :param concept: A concept to structure map the instance to
    :type concept: TrestleNode
    :param instance: An instance to map to the concept
    :type instance: :ref:`raw instance <raw-instance>`
    :return: A fully mapped and flattend copy of the instance
    :rtype: :ref:`mapped instance <fully-mapped>`

    """
    temp_instance = flattenJSON(instance)
    mapping = flatMatch(concept, temp_instance)
    temp_instance = renameFlat(temp_instance, mapping)
    return temp_instance

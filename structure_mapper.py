import search

gensym_counter = 0;

def gensym():
    """
    Generates unique names for naming renaming apart objects.
    """
    global gensym_counter
    gensym_counter += 1
    return 'o' + str(gensym_counter)

def standardizeApartNames(instance):
    """
    Given an raw input instance (i.e., relations are still lists, object is
    still structured), it renames all the components so they have unique names.

    >>> import pprint
    >>> pprint.pprint(standardizeApartNames({'a': {}, 'r1': ['is-good', 'a']}))
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
    """
    Given a flattened representation of an instance or concept.av_counts
    return a list of all of the component names.

    >>> sorted(getComponentNames({'c1.a': 0, 'c2.a': 0, '_c3._a': 0}))
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
    """
    Takes a component attribute (e.g., o1.o2) and renames the 
    components given a mapping.

    >>> renameComponent("c1.c2.a", {'c1': 'o1', 'c2': 'o2'})
    'o1.o2.a'

    >>> renameComponent("_c1._c2._a", {'c1': 'o1', 'c2': 'o2'})
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
    """
    Takes a relational attribute (e.g., (before o1 o2)) and renames
    the components based on mapping.

    >>> renameRelation(('before', 'c1', 'c2'),  {'c1': 'o1', 'c2': 'o2'})
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
    """
    Given a flattened instance and a mapping (type = dict) rename the
    components and relations and return the renamed instance. 

    >>> import pprint
    >>> pprint.pprint(renameFlat({'c1.a': 1, ('good', 'c1'): True}, {'c1': 'o1'}))
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
    """
    Takes a raw hierarchical JSON instance, standardizes apart the component
    names, and flattens it. It represents hierarchy with periods between variable
    names. It also converts the relations into tuples with values.

    >>> import pprint
    >>> pprint.pprint(flattenJSON({'a': 1, 'c1': {'b': 1}}))
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

def traverseStructure(path, instance):
    """
    Given an instance (hashmap) to the given subobject for the given path.
    Creates subobjects if they do not exist.

    >>> x = {}
    >>> traverseStructure(['c1', 'c2'], x)
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
    """
    Takes a flattened instance and adds the structure back in. This essentially
    "undoes" the flattening process. 

    >>> structurizeJSON({'c1.c2.a': 1})
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
            obj = traverseStructure(path, temp)
            obj[tuple(relation)] = True

        elif "." in attr:
            path = [p[1:] if p[0] == "_" else p for p in attr.split('.')]
            subatt = path[-1]
            path = path[:-1]
            curr = traverseStructure(path, temp)
            if attr[0] == "_":
                curr["_" + subatt] = instance[attr]
            else:
                curr[subatt] = instance[attr]

        else:
            temp[attr] = instance[attr]

    return temp

def bindFlatAttr(attr, mapping):
    """
    Renames the attribute given the mapping.

    >>> bindFlatAttr(('before', 'c1', 'c2'), {'c1': 'o1', 'c2':'o2'})
    ('before', 'o1', 'o2')

    If the mapping is incomplete then returns None (nothing) 
    >>> bindFlatAttr(('before', 'c1', 'c2'), {'c1': 'o1'}) is None
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
    """
    Check if the given component is in the attribute.
    
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

def flatMatch(concept, instance, optimal=True):
    """
    Given a concept and instance this function returns a mapping (dictionary)
    that can be used to rename the instance. The mapping returned maximizes
    similarity between the instance and the concept.
    """
    inames = frozenset(getComponentNames(instance))
    cnames = frozenset(getComponentNames(concept.av_counts))

    if(len(inames) == 0 or
       len(cnames) == 0):
        return {}
     
    initial = search.Node((frozenset(), inames, cnames), extra=(concept, instance))
    if optimal:
        solution = next(search.BestFGS(initial, flatMatchSuccessorFn, flatMatchGoalTestFn,
                                flatMatchHeuristicFn))
    else:
        solution = next(search.BeamGS(initial, flatMatchSuccessorFn, flatMatchGoalTestFn,
                           flatMatchHeuristicFn), initialBeamWidth=3)
    #print(solution.cost)

    if solution:
        mapping, unnamed, availableNames = solution.state
        return {a:v for a,v in mapping}
    else:
        return None

def flatMatchSuccessorFn(node):
    """
    Given a node (mapping, instance, concept), this function computes the
    successor nodes where an additional mapping has been added for each
    possible additional mapping. See the search library for more details.
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

def flatMatchHeuristicFn(node):
    """
    Considers all partial matches for each unbound attribute and assumes that
    you get the highest guess_gain match. This provides an over estimation of
    the possible reward (i.e., is admissible).
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

def flatMatchGoalTestFn(node):
    """
    Returns True if every component in the original instance has been renamed
    in the given node.
    """
    mapping, unnamed, availableNames = node.state
    return len(unnamed) == 0

def isPartialMatch(iAttr, cAttr, mapping):
    """
    Returns True if the instance attribute (iAttr) partially matches the
    concept attribute (cAttr) given the mapping.
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
    """
    Flatten the instance, perform structure mapping, rename the instance
    based on this structure mapping, and return the renamed instance.
    """
    temp_instance = flattenJSON(instance)
    mapping = flatMatch(concept, temp_instance)
    temp_instance = renameFlat(temp_instance, mapping)
    return temp_instance

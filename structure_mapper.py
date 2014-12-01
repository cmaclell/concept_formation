import re
from search import BeamGS, BestFGS, Node
from cobweb import CobwebTree

gensym_counter = 0;

def gensym():
    global gensym_counter
    gensym_counter += 1
    return 'o' + str(gensym_counter)

def standardizeApartNames(instance):
    """
    Given an raw input instance (relations are still lists), it renames all the
    components so they have unique names.
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
    Given a flat representation of an instance or concept.av_counts
    return a list of all of the component names.
    """
    names = set()
    for attr in instance:
        if isinstance(attr, tuple):
            continue
        attr = re.sub("_", "", attr)
        for name in attr.split(".")[:-1]:
            names.add(name)
    return list(names)

def renameComponent(attr, mapping):
    """
    Takes a component attribute (e.g., o1.o2) and renames the 
    components.
    """
    ignore = False
    if attr[0] == "_":
        ignore = True
    pattr = re.sub("_", "", attr)
    new_attr = []
    for name in pattr.split('.')[:-1]:
        new_attr.append(mapping[name])
    new_attr.append(pattr.split('.')[-1])
    if ignore:
        return "_" + "._".join(new_attr)
    else:
        return ".".join(new_attr)

def renameRelation(attr, mapping):
    """
    Takes a relational attribute (e.g., (before o1 o2)) and renames
    the components based on mapping.
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

def rename(instance, mapping):
    """
    Given a flattened instance and a mapping (type = dict) rename the
    components and relations and return the renamed instance.
    """
    # Ensure it is a complete mapping
    # Might be troublesome if there is a name collision
    for attr in instance:
        if isinstance(attr, tuple):
            continue
        attr = re.sub("_", "", attr)
        for name in attr.split('.')[:-1]:
            if name not in mapping:
                mapping[name] = name

    temp_instance = {}

    # rename all attribute values
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
    Takes a hierarchical instance and flattens it. It represents
    hierarchy with periods in variable names. It also converts the 
    relations into tuples with values.
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
                    # propagate ignore up.
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
    """
    temp = {}
    for attr in instance:
        if isinstance(attr, tuple):
            #attr.split('.')[:-1]
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
            # handle ignore
            ignore = ""
            if attr[0] == "_":
                ignore = "_"

            pattr = re.sub("_", "", attr)
            path = pattr.split('.')
            subatt = path[-1]
            path = path[:-1]
            curr = traverseStructure(path, temp)
            curr[ignore + subatt] = instance[attr]

        else:
            temp[attr] = instance[attr]

    return temp

def bindAttr(attr, mapping):
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

def flatMatch(concept, instance):
    inames = frozenset(getComponentNames(instance))
    cnames = frozenset(getComponentNames(concept.av_counts))

    if(len(inames) == 0 or
       len(cnames) == 0):
        return {}
     
    initial = Node((frozenset(), inames, cnames), extra=(concept, instance))
    solution = next(BeamGS(initial, flatMatchSuccessorFn, flatMatchGoalTestFn,
                       flatMatchHeuristicFn), 10)
    #solution = next(BestFGS(initial, flatMatchSuccessorFn, flatMatchGoalTestFn,
    #                        flatMatchHeuristicFn))
    #print(solution.cost)

    if solution:
        mapping, unnamed, availableNames = solution.state
        return {a:v for a,v in mapping}
    else:
        return None

def flatMatchSuccessorFn(node):
    mapping, inames, availableNames = node.state
    concept, instance = node.extra

    for n in inames:
        reward = 0
        m = {a:v for a,v in mapping}
        m[n] = n
        for attr in instance:
            new_attr = bindAttr(attr, m)
            if new_attr:
                reward -= concept.attr_val_guess_gain(new_attr, instance[attr])

        yield Node((mapping.union(frozenset([(n, n)])), inames -
                    frozenset([n]), availableNames), node, n + ":" + n,
                   node.cost + reward, node.depth + 1, node.extra)

        for new in availableNames:
            reward = 0
            m = {a:v for a,v in mapping}
            m[n] = new
            for attr in instance:
                new_attr = bindAttr(attr, m)
                if new_attr:
                    reward -= concept.attr_val_guess_gain(new_attr,
                                                          instance[attr])
            yield Node((mapping.union(frozenset([(n, new)])), inames -
                                      frozenset([n]), availableNames -
                                      frozenset([new])), node, n + ":" + new,
                        node.cost + reward, node.depth + 1, node.extra)

def flatMatchHeuristicFn(node):
    mapping, unnamed, availableNames = node.state
    concept, instance = node.extra

    h = 0
    m = {a:v for a,v in mapping}
    for attr in instance:
        new_attr = bindAttr(attr, m)
        if not new_attr:
            h -= 1

    return h

def flatMatchGoalTestFn(node):
    mapping, unnamed, availableNames = node.state
    return len(unnamed) == 0

if __name__ == "__main__":

    o = {'ob1': {'x':1, 'y': 1}, 
         'ob2': {'ob3': {'_ignore': 'a', 'inner':1}, 'ob4':{'inner':2},
         "r1": ["before", "ob3", "ob4"]}}

    fo = flattenJSON(o)
    print(fo)
    #so = structurizeJSON(fo)
    #print(so)

    tree = CobwebTree()
    tree.ifit(fo)

    o2 = {'ob0': {'x':1, 'y': 1}, 'ob01': {'ob02': {'inner':1},
                                           'ob03':{'inner':2}}}
    fo2 = flattenJSON(o2)
    print(fo2)
    print(tree)
    sol = flatMatch(tree.root, fo2)
    print(sol)



from search import DepthFGS

def standardizeApartNames(instance):
    pass

def flattenJSON(instance):
    """
    Takes a hierarchical instance and flattens it. It represents
    hierarchy with periods in variable names. It also converts the 
    relations into tuples with values.
    """
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
                    # propagate attribute ignore up.
                    temp["_" + attr + "." + so_attr] = subobject[so_attr]
                else:
                    temp[attr + "." + so_attr] = subobject[so_attr]
        else:
            if isinstance(instance[attr], list):
                temp[tuple(instance[attr])] = True
            else:
                temp[attr] = instance[attr]
    return temp

def structurizeJSON(data):
    """
    Takes a flattened instance and adds the structure back in. This essentially
    "undoes" the flattening process.
    """
    pass

def mapStructure(concept, instance):
    pass

if __name__ == "__main__":

    o = {'o1': {'x':1, 'y': 1}, 
         'o2': {'o3': {'_ignore': 'a', 'inner':1}, 'o4':{'inner':2},
         "r1": ["before", "o3", "o4"]}}

    print(flattenJSON(o))

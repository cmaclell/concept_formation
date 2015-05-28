from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from random import uniform
from numbers import Number
from math import sqrt


# A hashtable of values to use in the c4(n) function to apply corrections to
# estimates of std.
c4n_table = {2: 0.7978845608028654, 
      3:  0.886226925452758, 
      4:  0.9213177319235613, 
      5:  0.9399856029866254, 
      6:  0.9515328619481445, 
      7:  0.9593687886998328, 
      8:  0.9650304561473722, 
      9:  0.9693106997139539, 
      10: 0.9726592741215884, 
      11: 0.9753500771452293, 
      12: 0.9775593518547722, 
      13: 0.9794056043142177, 
      14: 0.9809714367555161, 
      15: 0.9823161771626504, 
      16: 0.9834835316158412, 
      17: 0.9845064054718315, 
      18: 0.985410043808079, 
      19: 0.9862141368601935, 
      20: 0.9869342675246552, 
      21: 0.9875829288261562, 
      22: 0.9881702533158311, 
      23: 0.988704545233999, 
      24: 0.9891926749585048, 
      25: 0.9896403755857028, 
      26: 0.9900524688409107, 
      27: 0.990433039209448, 
      28: 0.9907855696217323, 
      29: 0.9911130482419843}

def c4(n) :
    """
    Returns the correction factor to apply to unbias estimates of standard 
    deviation in low sample sizes. This implementation is based on a lookup 
    table for n in [2-29] and returns 1.0 for values >= 30.

    >>> c4(3)
    0.886226925452758
    """
    if n <= 1 :
        raise ValueError("Cannot apply correction for a sample size of 1.")
    else :
        return c4n_table[n] if n < 30 else 1.0

def mean(values):
    """Computes the mean of a list of values.

    This is primarily included to reduce dependency on external math libraries
    like numpy in the core algorithm.

    :param values: a list of numbers
    :type values: list
    :return: the mean of the list of values
    :rtype: float

    >>> mean([600, 470, 170, 430, 300])
    394.0
    """
    if len(values) <= 0:
        raise ValueError("Length of list must be greater than 0.")

    return float(sum(values))/len(values)

def std(values):
    """
    Computes the standard deviation of a list of values.

    This is primarily included to reduce dependency on external math libraries
    like numpy in the core algorithm.

    :param values: a list of numbers
    :type values: list
    :return: the standard deviation of the list of values
    :rtype: float

    >>> std([600, 470, 170, 430, 300])
    147.32277488562318
    """
    if len(values) <= 0:
        raise ValueError("Length of list must be greater than 0.")

    meanValue = mean(values)
    variance =  float(sum([(v - meanValue) * (v - meanValue) for v in
                           values]))/len(values)
    return sqrt(variance)

def lists_to_relations(instance, rel_name="Ordered", append_attr=True, attrs = None):
    """Take a raw structured instance containing lists and returns the same
    instance with all list elements converted to a series of binary ordering
    relations that describe the ordering of elements. This process will be
    applied recurrsively to any commponent attributes of the instance.

    :param rel_name: The name that should be used to describe the relation.
    :type rel_name: str
    :param append_attr: If ``True`` appends the original list's attribute name to the
        rel_name this is to prevent the structure mapper from over generalizing
        "Ordered" relations when there are multiple lists in a component.
    :type append_attr: bool
    :param attrs: a list of attribute names to specifically target for
        conversion. if ``None``, then the process will convert all applicable
        attributes.
    :type attrs: list
    :return: The original instance with lists converted to ordering relations
    :rtype: instance

    >>> import pprint
    >>> instance = {"list1":['a','b','c']}
    >>> instance = lists_to_relations(instance)
    >>> pprint.pprint(instance)
    {'list1-0': ['Orderedlist1', 'a', 'b'], 'list1-1': ['Orderedlist1', 'b', 'c']}

    >>> import pprint
    >>> instance = {"list1":['a','b','c']}
    >>> instance = lists_to_relations(instance, append_attr=False)
    >>> pprint.pprint(instance)
    {'list1-0': ['Ordered', 'a', 'b'], 'list1-1': ['Ordered', 'b', 'c']}
    """
    if attrs is None:
        attrs = [k for k in instance.keys()]

    if isinstance(attrs,str):
        attrs = [attrs]

    newInstance = {}
    for attr in instance:
        if isinstance(instance[attr], list) and attr in attrs:
            for i in range(len(instance[attr])-1):
                if append_attr:
                    newInstance[str(attr)+"-"+str(i)] = [str(rel_name+attr),
                        instance[attr][i],
                        instance[attr][i+1]]
                else:
                    newInstance[str(attr)+"-"+str(i)] = [str(rel_name),
                        instance[attr][i],
                        instance[attr][i+1]]
        elif isinstance(instance[attr],dict):
            newInstance[attr] = lists_to_relations(instance[attr],rel_name,append_attr,attrs)
        else:
            newInstance[attr] = instance[attr]
    return newInstance

def batch_lists_to_relations(instances, rel_name="Ordered", append_attr=True, attrs=None):
    """Perform :meth:`lists_to_relations` over a list of instances.
    """
    return [lists_to_relations(instance, rel_name, append_attr) for instance
            in instances]

def extract_components(instance, prefix="exob", attrs=None):
    """
    Extracts object literals from within any list elements and extracts them into
    their own component attributes. Note that this function does not add ordering
    relations see: :meth:`lists_to_relations`.

    :param prefix: Any objects found within list in the ojbect will be assigned
        new names of ``prefix + counter``. This value will eventually
        be aliased by TRESTLE's structure mapper but it should not collide with
        any object names that naturally exist wihtin the instance object.
    :type prefix: str
    :param attrs: a list of attribute names to specifically target for
        conversion. if ``None``, then the process will convert all applicable
        attributes.
    :type attrs: list
    :return: The original instance with in-line objects extracted to their own components
    :rtype: instance

    >>> import pprint
    >>> instance = {"a":"n","list1":["test",{"p":"q","j":"k"},{"n":"m"}]}
    >>> instance = extract_components(instance)
    >>> pprint.pprint(instance)
    {'a': 'n',
     'exob1': {'j': 'k', 'p': 'q'},
     'exob2': {'n': 'm'},
     'list1': ['test', 'exob1', 'exob2']}

    >>> import pprint
    >>> instance = {"a":"n","list1":["test",{"p":"q","j":"k"},{"n":"m"}]}
    >>> instance = extract_components(instance,prefix="newobject")
    >>> pprint.pprint(instance)
    {'a': 'n',
     'list1': ['test', 'newobject1', 'newobject2'],
     'newobject1': {'j': 'k', 'p': 'q'},
     'newobject2': {'n': 'm'}}
 
    """
    counter = 0

    if attrs is None:
        attrs = [k for k in instance.keys()]

    if isinstance(attrs,str):
        attrs = [attrs]

    for a in attrs:
        if  isinstance(instance[a],list):
            for i in range(len(instance[a])):
                if isinstance(instance[a][i],dict):
                    counter += 1
                    newName = prefix + str(counter)
                    instance[newName] = extract_components(instance[a][i])
                    instance[a][i] = newName
        if isinstance(instance[a],dict):
            instance[a] = extract_components(instance[a])
    return instance

def batch_extract_components(instances, prefix="exob", attrs=None):
    """Perform :meth:`extract_components` over a list of instances.
    """
    return [extract_components(instance, prefix, attrs) for instance
            in instances]

def numeric_to_nominal(instance, attrs = None):     
    """Takes an instance and takes any attributes that Trestle or Cobweb/3 would
    consider to be numeric attributes and converts them to nominal attributes.

    This is useful for when data natually contains values that would be numbers
    but you do not want to entertain that they have a distribution.

    :param attrs: a list of attribute names to specifically target for
        conversion. if ``None``, then the process will convert all applicable
        attributes.
    :type attrs: list
    :return: The original instance with numerical attributes converted to nominal
    :rtype: instance

    >>> import pprint
    >>> instance = {"x":12.5,"y":9,"z":"top"}
    >>> instance = numeric_to_nominal(instance)
    >>> pprint.pprint(instance)
    {'x': '12.5', 'y': '9', 'z': 'top'}

    >>> import pprint
    >>> instance = {"x":12.5,"y":9,"z":"12.6"}
    >>> instance = numeric_to_nominal(instance,attrs=["y","z"])
    >>> pprint.pprint(instance)
    {'x': 12.5, 'y': '9', 'z': '12.6'}

    >>> import pprint
    >>> instance = {"x":12.5,"y":9,"z":"top"}
    >>> instance = numeric_to_nominal(instance,attrs="y")
    >>> pprint.pprint(instance)
    {'x': 12.5, 'y': '9', 'z': 'top'}

    """
    if attrs is None:
        attrs = [k for k in instance.keys()]
    if isinstance(attrs,str):
        attrs = [attrs]
    for a in attrs:
        if isinstance(instance[a],Number):
            instance[a] = str(instance[a])
        if isinstance(instance[a],dict):
            instance[a] = numeric_to_nominal(instance[a],attrs)
    return instance

def batch_numeric_to_nominal(instances, attrs = None):
    """Perform :meth:`numeric_to_nominal` over a list of instances.
    """
    return [numeric_to_nominal(instance,attrs) for instance in instances]

def santize_JSON(instance,ext_ob_name="extOb",rel_name="Ordered",append_attr=True,attrs = None):
    """Takes a raw JSON object and applies some transformations to it that convert
    common structures into an equivalent format that Trestle would expect.

    Trestle can process any arbitrarily structured JSON object but this
    functions makes some conversions that preserve the intent of certain JSON
    notation in a way that Trestle would expect. In particular this process will
    extract an object litterals that are in-line within a list, and break any
    lists into a series of relations that describe their order.

    :param ext_ob_name: the prefix to use for objects extracted from in-line lists
    :type ext_ob_name: str
    :param relation_name: the name to use for ordering relations converted from lists
    :type relation_name: str
    :param append_attr: a flag for whether to append the original attribute name to a converted relation list
    :type append_attr: bool
    :param attrs: a list of attribute names to specifically target for
        conversion. if ``None``, then the process will convert all applicable
        attributes.
    :type attrs: list

    >>> import pprint
    >>> instance = {"stack":[{"a":1, "b":2, "c":3}, {"x":1, "y":2, "z":3}, {"i":1, "j":2, "k":3}]}
    >>> instance = santize_JSON(instance)
    >>> pprint.pprint(instance)
    {'extOb1': {'a': 1, 'b': 2, 'c': 3},
     'extOb2': {'x': 1, 'y': 2, 'z': 3},
     'extOb3': {'i': 1, 'j': 2, 'k': 3},
     'stack-0': ['Orderedstack', 'extOb1', 'extOb2'],
     'stack-1': ['Orderedstack', 'extOb2', 'extOb3']}

    """    
    return lists_to_relations(extract_components(instance,prefix=ext_ob_name,attrs=attrs),rel_name=rel_name,append_attr=append_attr,attrs=attrs)

def batch_santize_JSON(instances,ext_ob_name="extOb",relation_name="Ordered",append_attr=True,attrs = None):
    """Perform :meth:`santize_JSON` over a list of instances.
    """
    
    return [santize_JSON(instance,ext_ob_name,relation_name,append_attr,attrs) for instance in instances]


def weighted_choice(choices):
    """Given a list of tuples [(val, prob),...(val, prob)], return a
    randomly chosen value where the choice is weighted by prob.

    :param choices: A list of tuples
    :type choices: [(val, prob),...(val, prob)]
    :return: A choice sampled from the list according to the weightings
    :rtype: val

    .. seealso:: :meth:`CobwebNode.sample <concept_formation.cobweb.CobwebNode.sample>`
    """
    total = sum(w for c, w in choices)
    r = uniform(0, total)
    upto = 0
    for c, w in choices:
       if upto + w > r:
          return c
       upto += w
    assert False, "Shouldn't get here"



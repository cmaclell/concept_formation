from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
from random import uniform
from math import sqrt
from numbers import Number

import numpy as np
from scipy import linalg
from scipy.stats import sem
from scipy.stats import t

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
    """
    Computes the mean of a list of values.

    >>> mean([600, 470, 170, 430, 300])
    394.0
    """
    if len(values) <= 0:
        raise ValueError("Length of list must be greater than 0.")

    return float(sum(values))/len(values)

def std(values):
    """
    Computes the standard deviation of a list of values.

    >>> std([600, 470, 170, 430, 300])
    147.32277488562318
    """
    if len(values) <= 0:
        raise ValueError("Length of list must be greater than 0.")

    meanValue = mean(values)
    variance =  float(sum([(v - meanValue) * (v - meanValue) for v in
                           values]))/len(values)
    return sqrt(variance)

def listsToRelations(instance, relationName="Ordered", appendAttr=True, attrs = None):
    """
    Takes a structured instance containing lists and returns the same instance
    with all list elements converted to a series of relations that describe the
    ordering of elements. This process will be applied recurrsively to any
    commponent attributes of the instance.

    Arguments: 

        relationName -- The name that should be used to describe the relation,
        by default "Orderd" is used. However, a new name can be provided if
        that conflicts with other relations in the data already.

        appendAttr -- if True appends the originzal list's attribute name to the
        relationName this is to prevent the matcher from over generalizing
        ordering when there are multiple lists in an object.

        attrs -- A list of specific attribute names to convert. If no list is
        provided then all list attributes will be converted

    >>> import pprint
    >>> pprint.pprint(listsToRelations({"list1":['a','b','c']}))
    {'list1-0': ['Orderedlist1', 'a', 'b'], 'list1-1': ['Orderedlist1', 'b', 'c']}

    >>> import pprint
    >>> pprint.pprint(listsToRelations({"list1":['a','b','c']}, appendAttr=False))
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
                if appendAttr:
                    newInstance[str(attr)+"-"+str(i)] = [str(relationName+attr),
                        instance[attr][i],
                        instance[attr][i+1]]
                else:
                    newInstance[str(attr)+"-"+str(i)] = [str(relationName),
                        instance[attr][i],
                        instance[attr][i+1]]
        elif isinstance(instance[attr],dict):
            newInstance[attr] = listsToRelations(instance[attr],relationName,appendAttr,attrs)
        else:
            newInstance[attr] = instance[attr]
    return newInstance

def batchListsToRelations(instances, relationName="Ordered", appendAttr=True, attrs=None):
    """
    Takes a list of structured instances that contain lists and batch converts
    all the list elements to a relation format expected by TRESTLE.
    """
    return [listsToRelations(instance, relationName, appendAttr) for instance
            in instances]

def extractComponentsFromList(instance, prefix="exob", attrs=None):
    """

    Extracts object literals from within an list elements and extracts them into
    their own component attributes. Note that this function does not ad ordering
    relations so if that behavior is desired follow it up with listToRelation.

    Arguments:

        prefix -- Any objects found within list in the ojbect will be assigned
        new names of prefix + a sequential counter. This value will eventually
        be aliased by TRESTLE's structure mapper but it should not collide with
        any object names that naturally exist wihtin the instance object.

        attrs -- A list of attribute names to specifically target for object
        extraction. If this is not provided then the process will search within
        all of the instance's attribtues.

    >>> import pprint
    >>> pprint.pprint(extractComponentsFromList({"a":"n","l1":["test",{"p":"q","j":"k"},{"n":"m"}]}))
    {'a': 'n',
     'exob1': {'j': 'k', 'p': 'q'},
     'exob2': {'n': 'm'},
     'l1': ['test', 'exob1', 'exob2']}

    >>> import pprint
    >>> pprint.pprint(extractComponentsFromList({"a":"n","l1":["test",{"p":"q","j":"k"},{"n":"m"}]},prefix="newobject"))
    {'a': 'n',
     'l1': ['test', 'newobject1', 'newobject2'],
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
                    instance[newName] = extractComponentsFromList(instance[a][i])
                    instance[a][i] = newName
        if isinstance(instance[a],dict):
            instance[a] = extractComponentsFromList(instance[a])
    return instance

def batchExtractComponentsFromList(instances, prefix="exob", attrs=None):
    """

    Takes a list of instances and extracts components from lists within each of
    them.

    """
    return [extractComponentsFromList(instance, prefix, attrs) for instance
            in instances]

def numericToNominal(instance, attrs = None):     
    """         
    Takes a list of instances and converts any attributes that TRESTLE or
    COBWEB/3 would consider to be numeric attributes to nominal attributes in
    the case that should be treated as such. This is useful for when data
    natually contains values that would be numbers but you do not want to
    entertain that they have a distribution.    

    Arguments:

        attrs -- a list of attribute names to specifically target for
        conversion. If no list is provided it will default to all of the
        instance's attributes.

    >>> import pprint
    >>> pprint.pprint(numericToNominal({"x":12.5,"y":9,"z":"top"}))
    {'x': '12.5', 'y': '9', 'z': 'top'}

    >>> import pprint
    >>> pprint.pprint(numericToNominal({"x":12.5,"y":9,"z":"12.6"},attrs=["y","z"]))
    {'x': 12.5, 'y': '9', 'z': '12.6'}

    >>> import pprint
    >>> pprint.pprint(numericToNominal({"x":12.5,"y":9,"z":"top"},attrs="y"))
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
            instance[a] = numericToNominal(instance[a],attrs)
    return instance

def batchNumericToNominal(instances, attrs = None):
    """
    Takes a list of instances and batch converts any specified numeric
    attributes within instances to nominal attributes for times when this is the
    desired behavior
    """
    return [numericToNominal(instance,attrs) for instance in instances]

def santizeJSON(instance,ext_ob_name="extOb",relation_name="Ordered",appendAttr=True,attrs = None):
    """

    Takes a raw JSON object and applies some transformations to it that convert
    common structures into an equivalent format that TRESTLE would expect.
    TRESTLE should run given any arbitrarily structured JSON object but this
    process makes some conversions that preserve the intent of some JSON
    notation in a way that TRESTLE would expect. In particular this process will
    extract an object litterals that are in-line within a list, and break any
    lists into a series of tuples that describe their order.

    Attributes:

        ext_ob_name -- the prefix to use for objects extracted from in-line lists

        relation_name -- the name to use for ordering relations converted from lists

        appendAttr -- a flag for whether to append the original attribute name to a converted relation list

        attrs -- a list of particular attributes to perform conversion on.

    >>> import pprint
    >>> pprint.pprint(santizeJSON({"stack":[{"a":1,"b":2,"c":3},{"x":1,"y":2,"z":3},{"i":1,"j":2,"k":3}]}))
    {'extOb1': {'a': 1, 'b': 2, 'c': 3},
     'extOb2': {'x': 1, 'y': 2, 'z': 3},
     'extOb3': {'i': 1, 'j': 2, 'k': 3},
     'stack-0': ['Orderedstack', 'extOb1', 'extOb2'],
     'stack-1': ['Orderedstack', 'extOb2', 'extOb3']}

    """    

    return listsToRelations(extractComponentsFromList(instance,prefix=ext_ob_name,
        attrs=attrs),relationName=relation_name,appendAttr=appendAttr,attrs=attrs)

def santizeJSONList(instances,ext_ob_name="extOb",relation_name="Ordered",appendAttr=True,attrs = None):
    """

    Applies the santizeJSON function to a list of objects.

    """
    return [santizeJSON(instance,ext_ob_name,relation_name,appendAttr,attrs) for instance in instances]

def moving_average(a, n=3) :
    """
    A function for computing the moving average, so that we can smooth out the
    curves on a graph.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def lowess(x, y, f=1./3., iter=3, confidence=0.95):
    """
    Code taken from: https://gist.github.com/agramfort/850437

    lowess(x, y, f=2./3., iter=3) -> yest

    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.

    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(np.ceil(f*n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:,None] - x[None,:]) / h), 0.0, 1.0)
    w = (1 - w**3)**3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:,i]
            b = np.array([np.sum(weights*y), np.sum(weights*y*x)])
            A = np.array([[np.sum(weights), np.sum(weights*x)],
                   [np.sum(weights*x), np.sum(weights*x*x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1]*x[i]
 
        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta**2)**2

    h = np.zeros(n)
    for x_idx, x_val in enumerate(x):
        r2 = np.array([v*v for i, v in enumerate(residuals) if x[i] == x_val])
        n = len(r2)
        se = sqrt(mean(r2)) / sqrt(len(r2))
        h[x_idx] = se * t._ppf((1+confidence)/2., n-1)

    return yest, yest-h, yest+h

def avg_lines(x, y, confidence=0.95):
    n = len(x)
    mean = np.zeros(n)
    lower = np.zeros(n)
    upper = np.zeros(n)

    for x_idx, x_val in enumerate(x):
        ys = np.array([v for i,v in enumerate(y) if x[i] == x_val])
        m,l,u = mean_confidence_interval(ys)
        mean[x_idx] = m
        lower[x_idx] = l
        upper[x_idx] = u

    return mean, lower, upper


def weighted_choice(choices):
    """
    Given a list of tuples [(val, prob),...(val, prob)] this function returns a
    randomly chosen value where the choice is weighted by prob.
    """
    total = sum(w for c, w in choices)
    r = uniform(0, total)
    upto = 0
    for c, w in choices:
       if upto + w > r:
          return c
       upto += w
    assert False, "Shouldn't get here"

def mean_confidence_interval(data, confidence=0.95):
    """
    Given a list or vector of data, this returns the mean, lower, and upper
    confidence intervals to the level of confidence specified (default = 95%
    confidence interval).
    """
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

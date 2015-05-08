from math import sqrt
import numpy as np
from scipy import linalg
from numbers import Number

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
    	attrs = instance.keys()
    newInstance = {}
    for attr in instance:
        if isinstance(instance[attr], list) and attrs in attrs:
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

def numericToNominal(instance, attrs = None):     
	"""     	
	Takes a list of instances and converts any attributes that TRESTLE or
	COBWEB/3 would consider to be numeric attributes to nominal attributes in
	the case that should be treated as such. This is useful for when data
	natually contains values that would be numbers but you do not want to
	entertain that they have a distribution.	
	"""
	if attrs is None:
		attrs = instance.keys()
	for a in attrs:
		if isinstance(instance[a],Number):
			instance[a] = str(instnace[a])
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

def moving_average(a, n=3) :
    """
    A function for computing the moving average, so that we can smooth out the
    curves on a graph.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def lowess(x, y, f=2./3., iter=3):
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
 
    return yest

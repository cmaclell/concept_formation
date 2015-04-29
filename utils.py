import math
import uuid

# A hashtable of vlaues to use in the c4(n) function to apply corrections to
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
    table for n in [2-29] and returns 1.0 for vlaues >= 30.
    """
    if n <= 1 :
        raise ValueError("Cannot apply correction for a sample size of 1.")
    else :
        return c4n_table[n] if n < 30 else 1.0

def mean(values):
    """
    Computes the mean of a list of values.
    """
    if len(values) <= 0:
        raise ValueError("Length of list must be greater than 0.")

    return float(sum(values))/len(values)

def std(values):
    """
    Computes the standard deviation of a list of values.
    """
    if len(values) <= 0:
        raise ValueError("Length of list must be greater than 0.")

    meanValue = mean(values)
    variance =  float(sum([(v - meanValue) * (v - meanValue) for v in
                           values]))/len(values)
    return math.sqrt(variance)

class ContinuousValue():

    def __init__(self):
        """
        The number of values, the mean of the values, and the squared errors of
        the values.
        """
        self.num = 0
        self.mean = 0
        self.meanSq = 0

    def copy(self):
        """
        Returns a deep copy of itself.
        """
        v = ContinuousValue()
        v.num = self.num
        v.mean = self.mean
        v.meanSq = self.meanSq
        return v

    def unbiased_mean(self):
        """
        Returns the mean value.
        """
        return self.mean

    def biased_std(self):
        """
        Returns a biased estimate of the std (i.e., the sample std)
        """
        return math.sqrt(self.meanSq / (self.num))

    def unbiased_std(self):
        """
        Returns an unbiased estimate of the std that uses Bessel's correction
        and Cochran's theorem: 
            https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
        """
        if self.num < 2:
            return 0.0
        return math.sqrt(self.meanSq / (self.num - 1)) / c4(self.num)

    def __hash__(self):
        return hash("#ContinuousValue#")

    def __repr__(self):
        return repr(self.num) + repr(self.mean) + repr(self.meanSq)

    def __str__(self):
        return "%0.4f (%0.4f) [%i]" % (self.mean, self.unbiased_std(), self.num)

    def update_batch(self, data):
        """
        Calls the update function on every value in the given dataset
        """
        for x in data:
            self.update(x)

    def update(self, x):
        """
        Incrementally update the mean and squared mean error (meanSq) values in
        an efficient and practical (no precision problems) way. This uses and
        algorithm by Knuth, which I found here:
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        """
        self.num += 1
        delta = x - self.mean 
        self.mean += delta / self.num
        self.meanSq += delta * (x - self.mean)

    def combine(self, other):
        """
        Combine two clusters of means and squared mean error (meanSq) values in
        an efficient and practical (no precision problems) way. This uses the
        parallel algorithm by Chan et al. found here: 
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        """
        if not isinstance(other, ContinuousValue):
            raise ValueError("Can only merge 2 continuous values.")
        delta = other.mean - self.mean
        self.meanSq = (self.meanSq + other.meanSq + delta * delta * 
                       ((self.num * other.num) / (self.num + other.num)))
        self.mean = ((self.num * self.mean + other.num * other.mean) / 
                     (self.num + other.num))
        self.num += other.num

def listsToRelations(instance, relationName="Ordered", appendAttr=True):
    """
    Takes a structured instance containing lists and returns the same instance
    with all list elements converted to a series of relations that describe the
    ordering of elements. This process will be applied recurrsively to any
    commponent attributes of the instance.

    Example:
    
        {
            "list1":["a","b",c","d"]
        }

        becomes-

        {
            "uuid1":["Ordered", "a", "b"],
            "uuid2":["Ordered", "b", "c"],
            "uuid3":["Ordered", "c", "d"]
        }

    relationName -- The name that should be used to describe the relation, by
    default "Ordred" is used by a new name can be provided if that conflicts with
    other relations in the data already.

    appendAttr -- if True appends the original list's attribute name to the 
    relationName this is to prevent the matcher from over generalizing ordering
    when there are multiple lists in an object.
    """

    newInstance = {}
    for attr in instance:
        if isinstance(instance[attr], list):
            for i in range(len(instance[attr])-1):
                if appendAttr:
                    newInstance[str(uuid.uuid4())] = [str(relationName+attr),
                        instance[attr][i],
                        instance[attr][i+1]]
                else:
                    newInstance[str(uuid.uuid4())] = [str(relationName),
                        instance[attr][i],
                        instance[attr][i+1]]
        elif isinstance(instance[attr],dict):
            newInstance[attr] = listsToRelations(instance[attr],relationName,appendAttr)
        else:
            newInstance[attr] = instance[attr]
    return newInstance

def batchListsToRelations(instances,relationName="Ordered",appendAttr=True):
    """
    Takes a list of structured instances that contain lists and batch converts
    all the list elements to a relation format expected by TRESTLE
    """

    for i in len(instances):
        instances[i] = listsToRelations(instances[i],relationName,appendAttr)
    return instances

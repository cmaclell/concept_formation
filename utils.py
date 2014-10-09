# This file contains a number of utility functions that existed in the
# original python file. I've moved them here to simplify the python code.

import math

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

def combined_mean(m1,n1,m2,n2):
    """
    Function to compute the combined means given two means and the number
    of samples for each mean.
    """
    return (n1 * m1 + n2 * m2)/(n1 + n2)

def combined_unbiased_std(s1, m1, n1, s2, m2, n2, mX):
    """
    Computes a new mean from two estimated means and variances and n's as
    well as the combined mean.
    s1 = estimated std of sample 1
    m1 = mean of sample 1
    n1 = number values in sample 1

    s2 = estimated std of sample 2
    m2 = mean of sample 2
    n2 = number of values in sample 2

    mX = combined mean of two samples.
    """
    uc_s1 = s1
    uc_s2 = s2

    uc_s1 = uc_s1 * c4(n1)
    uc_s2 = uc_s2 * c4(n2)
    #if n1 > 1 and n1 < 30:
    #    uc_s1 = uc_s1 * utils.c4n_table[n1]
    #if n2 > 1 and n2 < 30:
    #    uc_s2 = uc_s2 * utils.c4n_table[n2]
    
    uc_std = math.sqrt(
        ((n1 * (math.pow(uc_s1,2) + math.pow((m1 - mX),2)) + 
          n2 * (math.pow(uc_s2,2) + math.pow((m2 - mX),2))) /
         (n1 + n2)))

    c4n = c4(n1 + n2)
    #c4n = 1.0
    #if (n1 + n2) < 30:
    #    c4n = utils.c4n_table[n1 + n2]

    # rounding correction due to summing small squares
    # this value was computed empirically with 1000 samples on 4/6/14
    # -Maclellan
    # TODO I'm not sure if this is a valid thing to do. I probably
    # just need a better algorithm.. see the parallel algorithm here:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    uc_std = uc_std / 1.0112143858578193

    return uc_std / c4n

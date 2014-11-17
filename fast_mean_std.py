import random
import math

# a hash table for fast c4n value lookup
c4n_table = {2: 0.7978845608028654, 3: 0.886226925452758, 4:
             0.9213177319235613, 5: 0.9399856029866254, 6: 0.9515328619481445,
             7: 0.9593687886998328, 8: 0.9650304561473722, 9:
             0.9693106997139539, 10: 0.9726592741215884, 11:
             0.9753500771452293, 12: 0.9775593518547722, 13:
             0.9794056043142177, 14: 0.9809714367555161, 15:
             0.9823161771626504, 16: 0.9834835316158412, 17:
             0.9845064054718315, 18: 0.985410043808079, 19: 0.9862141368601935,
             20: 0.9869342675246552, 21: 0.9875829288261562, 22:
             0.9881702533158311, 23: 0.988704545233999, 24: 0.9891926749585048,
             25: 0.9896403755857028, 26: 0.9900524688409107, 27:
             0.990433039209448, 28: 0.9907855696217323, 29: 0.9911130482419843}

def combined_biased_std(s1,m1,n1,s2,m2,n2,mX):

    uc_std = math.sqrt(((n1 * (s1 * s1 + (m1 - mX) * (m1 - mX)) + 
               n2 * (s2 * s2 + (m2 - mX) * (m2 - mX))) /
               (n1 + n2)))

    return uc_std

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

    if n1 > 1 and n1 < 30:
        uc_s1 = uc_s1 * c4n_table[n1]
    if n2 > 1 and n2 < 30:
        uc_s2 = uc_s2 * c4n_table[n2]
    
    uc_std = math.sqrt(
        ((n1 * (math.pow(uc_s1,2) + math.pow((m1 - mX),2)) + 
          n2 * (math.pow(uc_s2,2) + math.pow((m2 - mX),2))) /
         (n1 + n2)))

    c4n = 1.0
    if (n1 + n2) < 30:
        c4n = c4n_table[n1 + n2]

    # rounding correction due to summing small squares
    # this value was computed empirically with 1000 samples on 4/6/14
    # -Maclellan
    uc_std = uc_std / 1.0112143858578193

    return uc_std / c4n

def combined_mean(m1,n1,m2,n2):
    return (n1 * m1 + n2 * m2)/(n1 + n2)

def mean(values):
    """
    Computes the mean of a list of values.
    """
    if len(values) <= 0:
        raise ValueError("Length of list must be greater than 0.")

    return float(sum(values))/len(values)

def biased_std(values):
    """
    Computes the standard deviation of a list of values.
    """
    if len(values) <= 0:
        raise ValueError("Length of list must be greater than 0.")

    m = mean(values)
    variance =  float(sum([(v - m) * (v - m) for v in
                           values]))/len(values)
    return math.sqrt(variance)

def unbiased_std(sample):
    """
    This is an unbiased estimate of the std, which accounts for sample size
    if it is less than 30.
    
    The details of this unbiased estimate can be found here: 
        https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
    """
    if len(sample) <= 0:
        raise ValueError("No values in sample")

    if len(sample) == 1:
        return 0.0

    m = mean(sample)

    c4n = 1.0
    if len(sample) < 30:
        c4n = c4n_table[len(sample)]

        #c4n = (math.sqrt(2.0 / (len(sample) - 1.0)) *
        #       (math.gamma(len(sample)/2.0) /
        #        math.gamma((len(sample) - 1.0)/2.0)))

    variance = (float(sum([(v - m) * (v - m) for v in sample])) /
                (len(sample) - 1.0))

    std = math.sqrt(variance) / c4n

    # dont' need to correct for acuity here do it in CU calc
    #if std < acuity:
    #    std = acuity

    return std


if __name__ == "__main__":
    
    ratio = []
    for i in range(1000):
        for n in range(1,50):
            x = [random.normalvariate(0,100 * random.random()) for x in range(n)]
            y = [random.normalvariate(0,100 * random.random()) for x in range(n)]

            combined = x+y

            cm = combined_mean(mean(x),n,mean(y),n)
            #print(combined_mean(mean(x),n,mean(y),n), mean(combined))
            #assert combined_mean(mean(x),10,mean(y),10) == mean(combined)

            #print(combined_unbiased_std(unbiased_std(x), mean(x), n, unbiased_std(y),
            #                            mean(y), n, cm), unbiased_std(combined))
            ratio.append(combined_unbiased_std(unbiased_std(x), mean(x), n, unbiased_std(y),
                                        mean(y), n, cm) / unbiased_std(combined))
    #assert (combined_unbiased_std(unbiased_std(x), mean(x), 10,
    #                              unbiased_std(y), mean(y), 10, cm) ==
    #        unbiased_std(combined))

    print(mean(ratio))



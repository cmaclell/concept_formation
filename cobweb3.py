import math
import json
from random import normalvariate
from random import choice
from random import random
from cobweb import Cobweb

class ContinuousValue():

    # a hash table for fast c4n value lookup
    c4n_table = {2: 0.7978845608028654, 3: 0.886226925452758, 4:
                 0.9213177319235613, 5: 0.9399856029866254, 6:
                 0.9515328619481445, 7: 0.9593687886998328, 8:
                 0.9650304561473722, 9: 0.9693106997139539, 10:
                 0.9726592741215884, 11: 0.9753500771452293, 12:
                 0.9775593518547722, 13: 0.9794056043142177, 14:
                 0.9809714367555161, 15: 0.9823161771626504, 16:
                 0.9834835316158412, 17: 0.9845064054718315, 18:
                 0.985410043808079, 19: 0.9862141368601935, 20:
                 0.9869342675246552, 21: 0.9875829288261562, 22:
                 0.9881702533158311, 23: 0.988704545233999, 24:
                 0.9891926749585048, 25: 0.9896403755857028, 26:
                 0.9900524688409107, 27: 0.990433039209448, 28:
                 0.9907855696217323, 29: 0.9911130482419843}

    def __init__(self, mean, std, num):
        self.mean = mean
        self.std = std
        self.num = num

    def __hash__(self):
        return hash("#ContinuousValue#")

    def __str__(self):
        return "%0.4f (%0.4f) [%i]" % (self.mean, self.std, self.num)

    def combined_mean(self, m1,n1,m2,n2):
        """
        Function to compute the combined means given two means and the number
        of samples for each mean.
        """
        return (n1 * m1 + n2 * m2)/(n1 + n2)
    
    def combined_unbiased_std(self, s1, m1, n1, s2, m2, n2, mX):
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
            uc_s1 = uc_s1 * self.c4n_table[n1]
        if n2 > 1 and n2 < 30:
            uc_s2 = uc_s2 * self.c4n_table[n2]
        
        uc_std = math.sqrt(
            ((n1 * (math.pow(uc_s1,2) + math.pow((m1 - mX),2)) + 
              n2 * (math.pow(uc_s2,2) + math.pow((m2 - mX),2))) /
             (n1 + n2)))

        c4n = 1.0
        if (n1 + n2) < 30:
            c4n = self.c4n_table[n1 + n2]

        # rounding correction due to summing small squares
        # this value was computed empirically with 1000 samples on 4/6/14
        # -Maclellan
        # TODO I'm not sure if this is a valid thing to do. I probably
        # just need a better algorithm.. see the parallel algorithm here:
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        uc_std = uc_std / 1.0112143858578193

        return uc_std / c4n

    def update(self, n):
        self.combine_update(ContinuousValue(n, 0, 1))

    def combine_update(self, other):
        val = self.combine(other)
        self.mean = val.mean
        self.std = val.std
        self.num = val.num

    def combine(self, other):
        meanBoth = self.combined_mean(self.mean, self.num, other.mean,
                                      other.num)
        stdBoth = self.combined_unbiased_std(self.std, self.mean, self.num,
                                             other.std, other.mean, other.num,
                                             meanBoth)

        return ContinuousValue(meanBoth, stdBoth, self.num + other.num)

class Cobweb3(Cobweb):

    # Smallest possible acuity. Below this and probabilities will exceed 1.0
    acuity = 1.0 / math.sqrt(2.0 * math.pi)

    # a hash table for fast c4n value lookup
    c4n_table = {2: 0.7978845608028654, 3: 0.886226925452758, 4:
                 0.9213177319235613, 5: 0.9399856029866254, 6:
                 0.9515328619481445, 7: 0.9593687886998328, 8:
                 0.9650304561473722, 9: 0.9693106997139539, 10:
                 0.9726592741215884, 11: 0.9753500771452293, 12:
                 0.9775593518547722, 13: 0.9794056043142177, 14:
                 0.9809714367555161, 15: 0.9823161771626504, 16:
                 0.9834835316158412, 17: 0.9845064054718315, 18:
                 0.985410043808079, 19: 0.9862141368601935, 20:
                 0.9869342675246552, 21: 0.9875829288261562, 22:
                 0.9881702533158311, 23: 0.988704545233999, 24:
                 0.9891926749585048, 25: 0.9896403755857028, 26:
                 0.9900524688409107, 27: 0.990433039209448, 28:
                 0.9907855696217323, 29: 0.9911130482419843}

    def verify_counts(self):
        """
        Checks the property that the counts of the children sum to the same
        count as the parent. This is/was useful when debugging. If you are
        doing some kind of matching at each step in the categorization (i.e.,
        renaming such as with Labyrinth) then this will start throwing errors.
        """
        if len(self.children) == 0:
            return 

        temp = {}
        temp_count = self.count
        for attr in self.av_counts:
            if isinstance(self.av_counts[attr], ContinuousValue):
                temp[attr] = self.av_counts[attr].num
            else:
                if attr not in temp:
                    temp[attr] = {}
                for val in self.av_counts[attr]:
                    temp[attr][val] = self.av_counts[attr][val]

        for child in self.children:
            temp_count -= child.count
            for attr in child.av_counts:
                assert attr in temp
                if isinstance(child.av_counts[attr], ContinuousValue):
                    temp[attr] -= child.av_counts[attr].num
                else:
                    for val in child.av_counts[attr]:
                        if val not in temp[attr]:
                            print(val.concept_name)
                            print(attr)
                            print(self)
                        assert val in temp[attr]
                        temp[attr][val] -= child.av_counts[attr][val]

        assert temp_count == 0

        for attr in temp:
            if isinstance(temp[attr], int):
                assert temp[attr] == 0.0
            else:
                for val in temp[attr]:
                    assert temp[attr][val] == 0.0

        for child in self.children:
            child.verify_counts()

    def increment_counts(self, instance):
        """
        A modified version of increment counts that handles floats properly

        input:
            instance: {a1: v1, a2: v2, ...} - a hashtable of attr and values. 
        """
        self.count += 1 
            
        for attr in instance:
            if isinstance(instance[attr], float):
                if (attr not in self.av_counts or 
                    not isinstance(self.av_counts[attr], ContinuousValue)):
                    # TODO currently overrides nominals if a float comes in.
                    self.av_counts[attr] = ContinuousValue(instance[attr], 0, 1)
                else:
                    self.av_counts[attr].update(instance[attr])

            else:
                self.av_counts[attr] = self.av_counts.setdefault(attr,{})
                self.av_counts[attr][instance[attr]] = (self.av_counts[attr].get(instance[attr], 0) + 1)

    def update_counts_from_node(self, node):
        """
        modified to handle floats
        Increments the counts of the current node by the amount in the specified
        node.
        """
        self.count += node.count
        for attr in node.av_counts:
            if isinstance(node.av_counts[attr], ContinuousValue):
                if (attr not in self.av_counts or 
                    not isinstance(self.av_counts[attr], ContinuousValue)):
                    # TODO currently overrides nominals if a float comes in.
                    oldval = node.av_counts[attr]
                    self.av_counts[attr] = ContinuousValue(oldval.mean,
                                                           oldval.std,
                                                           oldval.num)
                else:
                    self.av_counts[attr].combine_update(node.av_counts[attr])
            else:
                for val in node.av_counts[attr]:
                    self.av_counts[attr] = self.av_counts.setdefault(attr,{})
                    self.av_counts[attr][val] = (self.av_counts[attr].get(val,0) +
                                         node.av_counts[attr][val])
    
    def unbiased_std(self, sample):
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

        m = self.mean(sample)

        c4n = 1.0
        if len(sample) < 30:
            c4n = self.c4n_table[len(sample)]

            # use the lookup table it is much faster
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

    def expected_correct_guesses(self):
        """
        Computes the number of attribute values that would be correctly guessed
        in the current concept. This extension supports both nominal and
        numeric attribute values. The acuity parameter should be set based on
        the domain cobweb is being used on. The acuity is set as a global
        parameter now. 
        """
        correct_guesses = 0.0

        for attr in self.av_counts:
            if isinstance(self.av_counts[attr], ContinuousValue):
                std = self.av_counts[attr].std
                if std < self.acuity:
                    std = self.acuity
                # this is implicit in the nominal case, but here it must be
                # computed explicitly in addition to the std based cu calc. 
                # -CM
                prob_attr = ((1.0 * self.av_counts[attr].num) / self.count)
                correct_guesses += ((prob_attr * prob_attr) * 
                                    (1.0 / (2.0 * math.sqrt(math.pi) * std)))
            else:
                for val in self.av_counts[attr]:
                    prob = ((1.0 * self.av_counts[attr][val]) / self.count)
                    correct_guesses += (prob * prob)

        return correct_guesses

    def pretty_print(self, depth=0):
        """
        Prints the categorization tree.
        """
        ret = str(('\t' * depth) + "|-")

        attributes = []

        for attr in self.av_counts:
            if isinstance(self.av_counts[attr], ContinuousValue):
                attributes.append("'%s': { %0.3f (%0.3f) [%i] }" % (attr,
                                                                    self.av_counts[attr].mean,
                                                                    self.av_counts[attr].std,
                                                                    self.av_counts[attr].num))
            else:
                values = []

                for val in self.av_counts[attr]:
                    values.append("'" + str(val) + "': " +
                                  str(self.av_counts[attr][val]))

                attributes.append("'" + attr + "': {" + ", ".join(values) + "}")
                  
        ret += "{" + ", ".join(attributes) + "}: " + str(self.count) + '\n'
        
        for c in self.children:
            ret += c.pretty_print(depth+1)

        return ret

    def predict(self, instance):
        """
        Given an instance predict any missing attribute values without
        modifying the tree. This has been modified to make predictions about
        nominal and numeric attribute values. 
        """
        prediction = {}

        # make a copy of the instance
        for attr in instance:
            prediction[attr] = instance[attr]

        concept = self._cobweb_categorize(instance)
        
        for attr in concept.av_counts:
            if attr in prediction:
                continue
            
            nominal_values = []
            #float_values = []

            num_floats = 0
            mean = 0.0
            std = 0.0
            for val in concept.av_counts[attr]:
                if isinstance(val, ContinuousValue):
                    num_floats = val.num
                    mean = val.mean
                    std = val.std
                else:
                    nominal_values += [val] * concept.av_counts[attr][val]

            if random() < ((len(nominal_values) * 1.0) / (len(nominal_values) +
                                                          num_floats)):
                prediction[attr] = choice(nominal_values)
            else:
                prediction[attr] = normalvariate(mean,
                                                 std)

        return prediction

    def get_probability(self, attr, val):
        """
        Gets the probability of a particular attribute value. This has been
        modified to support numeric and nominal values. The acuity is set as a
        global parameter now. 
        """
        if attr not in self.av_counts:
            return 0.0

        if isinstance(val, ContinuousValue):
            #float_values = []

            #for av in self.av_counts[attr]:
            #    if isinstance(av, float):
            #        float_values += [av] * self.av_counts[attr][av]

            #mean = self.mean(float_values)
            #std = self.unbiased_std(float_values)
            mean = val.mean
            std = val.std

            if std < self.acuity:
                std = self.acuity

            # assign 100% accuracy to the mean
            # TODO does this need to be scaled? so the area under the curve=1?
            return (math.exp(-((val - mean) * (val - mean)) / (2.0 * std * std)))
                                                                           
        if val in self.av_counts[attr]:
            return (1.0 * self.av_counts[attr][val]) / self.count

        return 0.0

    def output_json(self):
        """
        A modification of the cobweb output json to handle numeric values.
        """
        output = {}
        output["name"] = self.concept_name
        output["size"] = self.count
        output["children"] = []

        temp = {}
        for attr in self.av_counts:
            #float_vals = []
            if isinstance(self.av_counts[attr], ContinuousValue):
                temp[str(attr) + " = " + str(self.av_counts[attr])] = self.av_counts[attr].num
            else:
                for value in self.av_counts[attr]:
                    temp[str(attr) + " = " + str(value)] = self.av_counts[attr][value]

        for child in self.children:
            output["children"].append(child.output_json())

        output["counts"] = temp

        return output

if __name__ == "__main__":

    #Cobweb3().predictions("data_files/cobweb3_test3.json", 10, 20)
    #Cobweb3Tree().baseline_guesser("data_files/cobweb3_test.json", 30, 100)
    tree = Cobweb3()
    print(tree.cluster("data_files/cobweb3_test.json", 100, 1))
    print(json.dumps(tree.output_json()))




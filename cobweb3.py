import math
import json
import utils
from random import normalvariate
from random import choice
from random import random
from cobweb import Cobweb

class ContinuousValue():

    def __init__(self, mean, std, num):
        self.mean = mean
        self.std = std
        self.num = num

    def __hash__(self):
        return hash("#ContinuousValue#")

    def __str__(self):
        return "%0.4f (%0.4f) [%i]" % (self.mean, self.std, self.num)

    def update(self, n):
        self.combine_update(ContinuousValue(n, 0, 1))

    def combine_update(self, other):
        val = self.combine(other)
        self.mean = val.mean
        self.std = val.std
        self.num = val.num

    def combine(self, other):
        meanBoth = utils.combined_mean(self.mean, self.num, other.mean,
                                      other.num)
        stdBoth = utils.combined_unbiased_std(self.std, self.mean, self.num,
                                             other.std, other.mean, other.num,
                                             meanBoth)

        return ContinuousValue(meanBoth, stdBoth, self.num + other.num)

class Cobweb3(Cobweb):

    # Smallest possible acuity. Below this and probabilities will exceed 1.0
    acuity = 1.0 / math.sqrt(2.0 * math.pi)

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

        m = utils.mean(sample)

        c4n = utils.c4(len(sample))
        #c4n = 1.0
        #if len(sample) < 30:
        #    c4n = utils.c4n_table[len(sample)]

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

            #mean = utils.mean(float_values)
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




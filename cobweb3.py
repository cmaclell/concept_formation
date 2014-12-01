import math
from utils import ContinuousValue
from random import normalvariate
from random import choice
from random import random
from cobweb import CobwebNode, CobwebTree

class Cobweb3Tree(CobwebTree):

    def __init__(self):
        self.root = Cobweb3Node()

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
                    std = val.unbiased_std()
                else:
                    nominal_values += [val] * concept.av_counts[attr][val]

            if random() < ((len(nominal_values) * 1.0) / (len(nominal_values) +
                                                          num_floats)):
                prediction[attr] = choice(nominal_values)
            else:
                prediction[attr] = normalvariate(mean,
                                                 std)

        return prediction

class Cobweb3Node(CobwebNode):

    # Smallest possible acuity. Below this and probabilities will exceed 1.0
    acuity = 1.0 / math.sqrt(2.0 * math.pi)

    def increment_counts(self, instance):
        """
        A modified version of increment counts that handles floats properly
        """
        self.count += 1 
            
        for attr in instance:
            if isinstance(instance[attr], float):
                if (attr not in self.av_counts or 
                    not isinstance(self.av_counts[attr], ContinuousValue)):
                    # TODO currently overrides nominals if a float comes in.
                    self.av_counts[attr] = ContinuousValue()
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
                    self.av_counts[attr] = ContinuousValue()
                self.av_counts[attr].combine(node.av_counts[attr])
            else:
                for val in node.av_counts[attr]:
                    self.av_counts[attr] = self.av_counts.setdefault(attr,{})
                    self.av_counts[attr][val] = (self.av_counts[attr].get(val,0) +
                                         node.av_counts[attr][val])
    
    def attr_val_guess_gain(self, attr, val):
        """
        Returns the gain in number of correct guesses if a particular attr/val
        was added to a concept.
        """
        if attr[0] == "_":
            return 0.0
        elif attr not in self.av_counts:
            return 0.0
        elif isinstance(self.av_counts[attr], ContinuousValue):
            before_std = max(self.av_counts[attr].unbiased_std(), self.acuity)
            before_prob = ((1.0 * self.av_counts[attr].num) / (self.count + 1.0))
            before_count = ((before_prob * before_prob) * 
                            (1.0 / (2.0 * math.sqrt(math.pi) * before_std)))
            temp = self.av_counts[attr].copy()
            temp.update(val)
            after_std = max(temp.unbiased_std(), self.acuity)
            after_prob = ((1.0 + self.av_counts[attr].num) / (self.count + 1.0))
            after_count = ((after_prob * after_prob) * 
                            (1.0 / (2.0 * math.sqrt(math.pi) * after_std)))
            return after_count - before_count
        elif val not in self.av_counts[attr]:
            return 0.0
        else:
            before_prob = (self.av_counts[attr][val] / (self.count + 1.0))
            after_prob = (self.av_counts[attr][val] + 1) / (self.count + 1.0)

            return (after_prob * after_prob) - (before_prob * before_prob)

    def expected_correct_guesses(self):
        """
        Computes the number of attribute values that would be correctly guessed
        in the current concept. This extension supports both nominal and
        numeric attribute values. 
        
        The typical cobweb 3 calculation for correct guesses is:
            P(A_i = V_ij)^2 = 1 / (2 * sqrt(pi) * std)

        However, this does not take into account situations when P(A_i) != 1.0.
        To account for this we use a modified equation:
            P(A_i = V_ij)^2 = P(A_i)^2 * (1 / (2 * sqrt(pi) * std))
        """
        correct_guesses = 0.0

        for attr in self.av_counts:
            if attr[0] == "_":
                continue
            elif isinstance(self.av_counts[attr], ContinuousValue):
                std = max(self.av_counts[attr].unbiased_std(), self.acuity)
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
                                                                    max(self.acuity,
                                                                        self.av_counts[attr].unbiased_std()),
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

    def get_probability(self, attr, val):
        """
        Gets the probability of a particular attribute value. This has been
        modified to support numeric and nominal values. The acuity is set as a
        global parameter now. 
        """
        if attr not in self.av_counts:
            return 0.0

        if isinstance(val, ContinuousValue):
            mean = val.mean
            std = val.unbiased_std()

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
        if "_guid" in self.av_counts:
            for guid in self.av_counts['_guid']:
                output['guid'] = guid
        output["name"] = "Concept" + self.concept_id
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

    tree = Cobweb3Tree()

    data = [{'x': normalvariate(0,0.5)} for i in range(10)]
    data += [{'x': normalvariate(2,0.5)} for i in range(10)]
    data += [{'x': normalvariate(4,0.5)} for i in range(10)]

    clusters = tree.cluster(data)
    print(clusters)
    print(set(clusters))



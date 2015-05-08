import math
from numbers import Number
from utils import ContinuousValue
from random import normalvariate
from random import choice
from random import random
from cobweb import CobwebNode, CobwebTree

class Cobweb3Tree(CobwebTree):

    def __init__(self):
        self.root = Cobweb3Node()
        self.root.root = self.root

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
            if isinstance(instance[attr], Number):
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
    
    def attr_val_guess_gain(self, attr, val, scaling=True):
        """
        Returns the gain in number of correct guesses if a particular attr/val
        was added to a concept.

        The scaling parameter determines whether online normalization is used.
        This approach computes the amount of scaling prior to incorporating the
        new attribute value.
        """
        if attr[0] == "_":
            return 0.0
        elif attr not in self.av_counts:
            return 0.0
        elif isinstance(self.av_counts[attr], ContinuousValue):
            if scaling:
                scale = self.root.av_counts[attr].unbiased_std()
            else:
                scale = 1.0

            before_std = max(self.av_counts[attr].scaled_unbiased_std(scale), self.acuity)
            before_prob = ((1.0 * self.av_counts[attr].num) / (self.count + 1.0))
            before_count = ((before_prob * before_prob) * 
                            (1.0 / (2.0 * math.sqrt(math.pi) * before_std)))

            temp = self.av_counts[attr].copy()
            temp.update(val)
            after_std = max(temp.scaled_unbiased_std(scale), self.acuity)
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

    def expected_correct_guesses(self, alpha=0.001, scaling=True):
        """
        Computes the number of attribute values that would be correctly guessed
        in the current concept. This extension supports both nominal and
        numeric attribute values. 
        
        The typical cobweb 3 calculation for correct guesses is:
            P(A_i = V_ij)^2 = 1 / (2 * sqrt(pi) * std)

        However, this does not take into account situations when P(A_i) != 1.0.
        To account for this we use a modified equation:
            P(A_i = V_ij)^2 = P(A_i)^2 * (1 / (2 * sqrt(pi) * std))

        The alpha parameter is the parameter used for laplacian smoothing. The
        higher the value, the higher the prior that all attributes/values are
        equally likely. By default a minor smoothing is used: 0.001.

        The scaling parameter determines whether online normalization of
        continuous attributes is used. By default scaling is used. Scaling
        divides the std of each attribute by the std of the attribute in the
        root node. 
        """
        correct_guesses = 0.0

        for attr in self.root.av_counts:
            if attr[0] == "_":
                continue
            elif isinstance(self.root.av_counts[attr], ContinuousValue):
                n_values = 2
                if attr not in self.av_counts :
                    prob = 0
                    if alpha > 0:
                        prob = alpha / (alpha * n_values)
                    val_count = 0
                else:
                    val_count = self.av_counts[attr].num

                    if scaling:
                        scale = self.root.av_counts[attr].unbiased_std()
                    else:
                        scale = 1.0

                    std = max(self.av_counts[attr].scaled_unbiased_std(scale),
                              self.acuity)
                    prob_attr = ((1.0 * self.av_counts[attr].num + alpha) /
                                 (self.count + alpha * n_values ))
                    correct_guesses += ((prob_attr * prob_attr) * 
                                        (1.0 / (2.0 * math.sqrt(math.pi) * std)))

                #Factors in the probability mass of missing values
                prob = ((self.count - val_count + alpha) / (1.0 * self.count +
                                                            alpha * 2))
                correct_guesses += (prob * prob)

            else:
                val_count = 0
                n_values = len(self.root.av_counts[attr]) + 1
                for val in self.root.av_counts[attr]:
                    if attr not in self.av_counts or val not in self.av_counts[attr]:
                        prob = 0
                        if alpha > 0:
                            prob = alpha / (alpha * n_values)
                    else:
                        val_count += self.av_counts[attr][val]
                        prob = ((self.av_counts[attr][val] + alpha) / (1.0 * self.count + 
                                                                       alpha * n_values))
                    correct_guesses += (prob * prob)

                #Factors in the probability mass of missing values
                prob = ((self.count - val_count + alpha) / (1.0*self.count + alpha *
                                                    n_values))
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

    def get_probability(self, attr, val, alpha=0.001, scaling=True):
        """
        Gets the probability of a particular attribute value. This takes into
        account the possibility that a value is missing, it uses laplacian
        smoothing (alpha) and it normalizes the values using the root concept
        if scaling is enabled (scaling).
        """
        if attr not in self.av_counts:
            return 0.0

        if isinstance(self.av_counts[attr], ContinuousValue):
            n_values = 2

            if scaling:
                scale = self.root.av_counts[attr].unbiased_std()
                shift = self.root.av_counts[attr].mean
                val = val - shift
            else:
                scale = 1.0

            mean = (self.av_counts[attr].mean - shift) / scale
            std = max(self.av_counts[attr].scaled_unbiased_std(scale),
                      self.acuity)

            prob_attr = ((1.0 * self.av_counts[attr].num + alpha) /
                         (self.count + alpha * n_values ))

            return (prob_attr * math.exp(-((val - mean) * (val - mean)) / 
                                         (2.0 * std * std)))
        else:
            return super(Cobweb3Node ,self).get_probability(attr, val, alpha)

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



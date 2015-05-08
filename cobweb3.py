from math import sqrt
from math import pi
from math import exp
from numbers import Number
from random import normalvariate

from utils import c4
from cobweb import CobwebNode, CobwebTree

class Cobweb3Tree(CobwebTree):

    def __init__(self):
        self.root = Cobweb3Node()
        self.root.root = self.root

class Cobweb3Node(CobwebNode):

    # Smallest possible acuity. Below this and probabilities will exceed 1.0
    acuity = 1.0 / sqrt(2.0 * pi)

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

            # TODO consider incorporating laplace smoothing (alpha).
            before_std = max(self.av_counts[attr].scaled_unbiased_std(scale), self.acuity)
            before_prob = ((1.0 * self.av_counts[attr].num) / (self.count + 1.0))
            before_count = ((before_prob * before_prob) * 
                            (1.0 / (2.0 * sqrt(pi) * before_std)))

            temp = self.av_counts[attr].copy()
            temp.update(val)
            after_std = max(temp.scaled_unbiased_std(scale), self.acuity)
            after_prob = ((1.0 + self.av_counts[attr].num) / (self.count + 1.0))
            after_count = ((after_prob * after_prob) * 
                            (1.0 / (2.0 * sqrt(pi) * after_std)))
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
                                        (1.0 / (2.0 * sqrt(pi) * std)))

                #Factors in the probability mass of missing values
                prob = ((self.count - val_count + alpha) / (1.0 * self.count +
                                                            alpha * n_values))
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

            return (prob_attr * exp(-((val - mean) * (val - mean)) / 
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

class ContinuousValue():
    """ 
    This class scores the number of samples, the mean of the samples, and the
    squared error of the samples. It can be used to perform incremental
    estimation of the mean, std, and unbiased std.
    """

    def __init__(self):
        """
        Initializes the number of values, the mean of the values, and the
        squared errors of the values to 0.
        """
        self.num = 0.0
        self.mean = 0.0
        self.meanSq = 0.0

    def __len__(self):
        return 1

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
        return sqrt(self.meanSq / (self.num))

    def unbiased_std(self):
        """
        Returns an unbiased estimate of the std that uses Bessel's correction
        and Cochran's theorem: 
            https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
        """
        if self.num < 2:
            return 0.0
        return sqrt(self.meanSq / (self.num - 1)) / c4(self.num)

    def scaled_unbiased_std(self, scale):
        """
        Returns an unbiased estimate of the std (see comments on unbiased_std),
        but also adjusts the std given a scale parameter. This is used to
        return std values that have been normalized by some value.

        For edge cases, if scale is less than or equal to 0, then scaling is
        disabled (i.e., scale = 1.0).
        """
        if scale <= 0:
            scale = 1.0
        return self.unbiased_std() / scale

    def __hash__(self):
        """
        This hashing function returns the hash of a constant string, so that
        all lookups of a continuous value in a dictionary get mapped to the
        same entry. 
        """
        return hash("#ContinuousValue#")

    def __repr__(self):
        """
        The representation of a continuous value.
        """
        return repr(self.num) + repr(self.mean) + repr(self.meanSq)

    def __str__(self):
        """
        The string format for a continuous value."
        """
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

if __name__ == "__main__":

    tree = Cobweb3Tree()

    data = [{'x': normalvariate(0,0.5)} for i in range(10)]
    data += [{'x': normalvariate(2,0.5)} for i in range(10)]
    data += [{'x': normalvariate(4,0.5)} for i in range(10)]

    clusters = tree.cluster(data)
    print(clusters)
    print(set(clusters))



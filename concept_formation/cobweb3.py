from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from random import normalvariate
from numbers import Number
from math import sqrt
from math import pi
from math import exp

from concept_formation.utils import c4
from concept_formation.utils import weighted_choice
from concept_formation.utils import most_likely_choice
from concept_formation.cobweb import CobwebNode
from concept_formation.cobweb import CobwebTree

class Cobweb3Tree(CobwebTree):
    """
    The Cobweb3Tree contains the knoweldge base of a partiucluar instance of the
    Cobweb/3 algorithm and can be used to fit and categorize instances.
    Cobweb/3's main difference over Cobweb is the ability to handle numerical
    attributes by applying an assumption that they should follow a normal
    distribution. For the purposes of Cobweb/3's core algorithms a numeric
    attribute is any value where ``isinstance(instance[attr], Number)`` returns
    ``True``.

    The alpha parameter is the parameter used for laplacian smoothing of
    nominal values (or whether an attribute is present or not for both
    nominal and numeric attributes). The higher the value, the higher the
    prior that all attributes/values are equally likely. By default a minor
    smoothing is used: 0.001.

    The scaling parameter determines whether online normalization of
    continuous attributes is used. By default scaling is used. Scaling
    divides the std of each attribute by the std of the attribute in the
    parent node (no scaling is performed in the root). Scaling is useful to
    balance the weight of different numerical attributes, without scaling
    the magnitude of numerical attributes can affect category utility
    calculation meaning numbers that are naturally larger will recieve
    extra weight in the calculation.

    :param alpha: constant to use for laplacian smoothing.
    :type alpha: float
    :param scaling: whether or not numerical values should be scaled in
        online normalization.
    :type scaling: bool
    """

    def __init__(self, alpha=0.001, scaling=True):
        """The tree constructor."""

        self.root = Cobweb3Node()
        self.root.tree = self
        self.alpha = alpha
        self.scaling = scaling
        self.std_to_scale = 1.0

    def infer_missing(self, instance, choice_fn="most likely"):
        """
        Given a tree and an instance, returns a new instance with attribute 
        values picked using the specified choice function (wither "most likely"
        or "sampled"). 

        :param instance: an instance to be completed.
        :type instance: {a1: v1, a2: v2, ...}
        :param choice_fn: a string specifying the choice function to use,
            either "most likely" or "sampled". 
        :type choice_fn: a string
        :return: A completed instance
        :rtype: instance
        """
        if choice_fn == "most likely" or choice_fn == "m":
            choice_fn = most_likely_choice
        elif choice_fn == "sampled" or choice_fn == "s":
            choice_fn = weighted_choice
        else:
            raise Exception("Unknown choice_fn")

        temp_instance = {a:instance[a] for a in instance}
        concept = self._cobweb_categorize(temp_instance)

        for attr in concept.av_counts:
            if attr in temp_instance:
                continue

            missing_prob = concept.get_probability_missing(attr)
            attr_choices = ((None, missing_prob), (attr, 1 - missing_prob))
            if choice_fn(attr_choices) == attr:

                if isinstance(concept.av_counts[attr], ContinuousValue):
                    if choice_fn == most_likely_choice:
                        temp_instance[attr] = concept.av_counts[attr].unbiased_mean()
                    else:
                        temp_instance[attr] = normalvariate(concept.av_counts[attr].unbiased_mean(),
                                                            concept.av_counts[attr].unbiased_std())
                else:
                    temp_instance[attr] = choice_fn(concept.get_weighted_values(attr))

        return temp_instance

    def clear(self):
        """Clears the concepts of the tree, but maintains the alpha and
        scaling parameters.
        """
        self.root = Cobweb3Node()
        self.root.tree = self

class Cobweb3Node(CobwebNode):
    """
    A Cobweb3Node represents a concept within the knoweldge base of a particular
    :class:`Cobweb3Tree`. Each node contians a probability table that can be used to
    calculate the probability of different attributes given the concept that the
    node represents.

    In general the :meth:`Cobweb3Tree.ifit`, :meth:`Cobweb3Tree.categorize`
    functions should be used to initially interface with the Cobweb/3 knowledge
    base and then the returned concept can be used to calculate probabilities of
    certain attributes or determine concept labels.
    """

    # Smallest possible acuity. Below this probabilities will exceed 1.0
    acuity = 1.0 / sqrt(2.0 * pi)
    """
    acuity is used as a floor on standard deviation estimates for numeric
    attribute values. The default value is set to 
    :math:`\\frac{1}{\\sqrt{2 * \\pi}}` which is the smallest possible acuity
    before probability estimates begin to exceed 1.0.
    """

    def get_probability_missing(self, attr):
        """
        Returns the probability of a particular attribute not being present in a
        given concept.

        This takes into account the possibilities that an attribute can take any
        of the values available at the root, or be missing. Laplace smoothing is
        used to place a prior over these possibilites. Alpha determines the
        strength of this prior.

        :param attr: an attribute of an instance
        :type attr: str
        :return: The probability of attr not being present from an instance in
            the current concept.
        :rtype: float 
        """
        # the +1 is for the "missing" value
        if attr in self.tree.root.av_counts:
            n_values = len(self.tree.root.av_counts[attr]) + 1
        else:
            n_values = 1

        val_count = 0
        if attr in self.av_counts:
            if isinstance(self.av_counts[attr], ContinuousValue):
                val_count += self.av_counts[attr].num
            else:
                for val in self.av_counts[attr]:
                    val_count += self.av_counts[attr][val]

        if (1.0 * self.count + self.tree.alpha * n_values) == 0:
            return 0.0

        return ((self.count - val_count + self.tree.alpha) / 
                (1.0 * self.count + self.tree.alpha * n_values))

    def increment_counts(self, instance):
        """
        Increment the counts at the current node according to the specified
        instance.

        Cobweb3Node uses a modified version of
        :meth:`CobwebNode.increment_counts
        <concept_formation.cobweb.CobwebNode.increment_counts>` that handles
        numerical attributes properly. Any attribute value where
        ``isinstance(instance[attr], Number)`` returns ``True`` will be treated
        as a numerical attribute and included under an assumption that the
        number should follow a normal distribution.

        .. warning:: If a numeric attribute is found in an instance with the
            name of a previously nominal attribute, or vice versa, this function will raise
            an exception. See: :class:`NumericToNominal
            <concept_formation.preprocessor.NumericToNominal>` for a way to fix this error.
        
        :param instance: A new instances to incorporate into the node.
        :type instance: {a1: v1, a2: v2, ...} - a hashtable of attr and values,
            where values can be numeric or nominal.

        """
        self.count += 1 
            
        for attr in instance:
            if isinstance(instance[attr], Number):
                if attr not in self.av_counts:
                    self.av_counts[attr] = ContinuousValue()
                elif not isinstance(self.av_counts[attr], ContinuousValue):
                    raise Exception ('Numerical value found in nominal attribute. Try casting all values of "'+attr+'" to either string or a number type.')
                    
                self.av_counts[attr].update(instance[attr])
            else:
                if attr in self.av_counts and isinstance(self.av_counts[attr],ContinuousValue):
                    raise Exception ('Nominal value found in numerical attribute. Try casting all values of "'+attr+'" to either string or a number type.')
                self.av_counts[attr] = self.av_counts.setdefault(attr,{})
                self.av_counts[attr][instance[attr]] = (self.av_counts[attr].get(instance[attr], 0) + 1)

    def update_counts_from_node(self, node):
        """
        Increments the counts of the current node by the amount in the specified
        node, modified to handle numbers.

        .. warning:: If a numeric attribute is found in an instance with the
            name of a previously nominal attribute, or vice versa, this function will raise
            an exception. See: :class:`NumericToNominal
            <concept_formation.preprocessor.NumericToNominal>` for a way to fix this error.

        :param node: Another node from the same Cobweb3Tree
        :type node: Cobweb3Node
        """
        self.count += node.count
        for attr in node.av_counts:
            if isinstance(node.av_counts[attr], ContinuousValue):
                if attr not in self.av_counts:
                    self.av_counts[attr] = ContinuousValue()
                elif not isinstance(self.av_counts[attr], ContinuousValue):
                    raise Exception ('Numerical value found in nominal attribute. Try casting all values of "'+attr+'" to either string or a number type.')
                    
                self.av_counts[attr].combine(node.av_counts[attr])
            else:
                if attr in self.av_counts and isinstance(self.av_counts[attr],ContinuousValue):
                    raise Exception ('Nominal value found in numerical attribute. Try casting all values of "'+attr+'" to either string or a number type.')
                for val in node.av_counts[attr]:
                    self.av_counts[attr] = self.av_counts.setdefault(attr,{})
                    self.av_counts[attr][val] = (self.av_counts[attr].get(val,0) +
                                         node.av_counts[attr][val])
    
    def attr_val_guess_gain(self, attr, val):
        """
        Returns the gain in number of correct guesses if a particular attr/val
        was added to a concept.

        :param attr: An attribute in the concept
        :type attr: str
        :param val: A value for the given attribute in the concept
        :type val: float or str
        :return:  the gain in number of correct guesses from adding the partiucluar attr/val
        :rtype: float               
        """

        if attr[0] == "_":
            return 0.0
        elif attr not in self.av_counts:
            return 0.0
        elif isinstance(self.av_counts[attr], ContinuousValue):
            if self.tree.scaling and self.parent:
                scale = self.tree.std_to_scale * self.parent.av_counts[attr].unbiased_std()
            else:
                scale = 1.0

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

    def expected_correct_guesses(self):
        """
        Returns the number of attribute values that would be correctly guessed
        in the current concept. This extension supports both nominal and
        numeric attribute values. 
        
        The typical Cobweb/3 calculation for correct guesses is:

        .. math::

            P(A_i = V_{ij})^2 = \\frac{1}{2 * \\sqrt{\\pi} * \\sigma}

        However, this does not take into account situations when 
        :math:`P(A_i) \\neq 1.0`.

        To account for this we use a modified equation:

        .. math::

            P(A_i = V_{ij})^2 = P(A_i)^2 * \\frac{1}{2 * \\sqrt{\\pi} * \\sigma}

        :return: The number of attribute values that would be correctly guessed
            in the current concept.
        :rtype: float
        """
        correct_guesses = 0.0

        for attr in self.tree.root.av_counts:
            if attr[0] == "_":
                continue
            elif isinstance(self.tree.root.av_counts[attr], ContinuousValue):
                n_values = 2
                if attr not in self.av_counts :
                    prob = 0
                    if self.tree.alpha > 0:
                        prob = self.tree.alpha / (self.tree.alpha * n_values)
                    val_count = 0
                else:
                    val_count = self.av_counts[attr].num

                    if self.tree.scaling and self.parent:
                        scale = self.tree.std_to_scale * self.parent.av_counts[attr].unbiased_std()
                    else:
                        scale = 1.0

                    std = max(self.av_counts[attr].scaled_unbiased_std(scale),
                              self.acuity)
                    prob_attr = ((1.0 * self.av_counts[attr].num + self.tree.alpha) /
                                 (self.count + self.tree.alpha * n_values ))
                    correct_guesses += ((prob_attr * prob_attr) * 
                                        (1.0 / (2.0 * sqrt(pi) * std)))

                #Factors in the probability mass of missing values
                prob = ((self.count - val_count + self.tree.alpha) / (1.0 * self.count +
                                                            self.tree.alpha * n_values))
                correct_guesses += (prob * prob)

            else:
                val_count = 0
                n_values = len(self.tree.root.av_counts[attr]) + 1
                for val in self.tree.root.av_counts[attr]:
                    if attr not in self.av_counts or val not in self.av_counts[attr]:
                        prob = 0
                        if self.tree.alpha > 0:
                            prob = self.tree.alpha / (self.tree.alpha * n_values)
                    else:
                        val_count += self.av_counts[attr][val]
                        prob = ((self.av_counts[attr][val] + self.tree.alpha) / (1.0 * self.count + 
                                                                       self.tree.alpha * n_values))
                    correct_guesses += (prob * prob)

                #Factors in the probability mass of missing values
                prob = ((self.count - val_count + self.tree.alpha) /
                        (1.0*self.count + self.tree.alpha *
                                                    n_values))
                correct_guesses += (prob * prob)

        return correct_guesses

    def pretty_print(self, depth=0):
        """
        Print the categorization tree

        The string formatting inserts tab characters to align child nodes of the
        same depth. Numerical values are printed with their means and standard
        deviations.
        
        :param depth: The current depth in the print, intended to be called recursively
        :type depth: int
        :return: a formated string displaying the tree and its children
        :rtype: str
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

                attributes.append("'" + str(attr) + "': {" + ", ".join(values) + "}")
                  
        ret += "{" + ", ".join(attributes) + "}: " + str(self.count) + '\n'
        
        for c in self.children:
            ret += c.pretty_print(depth+1)

        return ret

    def sample(self, attr):
        """Samples the value of an attribute from the node's probability table.
        This takes into account the laplacian smoothing.

        If the attribute is a nominal then this function behaves the same as
        :meth:`CobwebNode.sample <concept_formation.cobweb.CobwebNode.sample>`.
        If the attribute is numeric then a value is sampled from the normal
        distribution defined by the :class:`ContinuousValue` of the attribute.

        :param attr: an attribute of an instance
        :type attr: str
        :return: A value sampled from the distribution of values in the node's
            probability table.
        :rtype: str or float

        .. seealso :meth:`Cobweb3Node.predict`
        """
        if attr not in self.tree.root.av_counts:
            return None

        if isinstance(self.tree.root.av_counts[attr], ContinuousValue):
            n_values = 2
            prob_attr = ((1.0 * self.av_counts[attr].num + self.tree.alpha) /
                         (self.count + self.tree.alpha * n_values ))

            if prob_attr < 0.5:
                return None

            return normalvariate(self.av_counts[attr].mean,
                                 self.av_counts[attr].unbiased_std())
        else:
            return super(Cobweb3Node, self).sample(attr)

    def predict(self, attr):
        """
        Predict the value of an attribute, by returning the most likely value.
        This takes into account the laplacian smoothing.

        If the attribute is a nominal then this function behaves the same as
        :meth:`CobwebNode.predict <concept_formation.cobweb.CobwebNode.predict>`.
        If the attribute is numeric then the mean value from the
        :class:`ContinuousValue` is chosen.

        :param attr: an attribute of an instance.
        :type attr: str
        :return: The most likely value for the given attribute in the node's 
            probability table.
        :rtype: str or float

        .. seealso :meth:`Cobweb3Node.sample`
        """
        if attr not in self.tree.root.av_counts:
            return None

        if isinstance(self.tree.root.av_counts[attr], ContinuousValue):
            n_values = 2
            prob_attr = ((1.0 * self.av_counts[attr].num + self.tree.alpha) /
                         (self.count + self.tree.alpha * n_values ))

            if prob_attr < 0.5:
                return None

            return self.av_counts[attr].mean
        else:
            return super(Cobweb3Node, self).predict(attr)

    def get_probability(self, attr, val):
        """
        Returns the probability of a particular attribute value at the current
        concept. 

        This takes into account the possibilities that an attribute can take any
        of the values available at the root, or be missing. Laplace smoothing is
        used to place a prior over these possibilites. Alpha determines the
        strength of this prior.

        For numerical attributes the probability of val given a
        normal distribution is returned. This normal distribution is defined by
        the mean and std of past values stored in the concept.
        
        :param attr: an attribute of an instance
        :type attr: str
        :param val: a value for the given attribute
        :type val: str:
        :return: The probability of attr having the value val in the current concept.
        :rtype: float
        """
        if attr not in self.tree.root.av_counts:
            return 0.0

        if isinstance(self.tree.root.av_counts[attr], ContinuousValue):
            n_values = 2
            prob_attr = ((1.0 * self.av_counts[attr].num + self.tree.alpha) /
                         (self.count + self.tree.alpha * n_values ))

            if val is None:
                return 1 - prob_attr

            if self.tree.scaling and self.parent:
                scale = self.tree.std_to_scale * self.parent.av_counts[attr].unbiased_std()
                if scale == 0:
                    scale = 1
                shift = self.parent.av_counts[attr].mean
                val = (val - shift) / scale
            else:
                scale = 1.0
                shift = 0.0

            mean = (self.av_counts[attr].mean - shift) / scale
            std = max(self.av_counts[attr].scaled_unbiased_std(scale),
                      self.acuity)

            return (prob_attr * 
                    (1.0 / (std * sqrt(2 * pi))) * 
                    exp(-((val - mean) * (val - mean)) / (2.0 * std * std)))

        else:
            return super(Cobweb3Node, self).get_probability(attr, val)

    def output_json(self):
        """
        Outputs the categorization tree in JSON form. 

        This is a modification of the :meth:`CobwebNode.output_json
        <concept_formation.cobweb.CobwebNode.output_json>` to handle numeric
        values.

        :return: an object that contains all of the structural information of
            the node and its children
        :rtype: obj
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
    This class is used to store the number of samples, the mean of the samples,
    and the squared error of the samples for numeric attribute values. It can be
    used to perform incremental estimation of the attribute's mean, std, and
    unbiased std.

    Initially the number of values, the mean of the values, and the
    squared errors of the values are set to 0.
    """

    def __init__(self):
        """constructor"""
        self.num = 0.0
        self.mean = 0.0
        self.meanSq = 0.0

    def __len__(self):
        return 1

    def copy(self):
        """
        Returns a deep copy of itself.

        :return: a deep copy of the continuous value
        :rtype: ContinuousValue
        """
        v = ContinuousValue()
        v.num = self.num
        v.mean = self.mean
        v.meanSq = self.meanSq
        return v

    def unbiased_mean(self):
        """
        Returns the mean value.

        :return: the unbiased mean
        :rtype: float
        """
        return self.mean

    def scaled_unbiased_mean(self, shift, scale):
        """
        Returns (self.mean - shift) / scale

        This is used as part of numerical value scaling.

        :param shift: the amount to shift the mean by
        :type shift: float
        :param scale: the amount to scale the returned mean by
        :type scale: float
        :return: ``(self.mean - shift) / scale``
        :rtype: float
        """
        if scale <= 0:
            scale = 1
        return (self.mean - shift) / scale

    def biased_std(self):
        """
        Returns a biased estimate of the std (i.e., the sample std)

        :return: biased estimate of the std (i.e., the sample std)
        :rtype: float
        """
        return sqrt(self.meanSq / (self.num))

    def unbiased_std(self):
        """Returns an unbiased estimate of the std 

        This implementation uses Bessel's correction and Cochran's theorem: 
        `<https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation#Bias_correction>`_

        :return: an unbiased estimate of the std
        :rtype: float

        .. seealso:: :meth:`concept_formation.utils.c4`
        """
        if self.num < 2:
            return 0.0
        return sqrt(self.meanSq / (self.num - 1)) / c4(self.num)

    def scaled_unbiased_std(self, scale):
        """
        Returns an unbiased estimate of the std (see:
        :meth:`ContinuousValue.unbiased_std`), but also adjusts the std given a
        scale parameter.

        This is used to return std values that have been normalized by some
        value. For edge cases, if scale is less than or equal to 0, then scaling
        is disabled (i.e., scale = 1.0).

        :param scale: an amount to scale unbiased std estimates by
        :type scale: float
        :return: A scaled unbiased estimate of std
        :rtype: float
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
        Calls the update function on every value in a given dataset

        :param data: A list of numberic values to add to the distribution
        :type data: [Number, Number, ...]
        """
        for x in data:
            self.update(x)

    def update(self, x):
        """
        Incrementally update the mean and squared mean error (meanSq) values in
        an efficient and practical (no precision problems) way. 

        This uses and algorithm by Knuth found here:
        `<https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_

        :param x: A new value to incorporate into the distribution
        :type x: Number
        """
        self.num += 1
        delta = x - self.mean 
        self.mean += delta / self.num
        self.meanSq += delta * (x - self.mean)

    def combine(self, other):
        """
        Combine another ContinuousValue's distribution into this one in
        an efficient and practical (no precision problems) way. 

        This uses the parallel algorithm by Chan et al. found at:
        `<https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm>`_

        :param other: Another ContinuousValue distribution to be incorporated
            into this one.
        :type other: ContinuousValue
        """
        if not isinstance(other, ContinuousValue):
            raise ValueError("Can only merge 2 continuous values.")
        delta = other.mean - self.mean
        self.meanSq = (self.meanSq + other.meanSq + delta * delta * 
                       ((self.num * other.num) / (self.num + other.num)))
        self.mean = ((self.num * self.mean + other.num * other.mean) / 
                     (self.num + other.num))
        self.num += other.num

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from random import normalvariate
from random import random
from numbers import Number
from math import sqrt
from math import pi
from math import exp

from concept_formation.utils import weighted_choice
from concept_formation.utils import most_likely_choice
from concept_formation.cobweb import CobwebNode
from concept_formation.cobweb import CobwebTree
from concept_formation.continuous_value import ContinuousValue

class Cobweb3Tree(CobwebTree):
    """
    The Cobweb3Tree contains the knoweldge base of a partiucluar instance of the
    Cobweb/3 algorithm and can be used to fit and categorize instances.
    Cobweb/3's main difference over Cobweb is the ability to handle numerical
    attributes by applying an assumption that they should follow a normal
    distribution. For the purposes of Cobweb/3's core algorithms a numeric
    attribute is any value where ``isinstance(instance[attr], Number)`` returns
    ``True``.

    The scaling parameter determines whether online normalization of
    continuous attributes is used. By default scaling is used. Scaling
    divides the std of each attribute by the std of the attribute in the
    parent node (no scaling is performed in the root). Scaling is useful to
    balance the weight of different numerical attributes, without scaling
    the magnitude of numerical attributes can affect category utility
    calculation meaning numbers that are naturally larger will recieve
    extra weight in the calculation.

    Acuity is used as a floor on standard deviation estimates for numeric
    attribute values. The default value is set to 
    :math:`\\frac{1}{\\sqrt{2 * \\pi}}` which is the smallest possible acuity
    before probability estimates begin to exceed 1.0.

    :param scaling: whether or not numerical values should be scaled in
        online normalization.
    :type scaling: "root", "parent", or None
    :param acuity: A lower bound on the standard deviation estimates for
        numeric attributes.
    :type acuity: float
    """

    def __init__(self, scaling=None, acuity=1.0/sqrt(2.0*pi)):
        """The tree constructor."""
        self.root = Cobweb3Node()
        self.root.tree = self
        self.scaling = scaling
        self.acuity = acuity

        # Number of stds to divide by when normalizing (2.0 -> everything
        # normalized to half a std deviation)
        self.std_to_scale = 1.0 

    def infer_missing(self, instance, choice_fn="most likely", allow_none=True):
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

            if isinstance(concept.av_counts[attr], ContinuousValue):
                if choice_fn == most_likely_choice:
                    val = concept.av_counts[attr].unbiased_mean()
                else:
                    val = normalvariate(concept.av_counts[attr].unbiased_mean(),
                                        concept.av_counts[attr].unbiased_std())
                if not allow_none:
                    temp_instance[attr] = val
                else:
                    missing_prob = concept.get_probability_missing(attr)
                    val_choices = ((None, missing_prob), (val, 1 - missing_prob))
                    temp_instance[attr] = choice_fn(val_choices)

            else:
                val_choices = concept.get_weighted_values(attr)
                if not allow_none:
                    val_choices = [(choice, prob) for choice,prob in val_choices if
                                  choice is not None]

                val = choice_fn(val_choices)
                if val is not None:
                    temp_instance[attr] = val

        probs = {attr: concept.get_probability(attr, temp_instance[attr]) for
                 attr in temp_instance}
        return temp_instance, probs

    def clear(self):
        """
        Clears the concepts of the tree, but maintains the scaling parameter.
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
            if not isinstance(instance[attr], bool) and isinstance(instance[attr], Number):
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
            # TODO check that this should be 0
            return 0.0
        elif isinstance(self.av_counts[attr], ContinuousValue):
            if self.tree.scaling == "parent" and self.parent:
                scale = self.tree.std_to_scale * self.parent.av_counts[attr].unbiased_std()
            elif self.tree.scaling == "root":
                scale = self.tree.std_to_scale * self.tree.root.av_counts[attr].unbiased_std()
            else:
                scale = 1.0

            before_std = max(self.av_counts[attr].scaled_unbiased_std(scale),
                             self.tree.acuity)
            before_prob = ((1.0 * self.av_counts[attr].num) / (self.count + 1.0))
            before_count = ((before_prob * before_prob) * 
                            (1.0 / (2.0 * sqrt(pi) * before_std)))

            temp = self.av_counts[attr].copy()
            temp.update(val)
            after_std = max(temp.scaled_unbiased_std(scale), self.tree.acuity)
            after_prob = ((1.0 + self.av_counts[attr].num) / (self.count + 1.0))
            after_count = ((after_prob * after_prob) * 
                            (1.0 / (2.0 * sqrt(pi) * after_std)))
            return after_count - before_count
        elif val not in self.av_counts[attr]:
            # TODO check that this should be 0
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

            if attr not in self.av_counts:
                val_count = 0
                
            elif isinstance(self.tree.root.av_counts[attr], ContinuousValue):
                if self.tree.scaling == "parent" and self.parent:
                    scale = (self.tree.std_to_scale *
                             self.parent.av_counts[attr].unbiased_std())
                elif self.tree.scaling == "root":
                    scale = (self.tree.std_to_scale *
                             self.tree.root.av_counts[attr].unbiased_std())
                else:
                    scale = 1.0

                std = max(self.av_counts[attr].scaled_unbiased_std(scale),
                          self.tree.acuity)
                prob_attr = self.av_counts[attr].num / self.count
                correct_guesses += ((prob_attr * prob_attr) * 
                                    (1.0 / (2.0 * sqrt(pi) * std)))
                val_count = self.av_counts[attr].num

            else:
                val_count = 0
                for val in self.tree.root.av_counts[attr]:
                    if val not in self.av_counts[attr]:
                        prob = 0
                    else:
                        val_count += self.av_counts[attr][val]
                        prob = (self.av_counts[attr][val]) / (1.0 * self.count)
                    correct_guesses += (prob * prob)

            #Factors in the probability mass of missing values
            prob = (self.count - val_count) / self.count
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
                attributes.append("'%s': %s}" % (attr, str(self.av_counts[attr])))
                #attributes.append("'%s': { %0.3f (%0.3f) [%i] }" % (attr,
                #                                                    self.av_counts[attr].mean,
                #                                                    max(self.tree.acuity,
                #                                                        self.av_counts[attr].unbiased_std()),
                #                                                    self.av_counts[attr].num))
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
        """
        Samples the value of an attribute from the node's probability table.

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
            prob_attr = self.av_counts[attr].num / self.count

            if prob_attr < random():
                return None

            return normalvariate(self.av_counts[attr].mean,
                                 self.av_counts[attr].unbiased_std())
        else:
            return super(Cobweb3Node, self).sample(attr)

    def predict(self, attr):
        """
        Predict the value of an attribute, by returning the most likely value.

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
            # get the right concept for this attribute using past performance
            best = self
            curr = self
            while curr is not None:
                if (curr.correct_at_node[attr].unbiased_mean() >=
                    curr.correct_at_decendents[attr].unbiased_mean()):
                    best = curr
                curr = curr.parent

            prob_attr = best.av_counts[attr].num / best.count

            if prob_attr < 0.5:
                return None

            return best.av_counts[attr].mean
        else:
            return super(Cobweb3Node, self).predict(attr)

    def get_probability(self, attr, val):
        """
        Returns the probability of a particular attribute value at the current
        concept. 

        This takes into account the possibilities that an attribute can take any
        of the values available at the root, or be missing. 

        For numerical attributes the probability of val given a normal
        distribution is returned. This normal distribution is defined by the
        mean and std of past values stored in the concept.
        
        :param attr: an attribute of an instance
        :type attr: str
        :param val: a value for the given attribute
        :type val: str:
        :return: The probability of attr having the value val in the current concept.
        :rtype: float
        """
        if (attr in self.tree.root.av_counts and
            isinstance(self.tree.root.av_counts[attr], ContinuousValue)):

            prob_attr = 0.0
            if attr in self.av_counts:
                prob_attr = self.av_counts[attr].num / self.count

            if val is None:
                return 1.0 - prob_attr

            if self.tree.scaling == "parent" and self.parent:
                scale = self.tree.std_to_scale * self.parent.av_counts[attr].unbiased_std()
                if scale == 0:
                    scale = 1
                shift = self.parent.av_counts[attr].mean
                val = (val - shift) / scale
            elif self.tree.scaling == "root":
                scale = self.tree.std_to_scale * self.tree.root.av_counts[attr].unbiased_std()
                if scale == 0:
                    scale = 1
                shift = self.tree.root.av_counts[attr].mean
                val = (val - shift) / scale
            else:
                scale = 1.0
                shift = 0.0

            mean = (self.av_counts[attr].mean - shift) / scale
            std = max(self.av_counts[attr].scaled_unbiased_std(scale),
                      self.tree.acuity)
            return (prob_attr * 
                    (1.0 / (std * sqrt(2 * pi))) * 
                    exp(-((val - mean) * (val - mean)) / (2.0 * std * std)))

        else:
            return super(Cobweb3Node, self).get_probability(attr, val)

    def get_probability_missing(self, attr):
        """
        Returns the probability of a particular attribute not being present in a
        given concept.

        This takes into account the possibilities that an attribute can take any
        of the values available at the root, or be missing. 

        :param attr: an attribute of an instance
        :type attr: str
        :return: The probability of attr not being present from an instance in
            the current concept.
        :rtype: float 
        """
        return self.get_probability(attr, None)

    def is_pure_match(self, instance):
        """
        Returns true if the concept exactly matches the instance.

        :param instance: The instance currently being categorized
        :type instance: {a1: v1, a2: v2, ...} - a hashtable of attr and values
        :return: whether the instance perfectly matches the concept
        :rtype: boolean

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        for attr in self.tree.root.av_counts:
            if attr in instance and attr not in self.av_counts:
                return False
            if attr in self.av_counts and attr not in instance:
                return False
            if attr in self.av_counts and attr in instance:
                if isinstance(self.av_counts[attr], ContinuousValue):
                    if (not self.av_counts[attr].unbiased_std() == 0.0):
                        return False
                    if (not self.av_counts[attr].unbiased_mean() ==
                        instance[attr]):
                        return False
                elif not instance[attr] in self.av_counts[attr]:
                    return False
                elif not self.av_counts[attr][instance[attr]] == self.count:
                    return False
        return True

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

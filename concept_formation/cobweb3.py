"""
The Cobweb3 module contains the :class:`Cobweb3Tree` and :class:`Cobweb3Node`
classes, which extend the traditional Cobweb capabilities to support numeric
values on attributes. 
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from random import normalvariate
from random import random
from math import sqrt
from math import pi
from math import exp
from math import log

from concept_formation.cobweb import CobwebNode
from concept_formation.cobweb import CobwebTree
from concept_formation.continuous_value import ContinuousValue
from concept_formation.utils import isNumber
from concept_formation.utils import weighted_choice
from concept_formation.utils import most_likely_choice

cv_key = "#ContinuousValue#"

class Cobweb3Tree(CobwebTree):
    """
    The Cobweb3Tree contains the knowledge base of a partiucluar instance of the
    Cobweb/3 algorithm and can be used to fit and categorize instances.
    Cobweb/3's main difference over Cobweb is the ability to handle numerical
    attributes by applying an assumption that they should follow a normal
    distribution. For the purposes of Cobweb/3's core algorithms a numeric
    attribute is any value where ``isinstance(instance[attr], Number)`` returns
    ``True``.

    The scaling parameter determines whether online normalization of continuous
    attributes is used, and to what standard deviation the values are scaled
    to. Scaling divides the std of each attribute by the std of the attribute
    in the root divided by the scaling constant (i.e., 
    :math:`\\sigma_{root} / scaling` when making category utility calculations.
    Scaling is useful to balance the weight of different numerical attributes,
    without scaling the magnitude of numerical attributes can affect category
    utility calculation meaning numbers that are naturally larger will recieve
    preference in the category utility calculation.

    :param scaling: What number of standard deviations numeric attributes
        should be scaled to. By default this value is 0.5 (half a standard
        deviation), which is the max std of nominal values. If disabiling
        scaling is desirable, then it can be set to False or None.
    :type scaling: a float greater than 0.0, None, or False
    :param inner_attr_scaling: Whether to use the inner most attribute name
        when scaling numeric attributes. For example, if `('attr', '?o1')` was
        an attribute, then the inner most attribute would be 'attr'. When using
        inner most attributes, some objects might have multiple attributes
        (i.e., 'attr' for different objects) that contribute to the scaling. 
    :param inner_attr_scaling: boolean
    """

    def __init__(self, scaling=0.5, inner_attr_scaling=True):
        """
        The tree constructor.
        """
        self.root = Cobweb3Node()
        self.root.tree = self
        self.scaling = scaling
        self.inner_attr_scaling = inner_attr_scaling 
        self.attr_scales = {}

    def clear(self):
        """
        Clears the concepts of the tree, but maintains the scaling parameter.
        """
        self.root = Cobweb3Node()
        self.root.tree = self
        self.attr_scales = {}

    def get_inner_attr(self, attr):
        """
        Extracts the inner most attribute name from the provided attribute, if
        the attribute is a tuple and inner_attr_scaling is on. Otherwise it
        just returns the attribute. This is used to for normalizing attributes. 

        >>> t = Cobweb3Tree()
        >>> t.get_inner_attr(('a', '?object1'))
        'a'
        >>> t.get_inner_attr('a')
        'a'
        """
        if isinstance(attr, tuple) and self.inner_attr_scaling:
            return attr[0]
        else:
            return attr

    def update_scales(self, instance):
        """
        Reads through all the attributes in an instance and updates the
        tree scales object so that the attributes can be properly scaled.
        """
        for attr in instance:
            if isNumber(instance[attr]):
                inner_attr = self.get_inner_attr(attr)
                if inner_attr not in self.attr_scales:
                    self.attr_scales[inner_attr] = ContinuousValue()
                self.attr_scales[inner_attr].update(instance[attr])

    def cobweb(self, instance):
        """
        A modification of the cobweb function to update the scales object
        first, so that attribute values can be properly scaled. 
        """
        self.update_scales(instance)
        return super(Cobweb3Tree, self).cobweb(instance)

    def ifit(self, instance):
        """
        Incrementally fit a new instance into the tree and return its resulting
        concept.

        The cobweb3 version of the :meth:`CobwebTree.ifit` function. This
        version keeps track of all of the continuous 

        :param instance: An instance to be categorized into the tree.
        :type instance:  :ref:`Instance<instance-rep>`
        :return: A concept describing the instance
        :rtype: Cobweb3Node

        .. seealso:: :meth:`CobwebTree.cobweb`
        """
        self._sanity_check_instance(instance) 
        return self.cobweb(instance)

class Cobweb3Node(CobwebNode):
    """
    A Cobweb3Node represents a concept within the knoweldge base of a
    particular :class:`Cobweb3Tree`. Each node contians a probability table
    that can be used to calculate the probability of different attributes given
    the concept that the node represents.

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
            name of a previously nominal attribute, or vice versa, this
            function will raise an exception. See: :class:`NumericToNominal
            <concept_formation.preprocessor.NumericToNominal>` for a way to fix
            this error.
        
        :param instance: A new instances to incorporate into the node.
        :type instance: :ref:`Instance<instance-rep>`

        """
        self.count += 1 
            
        for attr in instance:
            self.av_counts[attr] = self.av_counts.setdefault(attr,{})

            if isNumber(instance[attr]):
                if cv_key not in self.av_counts[attr]:
                    self.av_counts[attr][cv_key] = ContinuousValue()
                self.av_counts[attr][cv_key].update(instance[attr])
            else:
                prior_count = self.av_counts[attr].get(instance[attr], 0)
                self.av_counts[attr][instance[attr]] = prior_count + 1

    def update_counts_from_node(self, node):
        """
        Increments the counts of the current node by the amount in the specified
        node, modified to handle numbers.

        .. warning:: If a numeric attribute is found in an instance with the
            name of a previously nominal attribute, or vice versa, this
            function will raise an exception. See: :class:`NumericToNominal
            <concept_formation.preprocessor.NumericToNominal>` for a way to fix
            this error.

        :param node: Another node from the same Cobweb3Tree
        :type node: Cobweb3Node
        """
        self.count += node.count
        for attr in node.av_counts:
            self.av_counts[attr] = self.av_counts.setdefault(attr, {})
            for val in node.av_counts[attr]:
                if val == cv_key:
                    self.av_counts[attr][val] = self.av_counts[attr].get(val, ContinuousValue())
                    self.av_counts[attr][val].combine(node.av_counts[attr][val])
                else:
                    self.av_counts[attr][val] = (self.av_counts[attr].get(val,0) +
                                         node.av_counts[attr][val])
    
    def attr_val_guess_gain(self, attr, val, counts=1.0):
        """
        Returns the gain in number of correct guesses if a particular attr/val
        was added to a concept "counts" times. Effectively, it compares the
        expected correct guesses of the attribute if the overall counts is
        increased by one (i.e., a none is added) vs. if a particular value of
        the attribute is added. 

        :param attr: An attribute in the concept
        :type attr: :ref:`Attribute<attributes>`
        :param val: A value for the given attribute in the concept
        :type val: :ref:`Value<values>`
        :param counts: the number of times to add the attr val to the concept
        :type counts: integer or float
        :return: the gain in number of correct guesses from adding the
                 partiucluar attr/val
        :rtype: float               
        """
        if attr[0] == "_":
            return 0.0

        if attr not in self.av_counts:
            return 0.0

        if ((isNumber(val) or isinstance(val, ContinuousValue)) and
            cv_key not in self.av_counts[attr]):
            return 0.0

        if isNumber(val) or isinstance(val, ContinuousValue):
            if self.tree.scaling:
                inner_attr = self.tree.get_inner_attr(attr)
                scale = ((1/self.tree.scaling) *
                         self.tree.attr_scales[inner_attr].unbiased_std())
            else:
                scale = 1.0

            value = self.av_counts[attr][cv_key]

            before_std = sqrt(value.scaled_unbiased_std(scale) *
                              value.scaled_unbiased_std(scale) +
                             1/(4*pi))
            before_prob = ((1.0 * value.num) / 
                           (self.count + counts))
            before_count = ((before_prob * before_prob) * 
                            (1/(2 * sqrt(pi) * before_std)))

            temp = value.copy()
            if isinstance(val, ContinuousValue):
                temp.combine(val)
            else:
                temp.update(val)

            after_std = sqrt(temp.scaled_unbiased_std(scale) *
                             temp.scaled_unbiased_std(scale) + 1/(4*pi)) 
            after_prob = ((value.num + counts) / 
                          (self.count + counts))
            after_count = ((after_prob * after_prob) * 
                           (1/(2 * sqrt(pi) * after_std)))
            return after_count - before_count

        if val not in self.av_counts[attr]:
            return 0.0

        before_prob = (self.av_counts[attr][val] / (self.count + counts))
        after_prob = ((self.av_counts[attr][val] + counts) / 
                      (self.count + counts))
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
        :math:`P(A_i) < 1.0`. Additionally, the original formulation set
        :math:`\\sigma` to have a user specified minimum value. However, for
        small lower bounds, this lets cobweb achieve more than 1 expected
        correct guess per attribute, which is impossible for nominal attributes
        (and does not really make sense for continuous either). This causes
        problems when both nominal and continuous values are being used
        together; i.e., continuous attributes will get higher preference. 

        To account for this we use a modified equation:

        .. math::

            P(A_i = V_{ij})^2 = P(A_i)^2 * \\frac{1}{2 * \\sqrt{\\pi} * \\sigma}

        The key change here is that we multiply by :math:`P(A_i)^2`. 
        Further, instead of bounding :math:`\\sigma` by a user specified lower
        bound (often called acuity), we add some independent, normally
        distributed noise to sigma: 
        :math:`\\sigma = \\sqrt{\\sigma^2 + \\sigma_{noise}^2}`, where 
        :math:`\\sigma_{noise} = \\frac{1}{2 * \\sqrt{\\pi}}`. 
        This ensures the expected correct guesses never exceeds 1. From a
        theoretical point of view, it basically is an assumption that there is
        some independent, normally distributed measurement error that is added
        to the estimated error of the attribute (`<https://en.wikipedia.org/wi
        ki/Sum_of_normally_distributed_random_variables>`_). It is possible
        that there is additional measurement error, but the value is chosen so
        as to yield a sensical upper bound on the expected correct guesses.

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

            else:
                val_count = 0
                for val in self.av_counts[attr]:
                    if val == cv_key:
                        if self.tree.scaling:
                            inner_attr = self.tree.get_inner_attr(attr)
                            scale = ((1/self.tree.scaling) *
                                     self.tree.attr_scales[inner_attr].unbiased_std())
                        else:
                            scale = 1.0

                        # we basically add noise to the std and adjust the
                        # normalizing constant to ensure the probability of a
                        # particular value never exceeds 1.
                        cv = self.av_counts[attr][cv_key]
                        std = sqrt(cv.scaled_unbiased_std(scale) *
                                   cv.scaled_unbiased_std(scale) +
                                   (1 / (4 * pi)))
                        prob_attr = cv.num / self.count
                        correct_guesses += ((prob_attr * prob_attr) * 
                                            (1/(2 * sqrt(pi) * std)))
                        val_count += cv.num
                    else:
                        prob = (self.av_counts[attr][val]) / (1.0 * self.count)
                        correct_guesses += (prob * prob)
                        val_count += self.av_counts[attr][val]

            #Factors in the probability mass of missing values
            prob = (self.count - val_count) / self.count
            correct_guesses += (prob * prob)

        return correct_guesses / len(self.tree.root.av_counts)

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
            values = []
            for val in self.av_counts[attr]:
                values.append("'" + str(val) + "': " +
                              str(self.av_counts[attr][val]))

            attributes.append("'" + str(attr) + "': {" + ", ".join(values)
                              + "}")
                  
        ret += "{" + ", ".join(attributes) + "}: " + str(self.count) + '\n'
        
        for c in self.children:
            ret += c.pretty_print(depth+1)

        return ret

    def get_weighted_values(self, attr, allow_none=True):
        """
        Return a list of weighted choices for an attribute based on the node's
        probability table.

        This calculation will include an option for the change that an attribute
        is missing from an instance all together. This is useful for probability
        and sampling calculations. If the attribute has never appeared in the
        tree then it will return a 100% chance of None.

        :param attr: an attribute of an instance
        :type attr: :ref:`Attribute<attributes>`
        :param allow_none: whether attributes in the nodes probability table
            can be inferred to be missing. If False, then None will not be
            cosidered as a possible value.
        :type allow_none: Boolean
        :return: a list of weighted choices for attr's value
        :rtype: [(:ref:`Value<values>`, float), (:ref:`Value<values>`, float), ...]
        """
        choices = []
        if attr not in self.tree.root.av_counts:
            choices.append((None, 1.0))
            return choices

        val_count = 0
        for val in self.tree.root.av_counts[attr]:
            count = 0
            if attr in self.av_counts and val in self.av_counts[attr]:
                if val == cv_key:
                    count = self.av_counts[attr][val].num
                else:
                    count = self.av_counts[attr][val]
            choices.append((val, count / self.count))
            val_count += count

        if allow_none:
            choices.append((None, ((self.count - val_count) / self.count)))
        return choices

    def predict(self, attr, choice_fn="most likely", allow_none=True):
        """
        Predict the value of an attribute, using the provided strategy.

        If the attribute is a nominal then this function behaves the same as
        :meth:`CobwebNode.predict <concept_formation.cobweb.CobwebNode.predict>`.
        If the attribute is numeric then the mean value from the
        :class:`ContinuousValue<concept_formation.cv_key.ContinuousValue>` is chosen.

        :param attr: an attribute of an instance.
        :type attr: :ref:`Attribute<attributes>`
        :param allow_none: whether attributes not in the instance can be
            inferred to be missing. If False, then all attributes will be
            inferred with some value.
        :type allow_none: Boolean
        :return: The most likely value for the given attribute in the node's 
            probability table.
        :rtype: :ref:`Value<values>`

        .. seealso :meth:`Cobweb3Node.sample`
        """
        if choice_fn == "most likely" or choice_fn == "m":
            choose = most_likely_choice
        elif choice_fn == "sampled" or choice_fn == "s":
            choose = weighted_choice
        else:
            raise Exception("Unknown choice_fn")

        if attr not in self.tree.root.av_counts:
            return None

        choices = self.get_weighted_values(attr, allow_none)
        val = choose(choices)

        if val == cv_key:
            if choice_fn == "most likely" or choice_fn == "m":
                val = self.av_counts[attr][val].mean
            elif choice_fn == "sampled" or choice_fn == "s":
                val = normalvariate(self.av_counts[attr][val].unbiased_mean(),
                                    self.av_counts[attr][val].unbiased_std())
            else:
                raise Exception("Unknown choice_fn")

        return val

    def probability(self, attr, val):
        """
        Returns the probability of a particular attribute value at the current
        concept. 

        This takes into account the possibilities that an attribute can take any
        of the values available at the root, or be missing. 

        For numerical attributes the probability of val given a gaussian 
        distribution is returned. This distribution is defined by the
        mean and std of past values stored in the concept. However like
        :meth:`Cobweb3Node.expected_correct_guesses
        <concept_formation.cobweb3.Cobweb3Node.expected_correct_guesses>` it
        adds :math:`\\frac{1}{2 * \\sqrt{\\pi}}` to the estimated std (i.e,
        assumes some independent, normally distributed noise).
        
        :param attr: an attribute of an instance
        :type attr: :ref:`Attribute<attributes>`
        :param val: a value for the given attribute
        :type val: :ref:`Value<values>`
        :return: The probability of attr having the value val in the current concept.
        :rtype: float
        """
        if val is None:
            c = 0.0
            if attr in self.av_counts:
                c = sum([self.av_counts[attr][v].num if v == cv_key
                         else self.av_counts[attr][v] for v in
                         self.av_counts[attr]])
            return (self.count - c) / self.count

        if isNumber(val):
            if cv_key not in self.av_counts[attr]:
                return 0.0

            prob_attr = self.av_counts[attr][cv_key].num / self.count
            if self.tree.scaling:
                inner_attr = self.tree.get_inner_attr(attr)
                scale = ((1/self.tree.scaling) *
                         self.tree.attr_scales[inner_attr].unbiased_std())

                if scale == 0:
                    scale = 1
                shift = self.tree.attr_scales[inner_attr].mean
                val = (val - shift) / scale
            else:
                scale = 1.0
                shift = 0.0

            mean = (self.av_counts[attr][cv_key].mean - shift) / scale
            std = sqrt(self.av_counts[attr][cv_key].scaled_unbiased_std(scale) *
                       self.av_counts[attr][cv_key].scaled_unbiased_std(scale) + 
                       (1 / (4 * pi)))
            p = (prob_attr * 
                 (1/(sqrt(2*pi) * std)) * 
                 exp(-((val - mean) * (val - mean)) / (2.0 * std * std)))
            return p

        if attr in self.av_counts and val in self.av_counts[attr]:
            return self.av_counts[attr][val] / self.count

        return 0.0

    def log_likelihood(self, other):
        """
        Returns the log-likelihood of the concept.
        """

        ll = 0
        for attr in self.tree.root.av_counts:
            if attr[0] == '_':
                continue

            for val in list(self.tree.root.av_counts[attr]) + [None]:
                if val == cv_key:
                    if (attr in self.av_counts and cv_key in self.av_counts[attr] and 
                        attr in other.av_counts and cv_key in other.av_counts[attr]):
                        p = (self.probability(attr, other.av_counts[attr][cv_key].unbiased_mean()) * 
                             other.probability(attr, other.av_counts[attr][cv_key].unbiased_mean()))
                        if p > 0:
                            ll += log(p)
                else:
                    op = other.probability(attr, val)
                    if op > 0:
                        p = self.probability(attr,val) * op
                        if p > 0:
                            ll += log(p)
                        else:
                            raise Exception("p must be greater than 0")
        return ll

    def is_exact_match(self, instance):
        """
        Returns true if the concept exactly matches the instance.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
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
                if (isNumber(instance[attr]) and 
                    cv_key not in self.av_counts[attr]):
                    return False
                if (isNumber(instance[attr]) and cv_key in
                    self.av_counts[attr]):
                    if (len(self.av_counts[attr]) != 1 or 
                        self.av_counts[attr][cv_key].num != self.count):
                        return False
                    if (not self.av_counts[attr][cv_key].unbiased_std() == 0.0):
                        return False
                    if (not self.av_counts[attr][cv_key].unbiased_mean() ==
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
                temp[str(attr)] = {str(value):self.av_counts[attr][value] for
                                   value in self.av_counts[attr]}

        for child in self.children:
               output["children"].append(child.output_json())

        output["counts"] = temp

        return output


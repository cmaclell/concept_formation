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
from math import sqrt
from math import pi
from math import exp
from math import log
from collections import Counter
from itertools import chain

from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.cobweb3 import Cobweb3Tree
from concept_formation.cobweb3 import cv_key
from concept_formation.continuous_value import ContinuousValue
from concept_formation.context_instance import ContextInstance
from concept_formation.utils import isNumber
from concept_formation.utils import weighted_choice
from concept_formation.utils import most_likely_choice

ca_key = "#ContAttribute#"


class ContextualCobwebTree(Cobweb3Tree):
    """
    """

    def __init__(self, ctxt_scaling=1, scaling=0.5, inner_attr_scaling=True):
        """
        The tree constructor.

        :param ctxt_scaling: factor by which the context should be scaled
            when combining category utility with other attribute types
        :type ctxt_scaling: float
        """
        self.root = ContextualCobwebNode()
        self.root.tree = self
        self.context_scaling = ctxt_scaling
        self.scaling = scaling
        self.inner_attr_scaling = inner_attr_scaling
        self.attr_scales = {}

    def cobweb(self, instance):
        raise NotImplementedError

    def contextual_ifit(self, instances, context_key=lambda x: x):
        """
        Adds multiple instances, creating the correct context attributes for
        each of them.

        :param instances: instances to be added
        :type instances: Sequence<Instance>
        """
        raise NotImplementedError
        # context must be {ca_key: set<ContextInstance>}


class ContextualCobwebNode(Cobweb3Node):

    def __init__(self, other_node=None):
        self.descendants = set()
        super().__init__(other_node)

    def increment_counts(self, instance):
        """
        Increment the counts at the current node according to the specified
        instance.

        ContextualCobwebNode uses a modified version of
        :meth:`Cobweb3Node.increment_counts
        <concept_formation.cobweb3.Cobweb3Node.increment_counts>` that handles
        contextual attributes properly. The attribute equalling ca_key will be
        treated as context.

        :param instance: A new instances to incorporate into the node.
        :type instance: :ref:`Instance<instance-rep>`

        """
        self.count += 1

        for attr in instance:
            if attr == ca_key:
                self.av_counts.setdefault(attr, Counter())
                self.av_counts[attr].update(instance[attr])
                continue

            self.av_counts.setdefault(attr, {})

            if isNumber(instance[attr]):
                if cv_key not in self.av_counts[attr]:
                    self.av_counts[attr][cv_key] = ContinuousValue()
                self.av_counts[attr][cv_key].update(instance[attr])
            else:
                prior_count = self.av_counts[attr].get(instance[attr], 0)
                self.av_counts[attr][instance[attr]] = prior_count + 1

    def update_counts_from_node(self, node):
        """
        Increments the counts of the current node by the amount in the
        specified node, modified to handle context.

        :param node: Another node from the same Cobweb3Tree
        :type node: Cobweb3Node
        """
        self.count += node.count
        for attr in node.attrs('all'):
            if attr == ca_key:
                self.av_counts.setdefault(attr, Counter())
                self.av_counts[attr].update(node[attr])
                continue

            self.av_counts.setdefault(attr, {})

            for val in node.av_counts[attr]:
                if val == cv_key:
                    self.av_counts[attr][val] = self.av_counts[attr].get(
                        val, ContinuousValue())
                    self.av_counts[attr][val].combine(
                        node.av_counts[attr][val])
                else:
                    self.av_counts[attr][val] = (self.av_counts[attr].get(val,
                                                                          0) +
                                                 node.av_counts[attr][val])

    def expected_correct_guesses(self):
        raise NotImplementedError

    def pretty_print(self, depth=0):
        raise NotImplementedError

    def get_weighted_values(self, attr, allow_none=True):
        raise NotImplementedError

    def predict(self, attr, choice_fn="most likely", allow_none=True):
        raise NotImplementedError

    def probability(self, attr, val):
        raise NotImplementedError

    def log_likelihood(self, child_leaf):
        raise NotImplementedError

    def is_exact_match(self, instance):
        """
        Returns true if the concept exactly matches the instance.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: whether the instance perfectly matches the concept
        :rtype: boolean

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        instance_attrs = set(filter(lambda x: x[0] != "_", instance))
        self_attrs = set(self.attrs())
        # Test if they have the same attributes using set xor (^)
        if self_attrs ^ instance_attrs:
            return False

        for attr in self_attrs:
            attr_counts = self.av_counts[attr]
            if attr == ca_key:
                if instance[ca_key] != attr_counts.keys():
                    return False
                for ctxt_count in attr_counts.values():
                    if ctxt_count != self.count:
                        return False
            elif isNumber(instance[attr]):
                if (cv_key not in attr_counts
                        or len(attr_counts) != 1
                        or attr_counts[cv_key].num != self.count
                        or attr_counts[cv_key].unbiased_std() != 0.0
                        or attr_counts[cv_key].unbiased_mean() !=
                        instance[attr]):
                    return False
            elif attr_counts.get(instance[attr], default=0) != self.count:
                return False
        return True

    def output_json(self):
        raise NotImplementedError

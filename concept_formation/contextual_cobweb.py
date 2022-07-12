"""
The Cobweb3 module contains the :class:`Cobweb3Tree` and :class:`Cobweb3Node`
classes, which extend the traditional Cobweb capabilities to support numeric
values on attributes.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
# from random import normalvariate
from itertools import cycle
from math import sqrt
from math import pi
# from math import exp
# from math import log
from collections import Counter
# from token import AT

from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.cobweb3 import Cobweb3Tree
from concept_formation.cobweb3 import cv_key
from concept_formation.continuous_value import ContinuousValue
from concept_formation.context_instance import ContextInstance
from concept_formation.utils import isNumber
# from concept_formation.utils import weighted_choice
# from concept_formation.utils import most_likely_choice

ca_key = "#ContextualAttribute#"


class ContextualCobwebTree(Cobweb3Tree):
    """
    """

    def __init__(self, ctxt_weight=1, scaling=0.5, inner_attr_scaling=True):
        """
        The tree constructor.

        :param ctxt_weight: factor by which the context should be weighted
            when combining category utility with other attribute types
        :type ctxt_weight: float
        """
        self.root = ContextualCobwebNode()
        # Root will become a leaf node
        self.root.descendants.add(self.root)
        self.root.tree = self
        self.context_weight = ctxt_weight
        self.scaling = scaling
        self.inner_attr_scaling = inner_attr_scaling
        self.attr_scales = {}

    def cobweb(self, instance):
        raise NotImplementedError

    def initial_path(self, instance):
        path = []
        current = self.root
        while current:
            path.append(current)
            if not current.children:
                # print(path)
                return path

            _, best1, best2 = current.two_best_children(instance)
            current = best1

    def cobweb_path(self, instance):
        current = self.root
        node_path = []

        while current:
            node_path.append(current)

            if not current.children:
                # print("leaf")
                break

            best1_cu, best1, best2 = current.two_best_children(instance)
            _, best_action = current.get_best_operation(
                instance, best1, best2, best1_cu, possible_ops=["best", "new"])

            # print(best_action)
            if best_action == 'best':
                current = best1
            elif best_action == 'new':
                break
            else:
                raise Exception('Best action choice "{action}" not a '
                                'recognized option. This should be'
                                ' impossible...'.format(action=best_action))

        return node_path

    def add_by_path(self, instance, context, splits):
        """
        Returns the leaf node

        splits: a dictionary mapping deleted/moved nodes
            to the node that replaced them

        updates splits
        """
        where_to_add = context.instance

        while where_to_add in splits:
            where_to_add = splits[where_to_add]

        if where_to_add.children:
            where_to_add.increment_all_counts(instance)
            return where_to_add.create_new_leaf(instance, context)

        # Leaf match or...
        # (the where_to_add.count == 0 here is for the initially empty tree)
        if where_to_add.is_exact_match(instance) or where_to_add.count == 0:
            where_to_add.increment_all_counts(instance)
            return context.set_instance(where_to_add)

        # ... fringe split
        if where_to_add.parent:
            new = where_to_add.insert_parent_with_current_counts()
        else:
            new = where_to_add.insert_parent_with_current_counts()
            self.root = new

        splits[where_to_add] = new
        new.increment_all_counts(instance)
        return new.create_new_leaf(instance, context)

    def contextual_ifit(self, instances, context_func):
        """
        Adds multiple instances, creating the correct context attributes for
        each of them.

        :param instances: instances to be added
        :type instances: Sequence<Instance>
        :param context_func: returns a subsequence of the context instances to
            consider as context for the instance at the inputted index.
        :type context_func: func: Sequence<ContextInstance>, int+ ->
            Set<ContextInstance>
        """
        contexts = tuple(ContextInstance(self.initial_path(instance))
                         for instance in instances)
        for i, instance in enumerate(instances):
            instance[ca_key] = context_func(contexts, i)

        # The most recent index that was changed
        changed_index = 0
        for index, instance in cycle(enumerate(instances)):
            new_path = self.cobweb_path(instance)
            # If paths are not the same...
            if (len(new_path) != len(contexts[index].tenative_path)
                    or any(node not in contexts[index].tenative_path
                           for node in new_path)):
                changed_index = index
                contexts[index].set_path(new_path)
            # |NP| = |TP| and NP subset of TP => set(NP) = TP
            elif index == changed_index:
                break

        # Adds all the nodes, updates the contexts, and makes the list of nodes
        splits = {}
        return [self.add_by_path(instance, context, splits)
                for instance, context in zip(instances, contexts)]

    def contextual_cobweb(self, instances, context_size=4,
                          context_key='symmetric_window'):
        if context_key == 'symmetric_window':
            def context_func(context, index):
                return (*context[max(0, index-context_size): max(0, index)],
                        *context[index+1: index+1+context_size])
        else:
            raise ValueError("Unknown context evaluator %s" % context_key)
        self.contextual_ifit(instances, context_func)


class ContextualCobwebNode(Cobweb3Node):

    def __init__(self, other_node=None):
        # Descendant registry should be updated every time a new node is added
        # to the tree. This can be done by updating a ContextInstance with the
        # final node or updating counts from other nodes.
        self.descendants = set()
        super().__init__(other_node)

    def increment_counts(self, instance):
        """
        Increment the counts at the current node according to the specified
        instance. **Does not handle adding `instance` to descendants**.

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

    def increment_all_counts(self, instance):
        # Increments all counts up to the root
        self.increment_counts(instance)
        if self.parent:
            self.parent.increment_all_counts(instance)

    def update_counts_from_node(self, node):
        """
        Increments the counts of the current node by the amount in the
        specified node, modified to handle context.

        :param node: Another node from the same Cobweb3Tree
        :type node: Cobweb3Node
        """
        self.count += node.count
        self.descendants.update(node.descendants)
        for attr in node.attrs('all'):
            if attr == ca_key:
                self.av_counts.setdefault(attr, Counter())
                self.av_counts[attr].update(node.av_counts[attr])
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
        """
        Returns the number of attribute values that would be correctly guessed
        in the current concept. This extension supports both nominal, numeric,
        and contextual attribute values.

        The typical ContextualCobweb calculation for contextual guesses is the
        expected proportion of a context instance's path one can guess with a
        probability matching strategy. If each word has path C_0, C_1, ...
        C_{n-1} and this node's context is ctxt, the formula is

            Σ_(word in ctxt)
                (P(C_{n-1} | w in ctxt)·Σ_(i = 0 to n-1) P(C_i | w in ctxt))/n

        where P(C_i | w in ctxt) is the probability a context word w chosen at
        random from ctxt (weighted by frequency) has a path through C_i. This
        is then weighted by tree.context_weight since there will only be one
        contextual attribute but it may be more important than the nominal or
        numeric attributes.

        :return: The number of attribute values that would be correctly guessed
            in the current concept.
        :rtype: float
        """
        correct_guesses = 0.0
        attr_count = 0

        for attr in self.attrs():
            if attr == ca_key:
                attr_count += self.tree.context_weight
                correct_guesses += (self.__expected_contextual(
                    self.tree.root, 0, 0, self.av_counts[attr])
                                    * self.tree.context_weight)
                continue

            attr_count += 1

            # TODO: Factor out in Cobweb3
            for val in self.av_counts[attr]:
                if val == cv_key:
                    scale = 1.0
                    if self.tree is not None and self.tree.scaling:
                        inner_attr = self.tree.get_inner_attr(attr)
                        if inner_attr in self.tree.attr_scales:
                            inner = self.tree.attr_scales[inner_attr]
                            scale = ((1/self.tree.scaling) *
                                     inner.unbiased_std())

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
                else:
                    prob = (self.av_counts[attr][val]) / self.count
                    correct_guesses += (prob * prob)

        return correct_guesses / attr_count

    def __expected_contextual(self, cur_node, partial_guesses,
                              partial_len, ctxt):
        unadded_leaf_counts = []
        # The count of some added leaf of cur_node. If cur_node is a leaf, this
        # will be how many times cur_node appears as context (possibly 0).
        added_leaf_count = 0
        extra_guesses = 0
        for wd, count in ctxt.items():
            desc, unadded_leaf = wd.desc_of(cur_node)
            if desc:
                extra_guesses += count * count
                if unadded_leaf:
                    unadded_leaf_counts.append(count)
                else:
                    added_leaf_count = count

        # No category utility here because this path has no instances
        if extra_guesses == 0:
            return 0

        new_partial_guesses = partial_guesses + extra_guesses
        new_partial_len = partial_len + 1

        if cur_node.children == []:
            ctxt_len = sum(ctxt.values())
            # If unadded_leaves > 0, a fringe split would happen
            if unadded_leaf_counts:
                # A fringe split is equivalent to adding cur_node as a new
                # leaf node in addition to the unadded leaves.
                unadded_leaf_counts.append(added_leaf_count)
                # Calculate the cu of all leaf nodes
                return self.__unadded_leaves_cu(
                    unadded_leaf_counts, new_partial_guesses,
                    new_partial_len, ctxt_len)

            # ctxt_len divided out twice for P(C_i | w in ctxt) and once for
            # the outer weighted average. Because it's a weighted average, we
            # multiply by added_leaf_count (count of cur_node in context).
            return (added_leaf_count * new_partial_guesses /
                    # TODO: Factor out / ctxt_len^2
                    (new_partial_len * ctxt_len * ctxt_len))

        if unadded_leaf_counts:
            ctxt_len = sum(ctxt.values())
            # Calculate the cu of the leaf nodes
            partial_category_utility = self.__unadded_leaves_cu(
                unadded_leaf_counts, new_partial_guesses,
                new_partial_len, ctxt_len)
        else:
            partial_category_utility = 0

        for child in cur_node.children:
            partial_category_utility += self.__expected_contextual(
                child, new_partial_guesses, new_partial_len, ctxt)
        return partial_category_utility

    def __unadded_leaves_cu(self, unadded_leaf_counts, partial_guesses,
                            partial_len, ctxt_len):
        """partial_len depth of where the nodes will be added (0 indexed) """
        return (
            sum(count * (count * count + partial_guesses)
                for count in unadded_leaf_counts)
            / ((partial_len + 1) * ctxt_len * ctxt_len))

    def get_weighted_values(self, attr, allow_none=True):
        if attr == ca_key:
            raise NotImplementedError('Context prediction not implemented')
        else:
            super().get_weighted_values(attr, attr, allow_none)

    def predict(self, attr, choice_fn="most likely", allow_none=True):
        if attr == ca_key:
            raise NotImplementedError('Context prediction not implemented')
        else:
            super().predict(attr, choice_fn, allow_none)

    def probability(self, attr, val):
        raise NotImplementedError

    def log_likelihood(self, child_leaf):
        raise NotImplementedError

    def create_new_leaf(self, instance, context_wrapper):
        """
        Create a new leaf (to the current node) with the counts initialized by
        the *given instance*.

        This is the operation used for creating a new leaf beneath a node and
        adding the instance to it.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :param context_wrapper: context_wrapper to insert the new instance into
        :type context_wrapper: ContextInstance
        :return: The new child
        :rtype: ContextualCobwebNode
        """
        return context_wrapper.set_instance(self.create_new_child(instance))

    def create_child_with_current_counts(self):
        """Fringe splits cannot be done by adding nodes below."""
        raise AttributeError("Context-aware leaf nodes must remain leaf nodes")

    def insert_parent_with_current_counts(self):
        # does not handle updating root in tree if necessary
        if self.count > 0:
            new = self.__class__()
            new.update_counts_from_node(self)
            new.tree = self.tree

            if self.parent:
                # Replace self with new node in the parent's children
                index_of_self_in_parent = self.parent.children.index(self)
                self.parent.children[index_of_self_in_parent] = new

            new.parent = self.parent
            new.children.append(self)
            self.parent = new
            return new

    def cu_for_fringe_split(self, instance):
        """
        Return the category utility of performing a fringe split (i.e.,
        adding a leaf to a leaf).

        A "fringe split" is essentially a new operation performed at a leaf. It
        is necessary to have the distinction because unlike a normal split a
        fringe split must also push the parent down to maintain a proper tree
        structure. This is useful for identifying unnecessary fringe splits,
        when the two leaves are essentially identical. It can be used to keep
        the tree from growing and to increase the tree's predictive accuracy.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility of fringe splitting at the current node.
        :rtype: float

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        leaf = self.shallow_copy()

        parent = leaf.insert_parent_with_current_counts()
        parent.increment_counts(instance)
        parent.create_new_child(instance)

        return parent.category_utility()

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
            elif attr_counts.get(instance[attr], 0) != self.count:
                return False
        return True

    def output_json(self):
        raise NotImplementedError

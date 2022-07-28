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
from time import time
from collections import Counter, deque
# from token import AT

from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.cobweb3 import Cobweb3Tree
from concept_formation.cobweb3 import cv_key
from concept_formation.continuous_value import ContinuousValue
from concept_formation.context_instance import ContextInstance
from concept_formation.utils import isNumber
# from concept_formation.utils import weighted_choice
# from concept_formation.utils import most_likely_choice

ca_key = "#Ctxt#"  # TODO: Change to something longer


def enumerate_from_back(iter):
    index = -1
    for item in reversed(iter):
        yield (index, item)
        index -= 1


class ContextualCobwebTree(Cobweb3Tree):
    """

    :param ctxt_weight: factor by which the context should be weighted
        when combining category utility with other attribute types
    :type ctxt_weight: float
    :param scaling: The number of standard deviations numeric attributes
        are scaled to. By default this value is 0.5 (half a standard
        deviation), which is the max std of nominal values. If disabiling
        scaling is desirable, then it can be set to False or None.
    :type scaling: a float greater than 0.0, None, or False
    :param inner_attr_scaling: Whether to use the inner most attribute name
        when scaling numeric attributes. For example, if `('attr', '?o1')` was
        an attribute, then the inner most attribute would be 'attr'. When using
        inner most attributes, some objects might have multiple attributes
        (i.e., 'attr' for different objects) that contribute to the scaling.
    :type inner_attr_scaling: boolean
    """
    def __init__(self, ctxt_weight=1, scaling=0.5, inner_attr_scaling=True):
        """
        The tree constructor.
        """
        self.root = ContextualCobwebNode()
        # Root will become a leaf node
        self.root.descendants.add(self.root)
        self.root.tree = self
        self.context_weight = ctxt_weight
        self.scaling = scaling
        self.inner_attr_scaling = inner_attr_scaling
        self.attr_scales = {}

    def clear(self):
        """
        Clears the concepts of the tree, but maintains the scaling parameters.
        """
        self.root = ContextualCobwebNode()
        self.root.descendants.add(self.root)
        self.root.tree = self
        self.attr_scales = {}

    def contextual_ifit(self, instances, context_size=4,
                        context_key='symmetric_window'):
        """
        Incrementally fit new instances into the tree and return the resulting
        concepts.

        The instances are passed down the cobweb tree and update each node to
        incorporate themselves. **This process modifies the tree's knowledge**

        :param instances: instances to be added
        :type instances: Sequence<:ref:`Instance<instance-rep>`>
        :param context_size: hyperparameter used for constructing the context
            function.
        :type context_size: int
        :param context_key: how context should be chosen. Should be one of
            symmetric_window (size instances on either side of anchor)
            past_window (size instances on the left of anchor)
            future_window (size instances on the right of anchor)
        :type context_key: 'symmetric_window', 'past_window', 'future_window'
        :return: list of the nodes where the instances were added
        :rtype: List<ContextualCobwebNode>
        """
        assert context_key != 'past_window' or context_key != 'future_window'
        for instance in instances:
            self._sanity_check_instance(instance)
        return self.contextual_cobweb(instances, context_size, context_key)

    def contextual_cobweb(self, instances, context_size, context_key):
        """
        The core context-aware algorithm. Categorizes *and then adds* instances

        :param instances: instances to be added
        :type instances: Sequence<:ref:`Instance<instance-rep>`>
        :param context_key: how context should be chosen. Should be one of
            symmetric_window (size instances on either side of anchor)
            past_window (size instances on the left of anchor)
            future_window (size instances on the right of anchor)
        :type context_key: 'symmetric_window', 'past_window', 'future_window'
        :return: list of the created leaf nodes
        :rtype: List<ContextualCobwebNode>
        """
        assert context_key == 'symmetric_window'
        initial_contexts = [ContextInstance(self.initial_path(instance))
                            for instance in instances[:context_size + 1]]
        fixed = []  # [-context_size:]
        window = deque(zip(instances[:context_size + 1], initial_contexts))
        for inst, _ in window:
            inst[ca_key] = Counter(initial_contexts)
        next_to_initialize = context_size + 1

        while window:
            # # # We will be fixing the first element of the deque # # #
            # The index in the window of the node which most
            # recently had its path altered
            last_changed = len(window)-1
            # Whether the iterations have returned to a seen state
            looped = False
            records = {*()}

            iterations = 0
            start = time()
            for index, (inst, ctx) in cycle(enumerate(window)):
                if index == 0:
                    iterations += 1

                    new_record = tuple(ctx.instance for _, ctx in window)
                    if new_record in records:
                        looped = True
                        # print("Loop")
                    records.add(new_record)
                    # Actions is the splits and merges which should be done
                    path, new_actns = self.cobweb_path_and_restructurings(inst)
                else:
                    path = self.cobweb_path(inst)

                if looped:
                    if (not self.__path_eq(ctx.tenative_path, path)
                            and self.__update_if_better(window, path, ctx)):
                        last_changed = index
                        if index == 0:
                            actions = new_actns
                    # If we have stabilized
                    elif last_changed == index:
                        break
                    continue

                actions = new_actns
                if not self.__path_eq(ctx.tenative_path, path):
                    ctx.set_path(path)
                    # print('node %s changed in iter %s' % (index, iterations))
                    last_changed = index
                elif last_changed == index:
                    break

            next_to_initialize += 1
            # Add the instance to fixed
            fixed.append(self.add_by_path(*window.popleft(), actions, window))

            if next_to_initialize < len(instances):
                instance = instances[next_to_initialize]
                instance[ca_key] = Counter(ctx for _, ctx in window)
                context = ContextInstance(self.cobweb_path(instance))

                for inst, _ in window:
                    inst[ca_key][context] += 1
                window.append((instance, context))
            print(next_to_initialize-context_size,
                  time()-start, iterations, sep='\\t')
            # print('word_num', next_to_initialize-context_size, 'time',
            #       time()-start, 'iterations', iterations, sep='\\t')
        return fixed

    def __update_if_better(self, window, new_path, ctxt):
        """returns True if path changed, false otherwise"""
        old_path = ctxt.tenative_path
        old_inst = ctxt.instance

        old_cu = self.__cu_for_window(window)
        ctxt.set_path(new_path)
        new_cu = self.__cu_for_window(window)

        if new_cu > old_cu:
            return True

        ctxt.set_path_from_set(old_path, old_inst)
        return False

    def __path_eq(self, set_path, tup_path):
        """Checks if these are equal. Optimized for correct input types"""
        # If paths are not the same...
        # Observe |NP| = |TP| and NP subset of TP => set(NP) = TP
        return (len(tup_path) == len(set_path)
                and all(node in set_path for node in tup_path))

    def __cu_for_window(self, window):
        return sum(ctx.instance.cu_for_new_child(inst) for inst, ctx in window)

    def initial_path(self, instance):
        """
        Returns the path of the best guess for where instance will go,
        specifically for instances that do not yet have context.

        :param instance: the instance to categorize
        :type instance: :ref:`Instance<instance-rep>`
        :return: the best guess for the instance's insertion into the tree
        :rtype: Sequence<ContextualCobwebNode>"""
        current = self.root
        node_path = []

        while current:
            node_path.append(current)

            if not current.children:
                # print("leaf")
                break

            # TODO: Generalize hack beyond "Anchor" atrribute
            '''tie = False
            best_child = None
            best_count = float('-inf')
            for child in current.children:
                count = child.av_counts['Anchor'].get(instance['Anchor'], 0)
                if best_count <= count:
                    tie = best_count == count
                    best_count = count
                    best_child = child
            if not tie:
                current = best_child
                continue'''

            best1_cu, best1, best2 = current.two_best_children(instance)
            _, best_action = current.get_best_operation(
                instance, best1, best2, best1_cu, possible_ops=("best", "new"))

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

    def cobweb_path(self, instance):
        """
        Returns the path of the place where adding instance will maximize
        :meth:`category utility <CobwebNode.category_utility>`. Handles
        contextual attributes.

        In the general case, the cobweb algorithm considers making a new leaf
        for the instance or adding it to a child of the current node.

        At each node the alogrithm first calculates the category utility of
        inserting the instance at each of the node's children, keeping the best
        two (see: :meth:`CobwebNode.two_best_children
        <CobwebNode.two_best_children>`) and then calculates the
        category_utility of making a new node using the best two
        children (see: :meth:`CobwebNode.get_best_operation
        <CobwebNode.get_best_operation>`), either continuing down the tree or
        ending the path there  depending on which is better. Ties are randomly
        broken.

        In the base case, i.e. a leaf node, the algorithm returns the path it
        followed to get there.

        :param instance: the instance to categorize
        :type instance: :ref:`Instance<instance-rep>`
        :return: the best place for inserting instance
        :rtype: Sequence<ContextualCobwebNode>"""
        current = self.root
        node_path = []

        while current:
            node_path.append(current)

            if not current.children:
                # print("leaf")
                break

            best1_cu, best1, best2 = current.two_best_children(instance)
            _, best_action = current.get_best_operation(
                instance, best1, best2, best1_cu, possible_ops=("best", "new"))

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

    def cobweb_path_and_restructurings(self, instance):
        """GETS restructurings (only contains places where
        an action could actually take place"""
        current = self.root
        node_path = []
        actions = []

        while current:
            node_path.append(current)

            if not current.children:
                break

            best1_cu, best1, best2 = current.two_best_children(instance)
            action_cu, best_action = current.get_best_operation(
                instance, best1, best2, best1_cu, possible_ops=('best', 'new'))

            actions.append((current, action_cu, best1_cu, best2, best1))
            current = best1
            if best_action == 'new':
                break

        return (node_path, actions)

    def add_by_path(self, instance, context, actions, unadded_window):
        """
        Inserts instance at the path specified by the context, updating all the
        necessary counts. It also finalizes the context and logs any splits
        that were performed in splits.

        :param instance: the instance to add
        :type instance: :ref:`Instance<instance-rep>`
        :param context: ContextInstance with the instance's path, which will be
            updated to hold that instance.
        :type context: ContextInstance
        :param splits: a dictionary mapping deleted/moved nodes to the node
            that replaced them, where all splits performed in adding the node
            should be logged.
        :type splits: dict<ContextualCobwebNode, ContextualCobwebNode>
        :return: the newly created leaf node
        :rtype: ContextualCobwebNode
        """
        where_to_add = context.instance

        if where_to_add.children:
            # print('leaf')
            leaf = where_to_add.create_new_leaf(instance, context)
            self.increment_and_restructure(
                instance, where_to_add, actions, unadded_window)
            return leaf

        # Leaf match or...
        # (the where_to_add.count == 0 here is for the initially empty tree)
        if where_to_add.count == 0 or where_to_add.is_exact_match(instance):
            # print('leaf match')
            leaf = context.set_instance(where_to_add)
            self.increment_and_restructure(
                instance, where_to_add, actions, unadded_window)
            return leaf

        # ... fringe split
        # print('fringe split')
        new = where_to_add.insert_parent_with_current_counts()

        leaf = new.create_new_leaf(instance, context)
        self.__split_update(where_to_add, unadded_window)

        if actions:
            # Replace best1 with the new node
            actions[-1] = (*actions[-1][:-1], new)

        self.increment_and_restructure(instance, new, actions, unadded_window)
        return leaf

    def increment_and_restructure(self, instance, where_to_add,
                                  actions, unadded_window):
        where_to_add.increment_all_counts(instance)

        # By going backwards, splits don't cause problems
        for current, action_cu, best1_cu, best2, best1 in reversed(actions):
            # Note that comparing 'merges' and 'splits' don't consider how
            # the context attributes would change given the operation. This
            # is good because it prevents "cheating," where cobweb flattens
            # the hierarchy to make context proportionally easier to guess.
            # Empty object is inserted since all counts have been incremented.
            if len(current.children) <= 2 and len(best1.children) == 0:
                continue

            new_action_cu, new_best_action = current.get_best_operation(
                {}, best1, best2, best1_cu, possible_ops=('split', 'merge'))

            if new_action_cu <= action_cu:
                continue

            if new_best_action == 'merge':
                assert len(current.children) > 2
                node = current.merge(best1, best2)
                self.__merge_update(node, unadded_window)
            elif new_best_action == 'split':
                assert len(best1.children) > 1
                current.split(best1)
                self.__split_update(best1, unadded_window)
            else:
                raise Exception('Best action choice "' + new_best_action +
                                '" not a recognized option. This should be'
                                ' impossible...')

    def __merge_update(self, node, unadded_ctxts):
        # print('merge')
        for _, ctx in unadded_ctxts:
            assert ctx.unadded()
            if ctx.desc_of(node.parent):
                ctx.insert_into_path(node)

    def __split_update(self, dead_node, unadded_ctxts):
        # print('split')
        for _, ctx in unadded_ctxts:
            assert ctx.unadded()
            if ctx.instance == dead_node:
                ctx.instance = dead_node.parent

    def cobweb(self, instance):
        raise NotImplementedError

    def _cobweb_categorize(self, instance):
        """
        A cobweb specific version of categorize, not intended to be
        externally called.
        """
        current = self.root

        while current:
            if not current.children:
                # print("leaf")
                break

            best1_cu, best1, best2 = current.two_best_children(instance)
            _, best_action = current.get_best_operation(
                instance, best1, best2, best1_cu, possible_ops=("best", "new"))

            # print(best_action)
            if best_action == 'best':
                current = best1
            elif best_action == 'new':
                break
            else:
                raise Exception('Best action choice "{action}" not a '
                                'recognized option. This should be'
                                ' impossible...'.format(action=best_action))

        return current

    def infer_from_context(self, instances, context_size=4,
                           context_key='symmetric_window'):
        """
        Categorize instances and use this to predict the attributes of the
        unknown instance, entered as None.

        :param instances: instances to be added
        :type instances: Sequence<:ref:`Instance<instance-rep>` w/ one None>
        :param context_size: hyperparameter used for constructing the context
            function.
        :type context_size: int
        :param context_key: how context should be chosen. Should be one of
            symmetric_window (size instances on either side of anchor)
            past_window (size instances on the left of anchor)
            future_window (size instances on the right of anchor)
        :type context_key: 'symmetric_window', 'past_window', 'future_window'
        :return: list of the nodes where the instances were added
        :rtype: List<ContextualCobwebNode>
        """
        raise NotImplementedError
        instances = list(instances)
        assert len(instances) > 1, "Not enough context to make prediction"
        pred_ind = instances.index(None)

        # TODO: Should eventually be refactored
        if context_key == 'symmetric_window':
            def context_func(context, index):
                low = max(0, index-context_size)
                hi = index + 1 + context_size
                return (*context[low+(low < pred_ind <= index): index],
                        *context[index+1: hi-(index < pred_ind < hi)])
        elif context_key == 'past_window' or context_key == 'future_window':
            raise NotImplementedError
        else:
            raise ValueError("Unknown context evaluator %s" % context_key)
        if context_key == 'symmetric_window':
            def eval_func(context, index):
                low = max(0, index-context_size)
                hi = index + 1 + context_size
                return context[low + (low < pred_ind <= index):
                               hi - (index < pred_ind < hi)]
        elif context_key == 'past_window' or context_key == 'future_window':
            raise NotImplementedError
        else:
            raise ValueError("Unknown eval context evaluator %s" % context_key)

        del instances[pred_ind]
        for instance in instances:
            self._sanity_check_instance(instance)

        paths = list(
            self._contextual_categorize(instances, context_func, eval_func)[1])

        paths.insert(pred_ind, None)
        ind_copy = pred_ind

        # Revert context functions to become normal context functions
        pred_ind = -1

        node = self._cobweb_categorize({ca_key: context_func(paths, ind_copy)})
        result = dict(node.av_counts)
        del result[ca_key]
        return result


class ContextualCobwebNode(Cobweb3Node):
    """
    A ContextualCobwebNode represents a concept within the knowledge base of a
    particular :class:`ContextualCobwebTree`. Each node contains a probability
    table that can be used to calculate the probability of different attributes
    given the concept that the node represents.

    In general the :meth:`ContextualCobwebTree.contextual_ifit`,
    :meth:`ContextualCobwebTree.categorize` functions should be used to
    initially interface with the Contextual Cobweb knowledge base and then the
    returned concept can be used to calculate probabilities of certain
    attributes or determine concept labels.
    """
    def __init__(self, other_node=None):
        """Create a new ContextualCobwebNode"""
        # Descendant registry should be updated every time a new node is added
        # to the tree. This can be done by updating a ContextInstance with the
        # final node or updating counts from other nodes.
        self.descendants = set()
        super().__init__(other_node)

    def increment_counts(self, instance):
        """
        Increment the counts at the current node according to the specified
        instance. **Does not alter descendants registry**

        ContextualCobwebNode uses a modified version of
        :meth:`Cobweb3Node.increment_counts
        <concept_formation.cobweb3.Cobweb3Node.increment_counts>` that handles
        contextual attributes properly. The attribute equalling ca_key will be
        treated as context.

        :param instance: A new instance to incorporate into the node.
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
        """
        Increment the counts at the current node *and all its ancestors
        according to the specified instance*. **Does not alter descendants
        registry**

        :param instance: a new instance to incorporate into the nodes.
        :type instance: :ref:`Instance<instance-rep>`
        """
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
        Returns the expected proportion of attribute values that would be
        correctly guessed in the current concept. This extension supports
        nominal, numeric, and contextual attribute values.

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
        """
        Recursive helper for expected_correct_guesses. Calculates the expected
        proportion of the context's path guessed.

        :param cur_node: current node being examined
        :type cur_node: ContextualCobwebNode
        :param partial_guesses: partial sum of how many guesses were correct
        :type partial_guesses: int
        :param partial_len: number of nodes already examined
            alternatively, the depth of the cur_node (0-indexed)
        :type partial_len: int
        :param ctxt: context of node whose correct guesses are being evaluated
        :type ctxt: Counter<ContextInstance>"""
        ctx_len = sum(ctxt.values())
        if ctx_len == 0:
            return 0
        # ctxt_len will divided out twice for P(C_i | w in ctxt) and once for
        # the outer weighted average.
        return self.__exp_ctxt_helper(cur_node, partial_guesses, partial_len,
                                      ctxt.items()) / (ctx_len*ctx_len)

    def __exp_ctxt_helper(self, cur_node, partial_guesses, partial_len, ctxt):
        """
        Calculates the expected proportion of the context's path guessed times
        the length of the context squared.
        """
        # The full formula for unadded leaves is this:
        #   sum(count * (count + new_partial_guesses)
        #       for count in unadded_leaf_counts) / (new_partial_len + 1)
        # where unadded_leaf_counts is how many times each of the unadded
        # leaves appears as context. This is equivalent to
        #   (sum(count for count in unadded_leaf_counts) * new_partial_guesses
        #   + sum(count * count for count in unadded_leaf_counts))
        #   / (new_partial_len + 1)
        squared_ualeaf_count = 0
        cum_ualeaf_count = 0
        # The count of some added leaf of cur_node. If cur_node is a leaf, this
        # will be how many times cur_node appears as context (possibly 0).
        added_leaf_count = 0
        extra_guesses = 0
        descendents = []
        for wd, count in ctxt:
            if wd.desc_of(cur_node):
                descendents.append((wd, count))
                extra_guesses += count
                if wd.unadded_leaf(cur_node):
                    squared_ualeaf_count += count * count
                    cum_ualeaf_count += count
                else:
                    added_leaf_count = count

        # No category utility here because this path has no instances
        if extra_guesses == 0:
            return 0

        new_partial_guesses = partial_guesses + extra_guesses
        new_partial_len = partial_len + 1

        # Calculate the cu of the leaf nodes
        if cum_ualeaf_count:
            partial_cu = ((cum_ualeaf_count * new_partial_guesses
                           + squared_ualeaf_count) / (new_partial_len + 1))
        else:
            partial_cu = 0
        # Note that this will account for fringe splits when measuring unadded
        # leaves but not the leaf itself. The reason we don't consider the main
        # node as being fringe split is that it creates inconsistencies where
        # some cu calculations account for the changing structure of the tree
        # while others (those without the leaves as context) don't. Since the
        # philosophy is, in general, to not update the tree until the very end,
        # this is most consistent.

        if not cur_node.children or partial_len > 5:
            # Because it's a weighted average, we multiply by added_leaf_count
            # (count of cur_node in context).
            return (added_leaf_count * new_partial_guesses / new_partial_len
                    + partial_cu)

        for child in cur_node.children:
            partial_cu += self.__exp_ctxt_helper(
                child, new_partial_guesses, new_partial_len, descendents)
        return partial_cu

    def get_best_operation(self, instance, best1, best2, best1_cu,
                           possible_ops=("best", "new", "merge", "split")):
        """
        Given an instance, the two best children based on category utility and
        a set of possible operations, find the operation that produces the
        highest category utility, and then return the category utility and name
        for the best operation. In the case of ties, an operator is chosen with
        the following priorities: best, new, split, merge.

        Given the following starting tree the results of the 4 standard Cobweb
        operations are shown below:

        .. image:: images/Original.png
            :width: 200px
            :align: center

        * **Best** - Categorize the instance to child with the best category
          utility. This results in a recurisve call to :meth:`cobweb
          <concept_formation.cobweb.CobwebTree.cobweb>`.

            .. image:: images/Best.png
                :width: 200px
                :align: center

        * **New** - Create a new child node to the current node and add the
          instance there. See: :meth:`create_new_child
          <concept_formation.cobweb.CobwebNode.create_new_child>`.

            .. image:: images/New.png
                :width: 200px
                :align: center

        * **Merge** - Take the two best children, create a new node as their
          mutual parent and add the instance there. See: :meth:`merge
          <concept_formation.cobweb.CobwebNode.merge>`.

            .. image:: images/Merge.png
                    :width: 200px
                    :align: center

        * **Split** - Take the best node and promote its children to be
          children of the current node and recurse on the current node. See:
          :meth:`split <concept_formation.cobweb.CobwebNode.split>`

            .. image:: images/Split.png
                :width: 200px
                :align: center

        Each operation is entertained and the resultant category utility is
        used to pick which operation to perform. The list of operations to
        entertain can be controlled with the possible_ops parameter. For
        example, when performing categorization without modifying knoweldge
        only the best and new operators are used.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :param best1: A tuple containing the relative cu of the best child and
            the child itself, as determined by
            :meth:`CobwebNode.two_best_children`.
        :type best1: (float, CobwebNode)
        :param best2: A tuple containing the relative cu of the second best
            child and the child itself, as determined by
            :meth:`CobwebNode.two_best_children`.
        :type best2: (float, CobwebNode)
        :param possible_ops: A list of operations from ["best", "new", "merge",
            "split"] to entertain.
        :type possible_ops: ["best", "new", "merge", "split"]
        :return: A tuple of the category utility of the best operation and the
            name of the best operation.
        :rtype: (cu_bestOp, name_bestOp)
        """
        if not best1:
            raise ValueError("Need at least one best child.")

        operations = []

        if "best" in possible_ops:
            operations.append((best1_cu, 3, "best"))
        if "new" in possible_ops:
            operations.append((self.cu_for_new_child(instance), 2, 'new'))
        if "merge" in possible_ops and len(self.children) > 2 and best2:
            operations.append((self.cu_for_merge(best1, best2, instance),
                               0, 'merge'))
        if "split" in possible_ops and len(best1.children) > 0:
            operations.append((self.cu_for_split(best1), 1, 'split'))

        operations.sort(reverse=True)

        return (operations[0][0], operations[0][2])

    def two_best_children(self, instance):
        """
        Calculates the category utility of inserting the instance into each of
        this node's children and returns the best two. In the event of ties
        children are sorted first by category utility, then by their size, then
        by a random value.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility and indices for the two best children
            (the second tuple will be ``None`` if there is only 1 child).
        :rtype: ((cu_best1,index_best1),(cu_best2,index_best2))
        """
        if len(self.children) == 0:
            raise Exception("No children!")

        # Convert the relative CU's of the two best children into CU scores
        # that can be compared with the other operations.
        const = self.compute_relative_CU_const(instance)

        # If there's only one child, simply calculate the relevant utility
        if len(self.children) == 1:
            best1 = self.children[0]
            best1_relative_cu = self.relative_cu_for_insert(best1, instance)
            best1_cu = (best1_relative_cu / (self.count+1) / len(self.children)
                        + const)
            return best1_cu, best1, None

        children_relative_cu = [(self.relative_cu_for_insert(child, instance),
                                 child.count, child) for child in
                                self.children]
        children_relative_cu.sort(reverse=True, key=lambda x: x[:-1])

        best1_data, best2_data = children_relative_cu[:2]

        best1_relative_cu, _, best1 = best1_data
        best1_cu = (best1_relative_cu / (self.count+1) / len(self.children)
                    + const)
        best2 = best2_data[2]

        return best1_cu, best1, best2

    def create_new_leaf(self, instance, context_wrapper):
        """
        Create a new leaf (to the current node) with the counts initialized by
        the *given instance*.

        This is the operation used for creating a new leaf beneath a node and
        adding the instance to it.

        :param instance: the instance currently being categorized
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

    def insert_parent_with_current_counts(self, update_root=True):
        """
        Insert a parent above the current node with the counts initialized by
        the current node's counts. *By default this updates the root if needed*

        This operation is used in the speical case of a fringe split when a new
        node is created at a leaf.

        :return: the new parent
        :rtype: ContextualCobwebNode
        """
        # does not handle updating root in tree if necessary
        if self.count > 0:
            new = self.__class__()
            new.update_counts_from_node(self)
            new.tree = self.tree

            if self.parent:
                # Replace self with new node in the parent's children
                index_of_self_in_parent = self.parent.children.index(self)
                self.parent.children[index_of_self_in_parent] = new
            elif update_root:
                self.tree.root = new

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
        # TODO: call to insert_parent... has effects outside the shallow copy.
        # Do not remove error until this is fixed.
        raise NotImplementedError
        leaf = self.shallow_copy()

        parent = leaf.insert_parent_with_current_counts()
        parent.increment_counts(instance)
        parent.create_new_child(instance)

        return parent.category_utility()

    def is_exact_match(self, instance):
        """
        Returns true if the concept exactly matches the instance.

        :param instance: the instance currently being categorized
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

    def __repr__(self):
        return 'N%s' % self.concept_id

    def pretty_print(self, depth=0, include_cu=False, max_depth=7):
        """
        Print the categorization tree

        The string formatting inserts tab characters to align child nodes of
        the same depth. Numerical values are printed with their means and
        standard deviations.

        :param depth: the current depth in the print, intended to be called
            recursively
        :type depth: int
        :param include_cu: include category utilities in printout
        :type include_cu: bool
        :return: a formated string displaying the tree and its children
        :rtype: str
        """
        ret = str(('\t' * depth) + "|-%s " % self.concept_id)
        if depth >= max_depth:
            return ret + " {...}"

        attributes = []

        for attr in self.attrs('all'):
            values = []
            for val in self.av_counts[attr]:
                values.append("'" + str(val) + "': " +
                              str(self.av_counts[attr][val]))

            attributes.append("'" + str(attr) + "': {" + ", ".join(values)
                              + "}")

        ret += "{" + ", ".join(attributes) + "}: " + str(self.count)
        ret += (' (cu: %s)' % round(self.category_utility(), 5) if include_cu
                else '') + '\n'

        for c in self.children:
            ret += c.pretty_print(depth+1)

        return ret

    def output_json(self):
        raise NotImplementedError

    def get_weighted_values(self, attr, allow_none=True):
        """
        Return a list of weighted choices for an attribute based on the node's
        probability table. Same as Cobweb3

        See :meth:`Cobweb3Node.get_weighted_values"""
        if attr == ca_key:
            raise NotImplementedError('Context prediction not implemented')
        else:
            super().get_weighted_values(attr, attr, allow_none)

    def predict(self, attr, choice_fn="most likely", allow_none=True):
        """
        Predict the value of an attribute, using the provided strategy.
        Same as Cobweb3

        See :meth:`Cobweb3Node.predict"""
        if attr == ca_key:
            raise NotImplementedError('Context prediction not implemented')
        else:
            super().predict(attr, choice_fn, allow_none)

    def probability(self, attr, val):
        raise NotImplementedError

    def log_likelihood(self, child_leaf):
        raise NotImplementedError

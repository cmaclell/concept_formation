"""
The Cobweb module contains the :class:`CobwebTree` and :class:`CobwebNode`
classes which are used to achieve the basic Cobweb functionality.
"""
import json
from random import shuffle
from random import random
from math import log
from math import isclose
from collections import defaultdict
from collections import Counter

from concept_formation.utils import weighted_choice
from concept_formation.utils import most_likely_choice


class MultinomialCobwebTree(object):
    """
    The CobwebTree contains the knoweldge base of a partiucluar instance of the
    cobweb algorithm and can be used to fit and categorize instances.
    """

    def __init__(self):
        """
        The tree constructor.
        """
        self.root = MultinomialCobwebNode()
        self.root.tree = self
        self.alpha_weight = 10
        # self.alpha = 0.01
        self.attr_vals = defaultdict(set)

    def alpha(self, attr):
        n_vals = len(self.attr_vals[attr])
        if n_vals == 0:
            return 1.0
        else:
            return self.alpha_weight / n_vals

    def clear(self):
        """
        Clears the concepts of the tree.
        """
        self.root = MultinomialCobwebNode()
        self.root.tree = self

    def __str__(self):
        return str(self.root)

    def _sanity_check_instance(self, instance):
        return True

    def ifit(self, instance):
        """
        Incrementally fit a new instance into the tree and return its resulting
        concept.

        The instance is passed down the cobweb tree and updates each node to
        incorporate the instance. **This process modifies the tree's
        knowledge** for a non-modifying version of labeling use the
        :meth:`CobwebTree.categorize` function.

        :param instance: An instance to be categorized into the tree.
        :type instance:  :ref:`Instance<instance-rep>`
        :return: A concept describing the instance
        :rtype: CobwebNode

        .. seealso:: :meth:`CobwebTree.cobweb`
        """
        self._sanity_check_instance(instance)
        for attr in instance:
            for val in instance[attr]:
                self.attr_vals[attr].add(val)

        return self.cobweb(instance)

    def fit(self, instances, iterations=1, randomize_first=True):
        """
        Fit a collection of instances into the tree.

        This is a batch version of the ifit function that takes a collection of
        instances and categorizes all of them. The instances can be
        incorporated multiple times to burn in the tree with prior knowledge.
        Each iteration of fitting uses a randomized order but the first pass
        can be done in the original order of the list if desired, this is
        useful for initializing the tree with specific prior experience.

        :param instances: a collection of instances
        :type instances:  [:ref:`Instance<instance-rep>`,
            :ref:`Instance<instance-rep>`, ...]
        :param iterations: number of times the list of instances should be fit.
        :type iterations: int
        :param randomize_first: whether or not the first iteration of fitting
            should be done in a random order or in the list's original order.
        :type randomize_first: bool
        """
        instances = [i for i in instances]

        for x in range(iterations):
            if x == 0 and randomize_first:
                shuffle(instances)
            for i in instances:
                self.ifit(i)
            shuffle(instances)

    def cobweb(self, instance):
        """
        The core cobweb algorithm used in fitting and categorization.

        In the general case, the cobweb algorithm entertains a number of
        sorting operations for the instance and then commits to the operation
        that maximizes the :meth:`category utility
        <CobwebNode.category_utility>` of the tree at the current node and then
        recurses.

        At each node the alogrithm first calculates the category utility of
        inserting the instance at each of the node's children, keeping the best
        two (see: :meth:`CobwebNode.two_best_children
        <CobwebNode.two_best_children>`), and then calculates the
        category_utility of performing other operations using the best two
        children (see: :meth:`CobwebNode.get_best_operation
        <CobwebNode.get_best_operation>`), commiting to whichever operation
        results in the highest category utility. In the case of ties an
        operation is chosen at random.

        In the base case, i.e. a leaf node, the algorithm checks to see if
        the current leaf is an exact match to the current node. If it is, then
        the instance is inserted and the leaf is returned. Otherwise, a new
        leaf is created.

        .. note:: This function is equivalent to calling
            :meth:`CobwebTree.ifit` but its better to call ifit because it is
            the polymorphic method siganture between the different cobweb
            family algorithms.

        :param instance: an instance to incorporate into the tree
        :type instance: :ref:`Instance<instance-rep>`
        :return: a concept describing the instance
        :rtype: CobwebNode

        .. seealso:: :meth:`CobwebTree.ifit`, :meth:`CobwebTree.categorize`
        """
        current = self.root

        while current:
            # the current.count == 0 here is for the initially empty tree.
            if not current.children and (current.is_exact_match(instance) or
                                         current.count == 0):
                # print("leaf match")
                current.increment_counts(instance)
                break

            elif not current.children:
                # print("fringe split")
                new = current.__class__(current)
                current.parent = new
                new.children.append(current)

                if new.parent:
                    new.parent.children.remove(current)
                    new.parent.children.append(new)
                else:
                    self.root = new

                new.increment_counts(instance)
                current = new.create_new_child(instance)
                break

            else:
                best1_cu, best1, best2 = current.two_best_children(instance)
                _, best_action = current.get_best_operation(instance, best1,
                                                            best2, best1_cu)

                # print(best_action)
                if best_action == 'best':
                    current.increment_counts(instance)
                    current = best1
                elif best_action == 'new':
                    current.increment_counts(instance)
                    current = current.create_new_child(instance)
                    break
                elif best_action == 'merge':
                    current.increment_counts(instance)
                    new_child = current.merge(best1, best2)
                    current = new_child
                elif best_action == 'split':
                    current.split(best1)
                else:
                    raise Exception('Best action choice "' + best_action +
                                    '" not a recognized option. This should be'
                                    ' impossible...')

        return current

    def _cobweb_categorize(self, instance):
        """
        A cobweb specific version of categorize, not intended to be
        externally called.

        .. seealso:: :meth:`CobwebTree.categorize`
        """
        current = self.root;

        while (current != None):
            if len(current.children) == 0:
                return current

            best_logp = current.log_prob_class_given_instance(instance)
            parent = current
            current = None
            for child in parent.children:
                logp = child.log_prob_class_given_instance(instance)
                if current is None or logp > best_logp:
                # if logp > best_logp:
                    best_logp = logp
                    current = child

        return parent


    def infer_missing(self, instance, choice_fn="most likely",
                      allow_none=True):
        """
        Given a tree and an instance, returns a new instance with attribute
        values picked using the specified choice function (either "most likely"
        or "sampled").

        .. todo:: write some kind of test for this.

        :param instance: an instance to be completed.
        :type instance: :ref:`Instance<instance-rep>`
        :param choice_fn: a string specifying the choice function to use,
            either "most likely" or "sampled".
        :type choice_fn: a string
        :param allow_none: whether attributes not in the instance can be
            inferred to be missing. If False, then all attributes will be
            inferred with some value.
        :type allow_none: Boolean
        :return: A completed instance
        :rtype: :ref:`Instance<instance-rep>`
        """
        self._sanity_check_instance(instance)
        temp_instance = {a: instance[a] for a in instance}
        concept = self._cobweb_categorize(temp_instance)

        for attr in concept.av_counts:
            if attr in temp_instance:
                continue
            val = concept.predict(attr, choice_fn, allow_none)
            if val is not None:
                temp_instance[attr] = val

        return temp_instance

    def categorize(self, instance):
        """
        Sort an instance in the categorization tree and return its resulting
        concept.

        The instance is passed down the categorization tree according to the
        normal cobweb algorithm except using only the best operator and without
        modifying nodes' probability tables. **This process does not modify the
        tree's knowledge** for a modifying version of labeling use the
        :meth:`CobwebTree.ifit` function

        :param instance: an instance to be categorized into the tree.
        :type instance: :ref:`Instance<instance-rep>`
        :return: A concept describing the instance
        :rtype: CobwebNode

        .. seealso:: :meth:`CobwebTree.cobweb`
        """
        self._sanity_check_instance(instance)
        return self._cobweb_categorize(instance)


class MultinomialCobwebNode(object):
    """
    A CobwebNode represents a concept within the knoweldge base of a particular
    :class:`CobwebTree`. Each node contains a probability table that can be
    used to calculate the probability of different attributes given the concept
    that the node represents.

    In general the :meth:`CobwebTree.ifit`, :meth:`CobwebTree.categorize`
    functions should be used to initially interface with the Cobweb knowledge
    base and then the returned concept can be used to calculate probabilities
    of certain attributes or determine concept labels.

    This constructor creates a CobwebNode with default values. It can also be
    used as a copy constructor to "deepcopy" a node, including all references
    to other parts of the original node's CobwebTree.

    :param otherNode: Another concept node to deepcopy.
    :type otherNode: CobwebNode
    """
    # a counter used to generate unique concept names.
    _counter = 0

    def __init__(self, otherNode=None):
        """Create a new CobwebNode"""
        self.concept_id = self.gensym()
        self.count = 0.0
        # self.squared_counts = 0.0
        # self.attr_counts = 0.0
        self.attr_counts = Counter()
        self.av_counts = defaultdict(Counter)
        self.children = []
        self.parent = None
        self.tree = None

        if otherNode:
            self.tree = otherNode.tree
            self.parent = otherNode.parent
            self.update_counts_from_node(otherNode)

            for child in otherNode.children:
                self.children.append(self.__class__(child))

    def log_prob_class_given_instance(self, instance):

        log_prob = 0;

        for attr in instance:
            hidden = attr[0] == '_';
            if hidden or attr not in self.tree.root.av_counts:
                continue

            alpha = self.tree.alpha(attr)

            num_vals = len(self.tree.attr_vals[attr])
            # num_vals = len(self.tree.root.av_counts[attr])

            for val in instance[attr]:
                if val not in self.tree.root.av_counts[attr]:
                    continue
            
                av_count = alpha
                if attr in self.av_counts and val in self.av_counts[attr]:
                    av_count += self.av_counts[attr][val]

                log_prob += instance[attr][val] * log((1.0 * av_count) / (self.attr_counts[attr] + num_vals * alpha))

        log_prob += log((1.0 * self.count) / self.tree.root.count)

        return log_prob;

    def increment_counts(self, instance):
        """
        Increment the counts at the current node according to the specified
        instance.

        :param instance: A new instances to incorporate into the node.
        :type instance: :ref:`Instance<instance-rep>`
        """
        self.count += 1
        for attr in instance:
            for val in instance[attr]:
                self.attr_counts[attr] += instance[attr][val]
                self.av_counts[attr][val] += instance[attr][val]

    def update_counts_from_node(self, node):
        """
        Increments the counts of the current node by the amount in the
        specified node.

        This function is used as part of copying nodes and in merging nodes.

        :param node: Another node from the same CobwebTree
        :type node: CobwebNode
        """
        self.count += node.count
        for attr in node.av_counts:
            self.attr_counts[attr] += node.attr_counts[attr]
            
            for val in node.av_counts[attr]:
                self.av_counts[attr][val] += node.av_counts[attr][val]

    def entropy_insert(self, instance):
        """
        Returns the expected correct guesses that would result from inserting
        the instance into the current node. 

        This operation can be used instead of inplace and copying because it
        only looks at the attr values used in the instance and reduces iteration.
        """
        info = 0
        for attr in set(self.av_counts).union(set(instance)):
            a_count = self.attr_counts[attr]
            if attr in instance:
                a_count += sum(instance[attr].values())

            vals = set(self.av_counts[attr])
            if attr in instance:
                vals = vals.union(set(instance[attr]))

            for val in vals:
                av_count = self.av_counts[attr][val]

                if attr in instance and val in instance[attr]:
                    av_count += instance[attr][val]

                p = ((av_count + self.tree.alpha(attr))/
                     (a_count + len(self.tree.attr_vals[attr]) * self.tree.alpha(attr)))
                info += p * log(p)

            num_missing = len(self.tree.attr_vals[attr]) - len(vals)
            if num_missing > 0 and self.tree.alpha(attr) > 0:
                p = (self.tree.alpha(attr) /
                     (a_count + len(self.tree.attr_vals[attr]) * self.tree.alpha(attr)))
                info += num_missing * p * log(p)
        
        return info

    def entropy_merge(self, other, instance):
        """
        Returns the expected correct guesses that would result from merging the
        current concept with the other concept and inserting the instance.

        This operation can be used instead of inplace and copying because it
        only looks at the attr values used in the instance and reduces iteration.
        """
        av_counts = defaultdict(Counter)
        attr_counts = Counter()
        for attr in self.av_counts:
            attr_counts[attr] += self.attr_counts[attr]

            for val in self.av_counts[attr]:
                av_counts[attr][val] += self.av_counts[attr][val]

        for attr in other.av_counts:
            attr_counts[attr] += other.attr_counts[attr]

            for val in other.av_counts[attr]:
                av_counts[attr][val] += other.av_counts[attr][val]


        for attr in instance:
            for val in instance[attr]:
                av_counts[attr][val] += instance[attr][val]
                attr_counts[attr] += instance[attr][val]

        info = 0
        for attr in av_counts:
            for val in av_counts[attr]:
                # p = av_counts[attr][val] / attr_counts[attr]
                p = ((av_counts[attr][val] + self.tree.alpha(attr))/
                     (attr_counts[attr] + len(self.tree.attr_vals[attr]) *
                      self.tree.alpha(attr)))
                info += p * log(p)
                # info += p * p

            num_missing = len(self.tree.attr_vals[attr]) - len(av_counts[attr])
            if num_missing > 0 and self.tree.alpha(attr) > 0:
                p = (self.tree.alpha(attr) / (attr_counts[attr] +
                                        len(self.tree.attr_vals[attr]) *
                                        self.tree.alpha(attr)))
                info += num_missing * p * log(p)
        
        return info


    def entropy(self):
        """
        Returns the entropy or the amount of information needed to encode distributions.

        This formula contains a smoothing capability that ensures all
        experienced attribute values have some probability mass.
        """
        info = 0
        for attr in self.av_counts:
            for val in self.av_counts[attr]:
                p = ((self.av_counts[attr][val] + self.tree.alpha(attr))/
                     (self.attr_counts[attr] + len(self.tree.attr_vals[attr]) *
                      self.tree.alpha(attr)))
                info += p * log(p)

            num_missing = len(self.tree.attr_vals[attr]) - len(self.av_counts[attr])
            if num_missing > 0 and self.tree.alpha(attr) > 0:
                p = (self.tree.alpha(attr) / (self.attr_counts[attr] +
                                        len(self.tree.attr_vals[attr]) *
                                        self.tree.alpha(attr)))
                info += num_missing * p * log(p)
        
        return info

    def category_utility(self):
        """
        Return the category utility of a particular division of a concept into
        its children.

        Category utility is always calculated in reference to a parent node and
        its own children. This is used as the heuristic to guide the concept
        formation process. Category Utility is calculated as:

        .. math::

            CU(\\{C_1, C_2, \\cdots, C_n\\}) = \\frac{1}{n} \\sum_{k=1}^n
            P(C_k) \\left[ \\sum_i \\sum_j P(A_i = V_{ij} | C_k)^2 \\right] -
            \\sum_i \\sum_j P(A_i = V_{ij})^2

        where :math:`n` is the numer of children concepts to the current node,
        :math:`P(C_k)` is the probability of a concept given the current node,
        :math:`P(A_i = V_{ij} | C_k)` is the probability of a particular
        attribute value given the concept :math:`C_k`, and :math:`P(A_i =
        V_{ij})` is the probability of a particular attribute value given the
        current node.

        In general this is used as an internal function of the cobweb algorithm
        but there may be times when it would be useful to call outside of the
        algorithm itself.

        :return: The category utility of the current node with respect to its
                 children.
        :rtype: float
        """
        if len(self.children) == 0:
            return 0.0

        child_correct_guesses = 0.0

        info_c = 0.0
        for child in self.children:
            p_of_child = child.count / self.count
            info_c -= p_of_child * log(p_of_child)
            child_correct_guesses += (p_of_child *
                                      child.entropy())

        return ((child_correct_guesses - self.entropy()) 
                # - info_c)
                / len(self.children))

    def get_best_operation(self, instance, best1, best2, best1_cu,
                           best_op=True, new_op=True, merge_op=True, split_op=True):
        """
        Given an instance, the two best children based on category utility and
        a set of possible operations, find the operation that produces the
        highest category utility, and then return the category utility and name
        for the best operation. In the case of ties, an operator is randomly
        chosen.

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

        if best_op:
            operations.append((best1_cu, random(), "best"))
        if new_op:
            operations.append((self.cu_for_new_child(instance), random(),
                               'new'))
        if merge_op and len(self.children) > 2 and best2:
            operations.append((self.cu_for_merge(best1, best2, instance),
                               random(), 'merge'))
        if split_op and len(best1.children) > 0:
            operations.append((self.cu_for_split(best1), random(), 'split'))

        operations.sort(reverse=True)
        # print(operations)
        best_op = (operations[0][0], operations[0][2])
        # print(best_op)
        return best_op

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

        relative_cus = [(((child.count + 1) * child.entropy_insert(instance)) -
                         (child.count * child.entropy()),
                         child.count, random(), child) for child in
                        self.children]
        relative_cus.sort(reverse=True)

        best1 = relative_cus[0][3]
        best1_cu = self.cu_for_insert(best1, instance)

        best2 = None
        if len(relative_cus) > 1:
            best2 = relative_cus[1][3]

        return best1_cu, best1, best2
        
    def cu_for_insert(self, child, instance):
        """
        Compute the category utility of adding the instance to the specified
        child.

        This operation does not actually insert the instance into the child it
        only calculates what the result of the insertion would be. For the
        actual insertion function see: :meth:`CobwebNode.increment_counts` This
        is the function used to determine the best children for each of the
        other operations.

        :param child: a child of the current node
        :type child: CobwebNode
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility of adding the instance to the given node
        :rtype: float

        .. seealso:: :meth:`CobwebNode.two_best_children` and
            :meth:`CobwebNode.get_best_operation`

        """
        child_correct_guesses = 0.0

        info_c = 0.0
        for c in self.children:
            if c == child:
                p_of_child = (c.count+1) / (self.count + 1)
                info_c -= p_of_child * log(p_of_child)
                child_correct_guesses += ((c.count+1) *
                                          c.entropy_insert(instance))
            else:
                p_of_child = (c.count) / (self.count + 1)
                info_c -= p_of_child * log(p_of_child)
                child_correct_guesses += (c.count *
                                          c.entropy())

        child_correct_guesses /= (self.count + 1)
        parent_correct_guesses = self.entropy_insert(instance)

        return ((child_correct_guesses - parent_correct_guesses) 
                # - info_c)
                # / 1)
                / len(self.children))

    def create_new_child(self, instance):
        """
        Create a new child (to the current node) with the counts initialized by
        the *given instance*.

        This is the operation used for creating a new child to a node and
        adding the instance to it.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: The new child
        :rtype: CobwebNode
        """
        new_child = self.__class__()
        new_child.parent = self
        new_child.tree = self.tree
        new_child.increment_counts(instance)
        self.children.append(new_child)
        return new_child

    def cu_for_new_child(self, instance):
        """
        Return the category utility for creating a new child using the
        particular instance.

        This operation does not actually create the child it only calculates
        what the result of creating it would be. For the actual new function
        see: :meth:`CobwebNode.create_new_child`.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility of adding the instance to a new child.
        :rtype: float

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        child_correct_guesses = 0.0

        info_c = 0.0
        for c in self.children:
            p_of_child = c.count / (self.count + 1)
            info_c -= p_of_child * log(p_of_child)
            child_correct_guesses += (c.count * c.entropy())

        # sum over all attr (at 100% prob) divided by num attr should be 1.
        # child_correct_guesses += 1
        new_child = self.__class__()
        new_child.parent = self
        new_child.tree = self.tree
        new_child.increment_counts(instance)
        child_correct_guesses += new_child.entropy()
        p_of_child = 1 / (self.count + 1)
        info_c -= p_of_child * log(p_of_child)

        child_correct_guesses /= (self.count + 1)
        parent_correct_guesses = self.entropy_insert(instance)

        return ((child_correct_guesses - parent_correct_guesses)
                # - info_c)
                # / 1)
                / (len(self.children)+1))


    def merge(self, best1, best2):
        """
        Merge the two specified nodes.

        A merge operation introduces a new node to be the merger of the the two
        given nodes. This new node becomes a child of the current node and the
        two given nodes become children of the new node.

        :param best1: The child of the current node with the best category
            utility
        :type best1: CobwebNode
        :param best2: The child of the current node with the second best
            category utility
        :type best2: CobwebNode
        :return: The new child node that was created by the merge
        :rtype: CobwebNode
        """
        new_child = self.__class__()
        new_child.parent = self
        new_child.tree = self.tree

        new_child.update_counts_from_node(best1)
        new_child.update_counts_from_node(best2)
        best1.parent = new_child
        best2.parent = new_child
        new_child.children.append(best1)
        new_child.children.append(best2)
        self.children.remove(best1)
        self.children.remove(best2)
        self.children.append(new_child)

        return new_child

    def cu_for_merge(self, best1, best2, instance):
        """
        Return the category utility for merging the two best children.

        This does not actually merge the two children it only calculates what
        the result of the merge would be. For the actual merge operation see:
        :meth:`CobwebNode.merge`

        :param best1: The child of the current node with the best category
            utility
        :type best1: CobwebNode
        :param best2: The child of the current node with the second best
            category utility
        :type best2: CobwebNode
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: The category utility that would result from merging best1 and
            best2.
        :rtype: float

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        child_correct_guesses = 0.0

        info_c = 0.0
        for c in self.children:
            if c == best1 or c == best2:
                continue

            p_of_child = c.count / (self.count + 1)
            info_c -= p_of_child * log(p_of_child)

            child_correct_guesses += (c.count *
                                      c.entropy())

        p_of_child = (best1.count + best2.count + 1) / (self.count + 1)
        info_c -= p_of_child * log(p_of_child)
        child_correct_guesses += ((best1.count + best2.count + 1) *
                                  best1.entropy_merge(best2, instance))

        child_correct_guesses /= (self.count + 1)
        parent_correct_guesses = self.entropy_insert(instance)

        return ((child_correct_guesses - parent_correct_guesses)
                # - info_c)
                # / 1)
                / (len(self.children)-1))
    

    def split(self, best):
        """
        Split the best node and promote its children

        A split operation removes a child node and promotes its children to be
        children of the current node. Split operations result in a recursive
        call of cobweb on the current node so this function does not return
        anything.

        :param best: The child node to be split
        :type best: CobwebNode
        """
        self.children.remove(best)
        for child in best.children:
            child.parent = self
            child.tree = self.tree
            self.children.append(child)

    def cu_for_split(self, best):
        """
        Return the category utility for splitting the best child.

        This does not actually split the child it only calculates what the
        result of the split would be. For the actual split operation see:
        :meth:`CobwebNode.split`. Unlike the category utility calculations for
        the other operations split does not need the instance because splits
        trigger a recursive call on the current node.

        :param best: The child of the current node with the best category
            utility
        :type best: CobwebNode
        :return: The category utility that would result from splitting best
        :rtype: float

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        child_correct_guesses = 0.0

        info_c = 0.0
        for c in self.children:
            if c == best:
                continue

            p_of_child = c.count / self.count
            info_c -= p_of_child * log(p_of_child)

            child_correct_guesses += (c.count * c.entropy())

        for c in best.children:
            p_of_child = c.count / self.count
            info_c -= p_of_child * log(p_of_child)
            child_correct_guesses += (c.count * c.entropy())

        child_correct_guesses /= self.count
        parent_correct_guesses = self.entropy()

        return ((child_correct_guesses - parent_correct_guesses)
                # - info_c)
                # / 1)
                / (len(self.children) - 1 + len(best.children)))

    def is_exact_match(self, instance):
        """
        Returns true if the concept exactly matches the instance.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: whether the instance perfectly matches the concept
        :rtype: boolean

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        for attr in set(instance).union(set(self.av_counts)):
            if attr[0] == '_':
                continue
            if attr in instance and attr not in self.av_counts:
                return False
            if attr in self.av_counts and attr not in instance:
                return False
            if attr in self.av_counts and attr in instance:
                for val in set(instance[attr]).union(set(self.av_counts[attr])):
                    if val in instance[attr] and val not in self.av_counts[attr]:
                        return False
                    if val in self.av_counts[attr] and val not in instance[attr]:
                        return False
                    if not instance[attr][val] == self.av_counts[attr][val]:
                        return False
        return True

    def __hash__(self):
        """
        The basic hash function. This hashes the concept name, which is
        generated to be unique across concepts.
        """
        return hash("CobwebNode" + str(self.concept_id))

    def gensym(self):
        """
        Generate a unique id and increment the class _counter.

        This is used to create a unique name for every concept. As long as the
        class _counter variable is never externally altered these keys will
        remain unique.

        """
        self.__class__._counter += 1
        return self.__class__._counter

    def __str__(self):

        """
        Call :meth:`CobwebNode.pretty_print`
        """
        return self.pretty_print()

    def pretty_print(self, depth=0):
        """
        Print the categorization tree

        The string formatting inserts tab characters to align child nodes of
        the same depth.

        :param depth: The current depth in the print, intended to be called
                      recursively
        :type depth: int
        :return: a formated string displaying the tree and its children
        :rtype: str
        """
        ret = str(('\t' * depth) + "|-" + str(self.av_counts) + ":" +
                  str(self.count) + '\n')

        for c in self.children:
            ret += c.pretty_print(depth+1)

        return ret

    def depth(self):
        """
        Returns the depth of the current node in its tree

        :return: the depth of the current node in its tree
        :rtype: int
        """
        if self.parent:
            return 1 + self.parent.depth()
        return 0

    def is_parent(self, other_concept):
        """
        Return True if this concept is a parent of other_concept

        :return: ``True`` if this concept is a parent of other_concept else
                 ``False``
        :rtype: bool
        """
        temp = other_concept
        while temp is not None:
            if temp == self:
                return True
            try:
                temp = temp.parent
            except Exception:
                print(temp)
                assert False
        return False

    def num_concepts(self):
        """
        Return the number of concepts contained below the current node in the
        tree.

        When called on the :attr:`CobwebTree.root` this is the number of nodes
        in the whole tree.

        :return: the number of concepts below this concept.
        :rtype: int
        """
        children_count = 0
        for c in self.children:
            children_count += c.num_concepts()
        return 1 + children_count

    def output_json(self):
        return json.dumps(self.output_dict())

    def output_dict(self):
        """
        Outputs the categorization tree in JSON form

        :return: an object that contains all of the structural information of
                 the node and its children
        :rtype: obj
        """
        output = {}
        output['name'] = "Concept" + str(self.concept_id)
        output['size'] = self.count
        output['children'] = []

        temp = {}
        temp['_basic_cu'] = {"#ContinuousValue#": {'mean': self.entropy(),
                                                   'std': 1, 'n': 1}}
        temp['_basic_cu2'] = {"#ContinuousValue#": {'mean': self.category_utility(),
                                                   'std': 1, 'n': 1}}
        for attr in self.av_counts:
            for value in self.av_counts[attr]:
                temp[str(attr)] = {str(value): self.av_counts[attr][value] for
                                   value in self.av_counts[attr]}

        for child in self.children:
            output["children"].append(child.output_dict())

        output['counts'] = temp

        return output

    def get_weighted_values(self, attr, allow_none=True):
        """
        Return a list of weighted choices for an attribute based on the node's
        probability table.

        This calculation will include an option for the change that an
        attribute is missing from an instance all together. This is useful for
        probability and sampling calculations. If the attribute has never
        appeared in the tree then it will return a 100% chance of None.

        :param attr: an attribute of an instance
        :type attr: :ref:`Attribute<attributes>`
        :param allow_none: whether attributes in the nodes probability table
            can be inferred to be missing. If False, then None will not be
            cosidered as a possible value.
        :type allow_none: Boolean
        :return: a list of weighted choices for attr's value
        :rtype: [(:ref:`Value<values>`, float), (:ref:`Value<values>`, float),
                 ...]
        """
        choices = []
        if attr not in self.av_counts:
            choices.append((None, 1.0))
            return choices

        val_count = 0
        for val in self.av_counts[attr]:
            count = self.av_counts[attr][val]
            choices.append((val, count / self.count))
            val_count += count

        if allow_none:
            choices.append((None, ((self.count - val_count) / self.count)))

        return choices

    def predict(self, attr, choice_fn="most likely", allow_none=True):
        """
        Predict the value of an attribute, using the specified choice function
        (either the "most likely" value or a "sampled" value).

        :param attr: an attribute of an instance.
        :type attr: :ref:`Attribute<attributes>`
        :param choice_fn: a string specifying the choice function to use,
            either "most likely" or "sampled".
        :type choice_fn: a string
        :param allow_none: whether attributes not in the instance can be
            inferred to be missing. If False, then all attributes will be
            inferred with some value.
        :type allow_none: Boolean
        :return: The most likely value for the given attribute in the node's
                 probability table.
        :rtype: :ref:`Value<values>`
        """
        if choice_fn == "most likely" or choice_fn == "m":
            choose = most_likely_choice
        elif choice_fn == "sampled" or choice_fn == "s":
            choose = weighted_choice
        else:
            raise Exception("Unknown choice_fn")

        if attr not in self.av_counts:
            return None

        choices = self.get_weighted_values(attr, allow_none)
        val = choose(choices)
        return val

    def probability(self, attr, val):
        """
        Returns the probability of a particular attribute value at the current
        concept. This takes into account the possibilities that an attribute
        can take any of the values available at the root, or be missing.

        If you you want to check if the probability that an attribute is
        missing, then check for the probability that the val is ``None``.

        :param attr: an attribute of an instance
        :type attr: :ref:`Attribute<attributes>`
        :param val: a value for the given attribute or None
        :type val: :ref:`Value<values>`
        :return: The probability of attr having the value val in the current
            concept.
        :rtype: float
        """
        if val is None:
            c = 0.0
            if attr in self.av_counts:
                c = sum([self.av_counts[attr][v] for v in
                         self.av_counts[attr]])
            return (self.count - c) / self.count

        if attr in self.av_counts and val in self.av_counts[attr]:
            return self.av_counts[attr][val] / self.count

        return 0.0

    def log_likelihood(self, child_leaf):
        """
        Returns the log-likelihood of a leaf contained within the current
        concept. Note, if the leaf contains multiple instances, then it is
        treated as if it contained just a single instance (this function is
        just called multiple times for each instance in the leaf).
        """
        ll = 0

        for attr in set(self.av_counts).union(set(child_leaf.av_counts)):
            if attr[0] == "_":
                continue
            vals = set([None])
            if attr in self.av_counts:
                vals.update(self.av_counts[attr])
            if attr in child_leaf.av_counts:
                vals.update(child_leaf.av_counts[attr])

            for val in vals:
                op = child_leaf.probability(attr, val)
                if op > 0:
                    p = self.probability(attr, val) * op
                    if p >= 0:
                        ll += log(p)
                    else:
                        raise Exception("Should always be greater than 0")

        return ll

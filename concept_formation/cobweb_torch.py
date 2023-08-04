"""
The Cobweb module contains the :class:`CobwebTree` and :class:`CobwebNode`
classes which are used to achieve the basic Cobweb functionality.
"""
import json
import math
from random import shuffle
from random import random
from math import log
from math import isclose
from collections import defaultdict
from collections import Counter

import torch

from concept_formation.utils import weighted_choice
from concept_formation.utils import most_likely_choice

pi_tensor = torch.tensor(math.pi)

class CobwebTorchTree(object):
    """
    The CobwebTree contains the knoweldge base of a partiucluar instance of the
    cobweb algorithm and can be used to fit and categorize instances.
    """

    def __init__(self, shape, use_mutual_info=True, acuity_cutoff=False,
                 use_ll_for_categorize=True, prior_var=None, alpha=0.01,
                 device=None):
        """
        The tree constructor.
        """
        self.device = device
        self.use_mutual_info = use_mutual_info
        self.acuity_cutoff = acuity_cutoff
        self.use_ll_for_categorize = use_ll_for_categorize
        self.shape = shape
        self.alpha = alpha

        self.prior_var = prior_var
        if prior_var is None:
            self.prior_var = 1 / (4 * pi_tensor)

        self.clear()

    def clear(self):
        """
        Clears the concepts of the tree.
        """
        self.root = CobwebTorchNode(shape=self.shape, device=self.device)
        self.root.tree = self
        self.labels = {}
        self.reverse_labels = {}

    def __str__(self):
        return str(self.root)

    def dump_json(self):
        return self.root.output_json()

    # TODO UPDATE TO SAVE / LOAD LABELS
    def load_json(self, json_string):
        data = json.loads(json_string)
        mean = torch.tensor(data['mean'], dtype=torch.float)
        self.shape = mean.shape
        self.root = self.load_helper(data)

    # TODO UPDATE TO SAVE / LOAD LABELS
    def load_helper(self, json_node):
        node = CobwebTorchNode(self.shape, device=self.device)
        node.tree = self
        node.count = torch.tensor(json_node['size'], dtype=torch.float, device=self.device)
        node.mean = torch.tensor(json_node['mean'], dtype=torch.float, device=self.device)
        node.meanSq = torch.tensor(json_node['meanSq'], dtype=torch.float, device=self.device)
        node.children = [self.load_helper(child) for child in json_node['children']]
        return node

    def ifit(self, instance, label=None):
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
        if label not in self.labels:
            idx = len(self.labels)
            self.labels[label] = idx
            self.reverse_labels[idx] = label

        return self.cobweb(instance, label)

    def fit(self, instances, labels=None, iterations=1, randomize_first=True):
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
        if labels is None:
            labels = [None for i in range(len(instances))]
        
        instance_labels = [(inst, labels[i]) for i, inst in
                           enumerate(instances)]

        for x in range(iterations):
            if x == 0 and randomize_first:
                shuffle(instances)
            for inst, label in instances:
                self.ifit(inst, label)
            shuffle(instances)

    def cobweb(self, instance, label):
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
            if not current.children and (current.is_exact_match(instance, label) or
                                         current.count == 0):
                # print("leaf match")
                current.increment_counts(instance, label)
                break

            elif not current.children:
                # print("fringe split")
                new = CobwebTorchNode(shape=self.shape, device=self.device, otherNode=current)
                current.parent = new
                new.children.append(current)

                if new.parent:
                    new.parent.children.remove(current)
                    new.parent.children.append(new)
                else:
                    self.root = new

                new.increment_counts(instance, label)
                current = new.create_new_child(instance, label)
                break

            else:
                best1_pu, best1, best2 = current.two_best_children(instance,
                                                                   label)
                _, best_action = current.get_best_operation(instance, label, best1,
                                                            best2, best1_pu)

                # print(best_action)
                if best_action == 'best':
                    current.increment_counts(instance, label)
                    current = best1
                elif best_action == 'new':
                    current.increment_counts(instance, label)
                    current = current.create_new_child(instance, label)
                    break
                elif best_action == 'merge':
                    current.increment_counts(instance, label)
                    new_child = current.merge(best1, best2)
                    current = new_child
                elif best_action == 'split':
                    current.split(best1)
                else:
                    raise Exception('Best action choice "' + best_action +
                                    '" not a recognized option. This should be'
                                    ' impossible...')

        return current

    def _cobweb_categorize(self, instance, label):
        """
        A cobweb specific version of categorize, not intended to be
        externally called.

        .. seealso:: :meth:`CobwebTree.categorize`
        """
        current = self.root
        
        while True:
            if (len(current.children) == 0):
                return current

            parent = current
            current = None
            best_score = None

            for child in parent.children:
                if self.use_ll_for_categorize:
                    score = child.log_prob_class_given_instance(instance, label)
                else:
                    score = (child.count * child.score() -
                             (child.count + 1) * child.score_insert(instance, label))
            
                if ((current is None) or ((best_score is None) or (score > best_score))):
                    best_score = score
                    current = child

    def categorize(self, instance, label=None):
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
        return self._cobweb_categorize(instance, label)


class CobwebTorchNode(object):
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

    def __init__(self, shape, device=None, otherNode=None):
        """Create a new CobwebNode"""
        self.concept_id = self.gensym()
        self.count = torch.tensor(0.0, dtype=torch.float, device=device)
        self.mean = torch.zeros(shape, dtype=torch.float, device=device)
        self.meanSq = torch.zeros(shape, dtype=torch.float, device=device)
        self.label_counts = torch.tensor([], dtype=torch.float, device=device)
        self.total_label_count = torch.tensor(0, dtype=torch.float, device=device)
        self.children = []
        self.parent = None
        self.tree = None

        if otherNode:
            self.tree = otherNode.tree
            self.parent = otherNode.parent
            self.update_counts_from_node(otherNode)

            for child in otherNode.children:
                self.children.append(CobwebTorchNode(shape=self.tree.shape, device=self.tree.device, otherNode=child))

    def update_label_count_size(self):
        if self.label_counts.shape[0] < len(self.tree.labels):
            num_new = len(self.tree.labels) - self.label_counts.shape[0]
            new_counts = (self.tree.alpha + torch.zeros(num_new,
                                                        dtype=torch.float,
                                                        device=self.tree.device))
            self.label_counts = torch.cat((self.label_counts, new_counts))
            self.total_label_count += new_counts.sum()

    def increment_counts(self, instance, label):
        """
        Increment the counts at the current node according to the specified
        instance.

        :param instance: A new instances to incorporate into the node.
        :type instance: :ref:`Instance<instance-rep>`
        """
        self.count += 1
        delta = instance - self.mean
        self.mean += delta / self.count
        self.meanSq += delta * (instance - self.mean)

        self.update_label_count_size()

        if label is not None:
            self.label_counts[self.tree.labels[label]] += 1
            self.total_label_count += 1

    def update_counts_from_node(self, other):
        """
        Increments the counts of the current node by the amount in the
        specified node.

        This function is used as part of copying nodes and in merging nodes.

        :param node: Another node from the same CobwebTree
        :type node: CobwebNode
        """
        delta = other.mean - self.mean
        self.meanSq = (self.meanSq + other.meanSq + delta * delta *
                       ((self.count * other.count) / (self.count + other.count)))
        self.mean = ((self.count * self.mean + other.count * other.mean) /
                     (self.count + other.count))
        self.count += other.count

        self.update_label_count_size()
        other.update_label_count_size()

        # alpha is added to all the counts, so need to subtract from other to
        # not double count
        self.label_counts += (other.label_counts - self.tree.alpha)
        self.total_label_count += (other.total_label_count -
                                   other.label_counts.shape[0] * self.tree.alpha)

    @property
    def std(self):
        return torch.sqrt(self.var)

    @property
    def var(self):
        if self.tree.acuity_cutoff:
            return torch.clamp(self.meanSq / self.count, self.tree.prior_var) # with cutoff
        else:
            return self.meanSq / self.count + self.tree.prior_var # with adjustment

    def log_prob_class_given_instance(self, instance, label):
        log_prob = self.log_prob(instance, label)
        log_prob += torch.log(self.count) - torch.log(self.tree.root.count)
        return log_prob

    def log_prob(self, instance, label):
        var = self.var
        log_prob = (0.5 * torch.log(var) + 0.5 * torch.log(2 * pi_tensor) +
                     0.5 * torch.square(instance - self.mean) / var).sum()

        self.update_label_count_size()
        if label is not None:
            if label not in self.tree.labels:
                log_prob += (torch.log(selt.tree.alpha) -
                             torch.log(self.total_label_count +
                                       self.tree.alpha))
            elif self.total_label_count > 0:
                log_prob += (torch.log(self.label_counts[self.tree.labels[label]]) -
                             torch.log(self.total_label_count))

        return log_prob

    def score_insert(self, instance, label):
        """
        Returns the score that would result from inserting the instance into
        the current node. 

        This operation can be used instead of inplace and copying because it
        only looks at the attr values used in the instance and reduces iteration.
        """
        count = self.count + 1
        delta = instance - self.mean
        mean = self.mean + delta / count
        meanSq = self.meanSq + delta * (instance - mean)

        # hopefully cheap if already updated.
        self.update_label_count_size()

        label_counts = self.label_counts.clone()
        total_label_count = self.total_label_count.clone()
        if label is not None:
            label_counts[self.tree.labels[label]] += 1
            total_label_count += 1

        if self.tree.acuity_cutoff:
            var = torch.clamp(meanSq / count, self.tree.prior_var) # with cutoff
        else:
            var = meanSq / count + self.tree.prior_var # with adjustment

        std = torch.sqrt(var)

        if (self.tree.use_mutual_info):
            score = (0.5 * torch.log(2 * pi_tensor * var) + 0.5).sum()
        else:
            score = -(1 / (2 * torch.sqrt(pi_tensor) * std)).sum()

        if self.total_label_count > 0:
            p_label = label_counts / total_label_count
            if (self.tree.use_mutual_info):
                score += (-p_label * torch.log(p_label)).sum()
            else:
                score += (-p_label * p_label).sum()

        return score

    def score_merge(self, other, instance, label):
        """
        Returns the expected correct guesses that would result from merging the
        current concept with the other concept and inserting the instance.

        This operation can be used instead of inplace and copying because it
        only looks at the attr values used in the instance and reduces iteration.
        """
        delta = other.mean - self.mean
        meanSq = (self.meanSq + other.meanSq + delta * delta *
                  ((self.count * other.count) / (self.count + other.count)))
        mean = ((self.count * self.mean + other.count * other.mean) /
                (self.count + other.count))
        count = self.count + other.count

        count = count + 1
        delta = instance - mean
        mean += delta / count
        meanSq += delta * (instance - mean)

        # hopefully cheap if already updated.
        self.update_label_count_size()
        other.update_label_count_size()

        label_counts = self.label_counts.clone()
        total_label_count = self.total_label_count.clone()

        label_counts += (other.label_counts - self.tree.alpha)
        total_label_count += (other.total_label_count -
                              other.label_counts.shape[0] * self.tree.alpha)

        if label is not None:
            label_counts[self.tree.labels[label]] += 1
            total_label_count += 1

        if self.tree.acuity_cutoff:
            var = torch.clamp(meanSq / count, self.tree.prior_var) # with cutoff
        else:
            var = meanSq / count + self.tree.prior_var # with adjustment

        std = torch.sqrt(var)

        if (self.tree.use_mutual_info):
            score = (0.5 * torch.log(2 * pi_tensor * var) + 0.5).sum()
        else:
            score = -(1 / (2 * torch.sqrt(pi_tensor) * std)).sum()

        if self.total_label_count > 0:
            p_label = label_counts / total_label_count
            if (self.tree.use_mutual_info):
                score += (-p_label * torch.log(p_label)).sum()
            else:
                score += (-p_label * p_label).sum()

        return score

    def get_basic(self):
        """
        Climbs up the tree from the current node (probably a leaf),
        computes the category utility score, and returns the node with
        the highest score.
        """
        curr = self
        best = self
        best_cu = self.category_utility()

        while curr.parent:
            curr = curr.parent
            curr_cu = curr.category_utility()
            if curr_cu > best_cu:
                best = curr
                best_cu = curr_cu

        return best

    def get_best(self, instance, label=None):
        """
        Climbs up the tree from the current node (probably a leaf),
        computes the category utility score, and returns the node with
        the highest score.
        """
        curr = self
        best = self
        best_ll = self.log_prob_class_given_instance(instance, label)

        while curr.parent:
            curr = curr.parent
            curr_ll = curr.log_prob_class_given_instance(instance, label)
            if curr_ll > best_ll:
                best = curr
                best_ll = curr_ll

        return best

    def category_utility(self):
        p_of_c = self.count / self.tree.root.count
        return p_of_c * (self.tree.root.score() - self.score())

    def score(self):
        """
        Returns the number of correct guesses that are expected from the given
        concept.

        This is the sum of the probability of each attribute value squared.
        This function is used in calculating category utility.

        :return: the number of correct guesses that are expected from the given
                 concept.
        :rtype: float
        """
        self.update_label_count_size()

        if (self.tree.use_mutual_info):
            score = (0.5 * torch.log(2 * pi_tensor * self.var) + 0.5).sum()
        else:
            score = -(1 / (2 * torch.sqrt(pi_tensor) * self.std)).sum()

        if self.total_label_count > 0:
            p_label = self.label_counts / self.total_label_count
            if (self.tree.use_mutual_info):
                score += (-p_label * torch.log(p_label)).sum()
            else:
                score += (-p_label * p_label).sum()

        return score

    def partition_utility(self):
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

        child_score = 0.0

        for child in self.children:
            p_of_child = child.count / self.count
            child_score += p_of_child * child.score()

        return ((self.score() - child_score) / len(self.children))

    def get_best_operation(self, instance, label, best1, best2, best1_pu):
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

        operations.append((best1_pu, random(), "best"))
        operations.append((self.pu_for_new_child(instance, label), random(), 'new'))
        if len(self.children) > 2 and best2:
            operations.append((self.pu_for_merge(best1, best2, instance, label),
                               random(), 'merge'))
        if len(best1.children) > 0:
            operations.append((self.pu_for_split(best1), random(), 'split'))

        operations.sort(reverse=True)
        # print(operations)
        best_op = (operations[0][0], operations[0][2])
        # print(best_op)
        return best_op

    def two_best_children(self, instance, label):
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

        relative_pus = [(child.count * child.score() -
                         (child.count + 1) * child.score_insert(instance, label),
                         child.count, random(), child) for child in
                        self.children]
        relative_pus.sort(reverse=True)

        best1 = relative_pus[0][3]
        best1_pu = self.pu_for_insert(best1, instance, label)

        best2 = None
        if len(relative_pus) > 1:
            best2 = relative_pus[1][3]

        return best1_pu, best1, best2
        
    def pu_for_insert(self, child, instance, label):
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
        children_score = 0.0

        for c in self.children:
            if c == child:
                p_of_child = (c.count + 1) / (self.count + 1)
                children_score += p_of_child * c.score_insert(instance, label)
            else:
                p_of_child = (c.count) / (self.count + 1)
                children_score += p_of_child * c.score()

        return (self.score_insert(instance, label) - children_score) / len(self.children)

    def create_new_child(self, instance, label):
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
        new_child = CobwebTorchNode(shape=self.tree.shape, device=self.tree.device)
        new_child.parent = self
        new_child.tree = self.tree
        new_child.increment_counts(instance, label)
        self.children.append(new_child)
        return new_child

    def pu_for_new_child(self, instance, label):
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
        children_score = 0.0

        for c in self.children:
            p_of_child = c.count / (self.count + 1)
            children_score += p_of_child * c.score()

        new_child = CobwebTorchNode(shape=self.tree.shape, device=self.mean.device);
        new_child.parent = self
        new_child.tree = self.tree
        new_child.increment_counts(instance, label)
        p_of_child = 1.0 / (self.count + 1)
        children_score += p_of_child * new_child.score()

        return (self.score_insert(instance, label) - children_score) / (len(self.children) + 1)


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
        new_child = CobwebTorchNode(shape=self.tree.shape, device=self.tree.device)
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

    def pu_for_merge(self, best1, best2, instance, label):
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
        children_score = 0.0

        for c in self.children:
            if c == best1 or c == best2:
                continue

            p_of_child = c.count / (self.count + 1)
            children_score += p_of_child * c.score()

        p_of_child = (best1.count + best2.count + 1) / (self.count + 1)
        children_score += p_of_child * best1.score_merge(best2, instance, label)

        return (self.score_insert(instance, label) - children_score) / (len(self.children) - 1)
    

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

    def pu_for_split(self, best):
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
        children_score = 0.0

        for c in self.children:
            if c == best:
                continue
            p_of_child = c.count / self.count
            children_score += p_of_child * c.score()

        for c in best.children:
            p_of_child = c.count / self.count
            children_score += p_of_child * c.score()

        return ((self.score() - children_score) /
                (len(self.children) - 1 + len(best.children)))

    def is_exact_match(self, instance, label):
        """
        Returns true if the concept exactly matches the instance.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: whether the instance perfectly matches the concept
        :rtype: boolean

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        self.update_label_count_size()

        if label is not None and self.total_label_count == 0:
            return False

        if label is None and self.total_label_count > 0:
            return False

        if self.total_label_count > 0:
            label_counts = self.label_counts - self.tree.alpha
            p_labels = label_counts / label_counts.sum()

            if not math.isclose(p_labels[self.tree.labels[label]].item(), 1.0):
                return False

        std = torch.sqrt(self.meanSq / self.count)
        if not torch.isclose(std, torch.zeros(std.shape,device=self.tree.device)).all():
            return False
        return torch.isclose(instance, self.mean).all()

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
        ret = str(('\t' * depth) + "|-" + str(self.mean) + ":" +
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

    def visualize(self):
        from matplotlib import pyplot as plt
        plt.imshow(self.mean.numpy())
        plt.show()

    def output_dict(self):
        """
        Outputs the categorization tree in JSON form

        :return: an object that contains all of the structural information of
                 the node and its children
        :rtype: obj
        """
        self.update_label_count_size()

        output = {}
        output['name'] = "Concept" + str(self.concept_id)
        output['size'] = self.count.item()
        output['children'] = []

        temp = {}
        temp['_category_utility'] = {"#ContinuousValue#": {'mean': self.category_utility().item(), 'std': 1, 'n': 1}}

        if self.total_label_count > 0:
            temp['label'] = {label: self.label_counts[self.tree.labels[label]].item() for label in self.tree.labels}

        for child in self.children:
            output["children"].append(child.output_dict())

        output['counts'] = temp
        output['mean'] = self.mean.tolist()
        output['meanSq'] = self.meanSq.tolist()

        return output

    def predict(self, most_likely=True):
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
        self.update_label_count_size()

        if most_likely:
            label = None
            if self.total_label_count > 0:
                label = self.tree.reverse_labels[self.label_counts.argmax().item()]
            return self.mean.detach().clone(), label
        else:
            label = None
            if self.total_label_count > 0:
                p_labels = label_counts / label_counts.sum()
                label = self.tree.reverse_labels[torch.multinomial(p_labels, 1).item()]
            return torch.normal(self.mean, self.std), label

from cProfile import run
from collections import Counter
from tkinter import N
# from multiprocess import Pool
from tqdm import tqdm
import pickle
from os.path import dirname, join
from sys import setrecursionlimit
from os import listdir
import resource
from time import time
import random
from itertools import chain

from concept_formation.utils import random_tiebreaker
from concept_formation.cobweb import CobwebNode
from concept_formation.cobweb import CobwebTree
from concept_formation.utils import skip_slice, oslice
from concept_formation.utils import tiebreak_top_2
from visualize import visualize
from concept_formation.preprocess_text import load_text, stop_words
from concept_formation.training_and_testing import create_questions, generate_ms_sentence_variant_synonyms

TREE_RECURSION = 0x10000

# May segfault without this line. 0x100 is a guess at the size of stack frame.
try:
    resource.setrlimit(resource.RLIMIT_STACK,
                       [0x100 * TREE_RECURSION, resource.RLIM_INFINITY])
except ValueError:
    print(Warning("Warning: Saving this model may result in a segfault"))
setrecursionlimit(TREE_RECURSION)

SAVE = False
LOAD = False
MODELS_PATH = join(dirname(__file__), 'saved_models')
MODEL_SAVE_LOCATION = join(MODELS_PATH, 'saved_model_%s' % time())
if LOAD:
    for option_num, s in enumerate(listdir(MODELS_PATH)):
        print('%s:' % option_num, s)
    index = int(input('Which model would you like to load? '))
    MODEL_LOAD_LOCATION = join(MODELS_PATH, listdir(MODELS_PATH)[index])
print(listdir(MODELS_PATH)[0])
run

random.seed(16)
minor_key = '#MinorCtxt#'
major_key = '#MajorCtxt#'
anchor_key = 'anchor'

# Debugging output:
word_to_leaf = {}


def get_path(node):
    while node:
        yield node
        node = node.parent


class ContextualCobwebTree(CobwebTree):

    def __init__(self, minor_window, major_window):
        """
        Note window only specifies how much context to add to each side,
        doesn't include the anchor word.

        E.g., to get a window with 2 before and 2 words after the anchor, then
        set the window=2
        """
        super().__init__()
        self.root = ContextualCobwebNode()
        self.root.tree = self

        self.minor_window = minor_window
        self.major_window = major_window
        self.anchor_weight = 10
        self.minor_weight = 3
        self.major_weight = 1
        print(self.anchor_weight, self.minor_weight, self.major_weight)

    def clear(self):
        """
        Clears the concepts of the tree.
        """
        self.root = ContextualCobwebNode()
        self.root.tree = self

    def _sanity_check_instance(self, instance):
        for attr in instance:
            try:
                hash(attr)
                attr[0]
            except Exception:
                raise ValueError('Invalid attribute: '+str(attr) +
                                 ' of type: '+str(type(attr)) +
                                 ' in instance: '+str(instance) +
                                 ',\n'+type(self).__name__ +
                                 ' only works with hashable ' +
                                 'and subscriptable attributes' +
                                 ' (e.g., strings).')
            try:
                if not isinstance(instance[attr], str) and attr[0] != '_':
                    map(hash, instance[attr])
            except Exception:
                raise ValueError('Invalid value: '+str(instance[attr]) +
                                 ' of type: '+str(type(instance[attr])) +
                                 ' in instance: '+str(instance) +
                                 ',\n'+type(self).__name__ +
                                 ' only works with hashable values.')
            if instance[attr] is None:
                raise ValueError("Attributes with value None should"
                                 " be manually removed.")

    def fit_to_text_wo_stopwords(self, text):
        """filters stop words here"""
        ctxt_nodes = []
        instance_cache = []
        text = [word for word in text if word not in stop_words]

        for anchor_idx, anchor_wd in enumerate(text):#tqdm(text)):
            while ((len(ctxt_nodes) < anchor_idx + self.major_window + 1) and
                   len(ctxt_nodes) < len(text)):
                instance_cache.append(self.create_instance(len(ctxt_nodes),
                                                           text[len(ctxt_nodes)], ctxt_nodes, ignore=(anchor_idx,)))
                ctxt_nodes.append(self.categorize(instance_cache[-1]))

            for _ in range(2):
                for i in range(2 * self.major_window + 1):
                    idx = anchor_idx + i - self.major_window
                    if not (0 <= idx < len(text)):
                        continue

                    new_instance = self.create_instance(
                        idx, text[idx], ctxt_nodes, ignore=(anchor_idx,))
                    ctxt_nodes[idx] = self.categorize(new_instance)

            ctxt_nodes[anchor_idx] = self.ifit(self.create_instance(
                anchor_idx, anchor_wd, ctxt_nodes, ignore=()))

            word_to_leaf.setdefault(anchor_wd, set())
            word_to_leaf[anchor_wd].add(ctxt_nodes[anchor_idx])

        return ctxt_nodes

    def resort(self, idx, word, ctxt_nodes, instance_cache, ignore=()):
        # verify_structure(self.root)
        # if not all(map(in_tree, ctxt_nodes)):
        #     print(ctxt_nodes)
        #     assert False
        self.delete_node(ctxt_nodes[idx])
        new_instance = self.create_instance(
            idx, word, ctxt_nodes, ignore=ignore)
        leaf = ctxt_nodes[idx]
        leaf.decrement_counts(instance_cache[idx])
        leaf.increment_counts(new_instance)

        ctxt_nodes[idx] = self.ifit_leaf(leaf, new_instance, ctxt_nodes)
        instance_cache[idx] = new_instance
        # assert in_tree(ctxt_nodes[idx])

    def ifit_leaf(self, leaf, instance, ctxt_nodes):
        """Categorizes instances, incrementing counts from the leaf"""
        current = self.root

        while current:
            # the current.count == 0 here is for the initially empty tree.
            if not current.children and (current.is_exact_match(instance) or
                                         current.count == 0):
                print("leaf match")
                self.root.replace_node_as_context(leaf, current)
                for i, node in enumerate(ctxt_nodes):
                    if node == leaf:
                        ctxt_nodes[i] = current
                word = next(iter(leaf.av_counts[anchor_key]))
                if word in word_to_leaf:  # Might not be here since words are only added when solidified
                    for i, node in enumerate(word_to_leaf[word]):
                        if node == leaf:
                            ctxt_nodes[i] = current
                current.update_counts_from_node(leaf)
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

                new.update_counts_from_node(leaf)
                new.children.append(leaf)
                leaf.parent = new
                current = leaf
                break

            else:
                best1_cu, best1, best2 = current.two_best_children(instance)
                _, best_action = current.get_best_operation(instance, best1,
                                                            best2, best1_cu)

                # print(best_action)
                if best_action == 'best':
                    current.update_counts_from_node(leaf)
                    current = best1
                elif best_action == 'new':
                    current.update_counts_from_node(leaf)
                    current.children.append(leaf)
                    leaf.parent = current
                    current = leaf
                    break
                elif best_action == 'merge':
                    current.update_counts_from_node(leaf)
                    new_child = current.merge(best1, best2)
                    current = new_child
                elif best_action == 'split':
                    current.split(best1)
                else:
                    raise Exception('Best action choice "' + best_action +
                                    '" not a recognized option. This should be'
                                    ' impossible...')

        return current

    def delete_node(self, node):
        # TODO Slight incorrect structure when deleting 2-leaf tree
        node.parent.decrement_all_counts(node)
        node.parent.children.remove(node)
        if len(node.parent.children) == 1 and node.parent.parent is not None:
            # Undo fringe split
            node.parent.parent.split(node.parent)

    def create_instance(self, anchor_idx, anchor_word, ctxts, ignore=(),
                        filter_stop_for_minor=False):
        major_context = list(filter(None, self.surrounding(
            ctxts, anchor_idx, self.major_window, ignore=ignore)))
        if filter_stop_for_minor:
            raise NotImplementedError
        else:
            minor_context = list(filter(None, self.surrounding(
                ctxts, anchor_idx, self.minor_window, ignore=ignore)))

        return {minor_key: minor_context,
                major_key: major_context,
                anchor_key: anchor_word, }
        # '_idx': anchor_idx} DOES NOT HANDLE HIDDEN ATTRS

    def similarity_categorize(self, instance):
        current = self.root

        def similarity(node):
            """Average probability"""
            correct_guesses = 0
            if anchor_key in instance:
                correct_guesses += node.av_counts[anchor_key].get(
                    instance[anchor_key], 0) * self.anchor_weight
            if minor_key in instance:
                correct_guesses += sum(
                    node.av_counts[minor_key].get(ctxt, 0)
                    for ctxt in instance[minor_key]) * self.minor_weight
            if major_key in instance:
                correct_guesses += sum(
                    node.av_counts[major_key].get(ctxt, 0)
                    for ctxt in instance[major_key]) * self.major_weight

            return correct_guesses  # / node.count

        while current:
            if not current.children:
                return current

            best = max(current.children, key=similarity)
            current = best
        return current

    def surrounding(self, sequence, center, dist, ignore=()):
        if ignore:
            return list(
                oslice(sequence, max(0, center-dist), *sorted((*(x for x in ignore if 0 < x < center+dist+1), center)), center+dist+1))
        return list(
            skip_slice(sequence, max(0, center-dist), center+dist+1, center))

    def guess_missing(self, text, options, options_needed=1,
                      filter_stop_for_minor=False):
        text = [word for word in text if word not in stop_words]

        assert len(options) >= options_needed and not filter_stop_for_minor
        missing_idx = text.index(None)
        ctxt_nodes = []

        for anchor_idx, anchor_wd in enumerate(text):
            # if anchor_idx > missing_idx:
            #     break

            while ((len(ctxt_nodes) < anchor_idx + self.major_window + 1) and
                    len(ctxt_nodes) < len(text)):
                if len(ctxt_nodes) == missing_idx:
                    ctxt_nodes.append(None)
                    continue
                ctxt_nodes.append(self.categorize(
                    self.create_instance(len(ctxt_nodes),
                                         text[len(ctxt_nodes)], ctxt_nodes, ignore=(missing_idx, anchor_idx,))))

            if anchor_idx == missing_idx:
                continue

            for _ in range(2):
                for i in range(2 * self.major_window + 1):
                    idx = anchor_idx + i - self.major_window
                    if idx < 0 or idx >= len(text) or idx == missing_idx:
                        continue

                    instance = self.create_instance(
                        idx, text[idx], ctxt_nodes, ignore=(missing_idx, anchor_idx,))
                    ctxt_nodes[idx] = self.categorize(instance)

            instance = self.create_instance(
                anchor_idx, anchor_wd, ctxt_nodes, ignore=(missing_idx,))
            ctxt_nodes[anchor_idx] = self.categorize(instance)

        missing_instance = self.create_instance(
            missing_idx, None, ctxt_nodes)
        del missing_instance['anchor']

        concept = self.similarity_categorize(missing_instance)
        path = [concept]
        while sum([(option in concept.av_counts[anchor_key])
                   for option in options]) < options_needed:
            concept = concept.parent
            path.append(concept)
            if concept is None:
                print('Words not seen')
                return random.choice(options)
                raise ValueError('None of the options have been seen')

        return max(options,
                   key=lambda opt: concept.av_counts[anchor_key].get(opt, 0)), path, missing_instance[minor_key]

    def _cobweb_categorize(self, instance):
        """
        A cobweb specific version of categorize, not intended to be
        externally called.
        .. seealso:: :meth:`CobwebTree.categorize`
        """
        if anchor_key not in instance:
            return super()._cobweb_categorize(instance)
        try:
            paths = [list(reversed(list(get_path(word)))) for word in word_to_leaf[instance[anchor_key]]]
        except KeyError:
            return None

        current = self.root
        while current:
            if not current.children:
                return current
            paths = [option[1:] for option in paths if option[0] == current]
            options = list({option[0] for option in paths})

            _, best1, best2 = current.two_best_children(instance, options=options)
            current = best1


def unhidden_attr_val(pair):
    return pair[0][0] != '_'


class ContextualCobwebNode(CobwebNode):
    def shallow_copy(self):
        """
        Create a shallow copy of the current node (and not its children)
        This can be used to copy only the information relevant to the node's
        probability table without maintaining reference to other elements of
        the tree, except for the root which is necessary to calculate category
        utility.
        """
        temp = self.__class__()
        temp.tree = self.tree
        temp.parent = self.parent
        temp.set_counts_from_node(self)
        return temp

    def attr_values(self, attr_filter=unhidden_attr_val):
        """
        Iterates over the attributes-value pairs in the node's attribute-value
        table with the option to filter certain types. By default the filter
        will ignore hidden attributes and yield all others. If the string 'all'
        is provided then all attributes will be yielded. In neither of those
        cases the filter will be interpreted as a function that returns true if
        an attribute should be yielded and false otherwise.
        """
        if attr_filter == 'all':
            return self.av_counts.items()
        else:
            return filter(attr_filter, self.av_counts.items())

    def replace_node_as_context(self, node, new_node):
        """cur_node in iteration, node to be replace"""
        major_ctxt = self.av_counts[major_key]
        if node in major_ctxt:
            minor_ctxt = self.av_counts[minor_key]
            major_ctxt[new_node] += major_ctxt[node]
            minor_ctxt[new_node] += minor_ctxt[node]
            del major_ctxt[node]
            del minor_ctxt[node]
            for child in self.children:
                child.replace_node_as_context(node, new_node)

    def increment_counts(self, instance):
        """
        Increment the counts at the current node according to the specified
        instance.
        :param instance: A new instances to incorporate into the node.
        :type instance: :ref:`Instance<instance-rep>`
        """
        self.count += 1
        for attr in instance:
            if attr == minor_key or attr == major_key:
                self.av_counts.setdefault(
                    attr, Counter()).update(instance[attr])
                continue

            self.av_counts.setdefault(attr, {})
            self.av_counts[attr][instance[attr]] = (
                self.av_counts[attr].get(instance[attr], 0) + 1)

    def decrement_counts(self, instance):
        self.count -= 1
        for attr in instance:
            if attr == minor_key or attr == major_key:
                self.av_counts[attr] -= Counter(instance[attr])
                continue

            self.av_counts[attr][instance[attr]] -= 1

    def update_all_counts(self, node):
        cur = self
        while cur:
            cur.update_counts_from_node(node)
            cur = cur.parent

    def decrement_all_counts(self, node):
        cur = self
        while cur:
            cur.decrement_counts_from_node(node)
            cur = cur.parent

    def decrement_counts_from_node(self, node):
        """
        Decrements counts but does not restructure tree
        """
        self.count -= node.count

        for attr in node.attrs('all'):
            if attr == minor_key or attr == major_key:
                self.av_counts[attr] -= node.av_counts[attr]
                continue

            for val in node.av_counts[attr]:
                self.av_counts[attr][val] -= node.av_counts[attr][val]

    def update_counts_from_node(self, node):
        """
        """
        self.count += node.count

        for attr in node.attrs('all'):
            if attr == minor_key or attr == major_key:
                self.av_counts.setdefault(
                    attr, Counter()).update(node.av_counts[attr])
                continue

            counts = self.av_counts.setdefault(attr, {})
            for val in node.av_counts[attr]:
                counts[val] = counts.get(val, 0)+node.av_counts[attr][val]

    def set_counts_from_node(self, node):
        self.count = node.count
        for attr, counts in node.attr_values('all'):
            if attr == minor_key or attr == major_key:
                self.av_counts[attr] = Counter(counts)
                continue

            self.av_counts[attr] = dict(counts)

    def __str__(self):
        return "Concept-{}".format(self.concept_id)

    def __repr__(self):
        return str(self)

    def expected_correct_guesses(self, instance=None):
        """
        Returns the number of correct guesses that are expected from the given
        concept.
        This is the sum of the probability of each attribute value squared.
        This function is used in calculating category utility.
        :return: the number of correct guesses that are expected from the given
                 concept.
        :rtype: float
        """
        correct_guesses = 0.0
        if instance is None:
            attrs = self.attrs()
        else:
            attrs = self.attrs(attr_filter=lambda x: x in instance and x[0] != '_')
        for attr in attrs:
            temp = 0
            counts = self.av_counts[attr]
            for val in counts:
                count = counts[val]
                temp += (count * count)
            # Turns counts into probabilities
            temp /= (self.count * self.count)

            if attr == major_key:
                correct_guesses += temp * self.tree.major_weight
            elif attr == minor_key:
                correct_guesses += temp * self.tree.minor_weight
            elif attr == anchor_key:
                correct_guesses += temp * self.tree.anchor_weight
            else:
                assert False

        return correct_guesses

    def category_utility(self, instance=None):
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

        for child in self.children:
            p_of_child = child.count / self.count
            child_correct_guesses += (p_of_child *
                                      child.expected_correct_guesses(instance))

        return ((child_correct_guesses - self.expected_correct_guesses(instance))
                / len(self.children))

    def get_best_operation(self, instance, best1, best2, best1_cu,
                           possible_ops=("best", "new", "merge", "split")):
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

        if "best" in possible_ops:
            operations.append((best1_cu, "best"))
        if "new" in possible_ops:
            operations.append((self.cu_for_new_child(instance), 'new'))
        if "merge" in possible_ops and len(self.children) > 2 and best2:
            operations.append((self.cu_for_merge(best1, best2, instance),
                               'merge'))
        if "split" in possible_ops and len(best1.children) > 0:
            operations.append((self.cu_for_split(best1, instance), 'split'))

        operations.sort(reverse=True)

        return random_tiebreaker(operations, key=lambda x: x[0])

    def two_best_children(self, instance, options=None):
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
        if options is None:
            options = self.children
        if len(options) == 0:
            raise Exception("No children!")

        # Convert the relative CU's of the two best children into CU scores
        # that can be compared with the other operations.
        const = self.compute_relative_CU_const(instance)

        # If there's only one child, simply calculate the relevant utility
        if len(options) == 1:
            best1 = options[0]
            best1_relative_cu = self.relative_cu_for_insert(best1, instance)
            best1_cu = (best1_relative_cu / (self.count+1) / len(options)
                        + const)
            return best1_cu, best1, None

        children_relative_cu = [(self.relative_cu_for_insert(child, instance),
                                 child.count, child) for child in
                                options]
        children_relative_cu.sort(reverse=True, key=lambda x: x[:-1])

        best1_data, best2_data = tiebreak_top_2(
            children_relative_cu, key=lambda x: x[:-1])

        best1_relative_cu, _, best1 = best1_data
        best1_cu = (best1_relative_cu / (self.count+1) / len(options)
                    + const)
        best2 = best2_data[2]

        return best1_cu, best1, best2

    def compute_relative_CU_const(self, instance):
        """
        Computes the constant value that is used to convert between CU and
        relative CU scores. The constant value is basically the category
        utility that results from adding the instance to the root, but none of
        the children. It can be computed directly as:
        .. math::
            const = \\frac{1}{n} \\sum_{k=1}^{n} \\left[
            \\frac{C_k.count}{count + 1} \\sum_i \\sum_j P(A_i = V_{ij} |
            C_k)^2 \\right] - \\sum_i \\sum_j P(A_i = V_{ij} | UpdatedRoot)^2
        where :math:`n` is the number of children of the root, :math:`C_k` is
        child :math:`k`,  :math:`C_k.count` is the number of instances stored
        in child :math:`C_k`, :math:`count` is the number of instances stored
        in the root. Finally, :math:`UpdatedRoot` is a copy of the root that
        has been updated with the counts of the instance.
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: The value of the constant used to relativize the CU.
        :rtype: float
        """
        temp = self.shallow_copy()
        temp.increment_counts(instance)
        ec_root_u = temp.expected_correct_guesses(instance)

        const = 0
        for c in self.children:
            const += c.count * c.expected_correct_guesses(instance)

        const /= self.count + 1  # Turns counts into probabilities
        const -= ec_root_u
        const /= len(self.children)
        return const

    def relative_cu_for_insert(self, child, instance):
        """
        Computes a relative CU score for each insert operation. The relative CU
        score is more efficient to calculate for each insert operation and is
        guranteed to have the same rank ordering as the CU score so it can be
        used to determine which insert operation is best. The relative CU can
        be computed from the CU using the following transformation.
        .. math::
            relative\\_cu(cu) = (cu - const) * n * (count + 1)
        where :math:`const` is the one returned by
        :meth:`CobwebNode.compute_relative_CU_const`, :math:`n` is the number
        of children of the current node, and :math:`count` is the number of
        instances stored in the current node (the root).
        The particular :math:`const` value was chosen to make the calculation
        of the relative cu scores for each insert operation efficient. When
        computing the CU for inserting the instance into a particular child,
        the terms in the formula above can be expanded and many of the
        intermediate calculations cancel out. After these cancelations,
        computing the relative CU for inserting into a particular child
        :math:`C_i` reduces to:
        .. math::
            relative\\_cu\\_for\\_insert(C_i) = (C_i.count + 1) * \\sum_i
            \\sum_j P(A_i = V_{ij}| UpdatedC_i)^2 - (C_i.count) * \\sum_i
            \\sum_j P(A_i = V_{ij}| C_i)^2
        where :math:`UpdatedC_i` is a copy of :math:`C_i` that has been updated
        with the counts from the given instance.
        By computing relative_CU scores instead of CU scores for each insert
        operation, the time complexity of the underlying Cobweb algorithm is
        reduced from :math:`O(B^2 \\times log_B(n) \\times AV)` to
        :math:`O(B \\times log_B(n) \\times AV)` where :math:`B` is the average
        branching factor of the tree, :math:`n` is the number of instances
        being categorized, :math:`A` is the average number of attributes per
        instance, and :math:`V` is the average number of values per attribute.
        :param child: a child of the current node
        :type child: CobwebNode
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility of adding the instance to the given node
        :rtype: float
        """
        temp = child.shallow_copy()
        temp.increment_counts(instance)
        return ((child.count + 1) * temp.expected_correct_guesses(instance) -
                child.count * child.expected_correct_guesses(instance))

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
        temp = self.shallow_copy()
        temp.increment_counts(instance)

        for c in self.children:
            temp_child = c.shallow_copy()
            temp.children.append(temp_child)
            temp_child.parent = temp
            if c == child:
                temp_child.increment_counts(instance)
        return temp.category_utility(instance)

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
        temp = self.shallow_copy()
        for c in self.children:
            temp.children.append(c.shallow_copy())

        # temp = self.shallow_copy()

        temp.increment_counts(instance)
        temp.create_new_child(instance)
        return temp.category_utility(instance)

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
        temp = self.shallow_copy()
        temp.increment_counts(instance)

        new_child = self.__class__()
        new_child.tree = self.tree
        new_child.parent = temp
        new_child.update_counts_from_node(best1)
        new_child.update_counts_from_node(best2)
        new_child.increment_counts(instance)
        temp.children.append(new_child)

        for c in self.children:
            if c == best1 or c == best2:
                continue
            temp_child = c.shallow_copy()
            temp.children.append(temp_child)

        return temp.category_utility(instance)

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
        # Anchor pruning
        if (len(best1.av_counts[anchor_key]) == 1
                and len(best2.av_counts[anchor_key]) == 1
                and (next(iter(best1.av_counts[anchor_key]))
                     == next(iter(best2.av_counts[anchor_key])))):
            assert not (best1.children or best2.children)
            self.tree.root.replace_node_as_context(best2, best1)
            best1.update_counts_from_node(best2)
            self.tree.delete_node(best2)
            return best1
        new_child = self.__class__()
        new_child.parent = self
        new_child.tree = self.tree

        best1.parent = new_child
        # best1.tree = new_child.tree
        best2.parent = new_child
        # best2.tree = new_child.tree
        new_child.children.append(best1)
        new_child.children.append(best2)
        self.children.remove(best1)
        self.children.remove(best2)
        self.children.append(new_child)
        new_child.update_counts_from_node(best1)
        new_child.update_counts_from_node(best2)

        return new_child

    def cu_for_split(self, best, instance):
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
        temp = self.shallow_copy()

        for c in chain(self.children, best.children):
            if c == best:
                continue
            temp_child = c.shallow_copy()
            temp.children.append(temp_child)

        return temp.category_utility(instance)

    def is_exact_match(self, instance):
        """
        Returns true if the concept exactly matches the instance.
        :param instance: the instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: whether the instance perfectly matches the concept
        :rtype: boolean
        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        anchors = self.av_counts.get(anchor_key, ())
        result = len(anchors) == 1 and instance[anchor_key] in anchors
        if result:
            global overlaps
            overlaps += 1
        return result


overlaps = 0


def verify_structure(node):
    assert all(child.parent == node for child in node.children)
    [verify_structure(child) for child in node.children]


def in_tree(node):
    if node == node.tree.root:
        return True
    result = node in node.parent.children and in_tree(node.parent)
    if not result:
        print(node)
    return result


'''tree = ContextualCobwebTree(1, 4)
data = list(generate_ms_sentence_variant_synonyms(3, 10, 50))
random.shuffle(data)
for sent in tqdm(data):
    tree.fit_to_text_wo_stopwords([word for word in sent if word[:-2] not in stop_words])
visualize(tree)
1/0'''

if __name__ == "__main__":

    if LOAD:
        tree = pickle.load(open(MODEL_LOAD_LOCATION, mode='rb'))
    else:
        tree = ContextualCobwebTree(1, 4)

    for text_num in range(1):
        text = list(load_text(text_num))[:5000]

        # run('tree.fit_to_text_wo_stopwords(text)', sort='calls')
        tree.fit_to_text_wo_stopwords(text)
        text = [word for word in text if word not in stop_words]
        text_counts = Counter(text)

        print(overlaps)
        print('total overlaps',
              len(list(text_counts.elements()))-len(text_counts))

        questions = create_questions(text, 10, 4, 200)

        correct = 0
        answers_needed = 1
        for question in tqdm(questions):
            guess, path, minctxt = tree.guess_missing(
                *question, answers_needed)
            answer = question[1][0]
            # print(question)
            if guess == answer:
                correct += 1
                ...  # print('correct')
            else:
                print()
                print('question', question[0])
                print('minor ctxt', minctxt)
                ctxt_words = [next(iter(concept.av_counts[anchor_key])) for concept in minctxt]
                print('ctxt_words', ctxt_words)
                print('ctxt_counts', [text_counts[word] for word in ctxt_words])
                for word in ctxt_words:
                    for leaf in word_to_leaf[word]:
                        print(word, list(get_path(leaf)),)
                print('-'*90)
                print('path', [concept.concept_id for concept in path])
                print('initial counts', path[0].av_counts)
                print('id', path[-1].concept_id)
                print('counts', [(answer, path[-1].av_counts[anchor_key].get(answer, 0)) for answer in question[1]])
                print('answer', answer)
                print()
                ...  # print('incorrect. guessed "{}" when "{}" was correct'.format(guess, answer))
        print(correct/len(questions), 'answers needed: ', answers_needed)
    visualize(tree)

    # valid_concepts = tree.root.get_concepts()
    # tree.root.test_valid(valid_concepts)

    if SAVE:
        setrecursionlimit(TREE_RECURSION)
        pickle.dump(tree, open(MODEL_SAVE_LOCATION, mode='xb'))

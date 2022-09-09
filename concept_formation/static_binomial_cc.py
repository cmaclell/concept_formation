from cProfile import run
from functools import lru_cache
from collections import Counter
from math import log2
# from multiprocess import Pool
from tqdm import tqdm
import timeit
import pickle
from os.path import dirname, join
from sys import setrecursionlimit
from os import listdir
import resource
from time import time
import random

from concept_formation.cobweb import CobwebNode
from concept_formation.cobweb import CobwebTree
from concept_formation.utils import skip_slice, oslice
from concept_formation.utils import tiebreak_top_2
from visualize import visualize
from preprocess_text import load_text, stop_words


def word_to_base(word):
    return word.split('-')[0]


# TREE_RECURSION = 0x1000
# 
# # May segfault without this line. 0x100 is a guess at the size of stack frame.
# try:
#     resource.setrlimit(resource.RLIMIT_STACK,
#                        [0x100 * TREE_RECURSION, resource.RLIM_INFINITY])
# except ValueError:
#     print(Warning("Warning: Saving this model may result in a segfault"))
# setrecursionlimit(TREE_RECURSION)
# 
# SAVE = False
# TREE_RECURSION = 900000
# LOAD = False
# MODELS_PATH = join(dirname(__file__), 'saved_models')
# MODEL_SAVE_LOCATION = join(MODELS_PATH, 'saved_model_%s' % time())
# if LOAD:
#     MODEL_LOAD_LOCATION = join(MODELS_PATH, listdir(MODELS_PATH)[0])
# print(listdir(MODELS_PATH)[0])
# run

# random.seed(16)
ca_key = 'context'
anchor_key = 'anchor'
overlaps = 0


def get_path(node):
    while node:
        yield node
        node = node.parent


class ContextualCobwebTree(CobwebTree):

    def __init__(self, window):
        """
        Note window only specifies how much context to add to each side,
        doesn't include the anchor word.

        E.g., to get a window with 2 before and 2 words after the anchor, then
        set the window=2
        """
        super().__init__()
        self.root = ContextualCobwebNode()
        self.root.tree = self

        self.n_concepts = 1
        self.window = window
        self.instance = None
        self.prune_threshold = 0.0
        self.anchor_weight = 1
        self.context_weight = 1

        self.log_times = False
        self.word_to_leaf = {}
        # print(self.anchor_weight, self.context_weight)

    def _sanity_check_instance(self, instance):
        for attr in instance:
            if attr == ca_key:
                for concept in instance[ca_key]:
                    if not isinstance(concept, CobwebNode):
                        raise ValueError('context must be of type CobweNode')
                continue
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
                hash(instance[attr])
            except Exception:
                raise ValueError('Invalid value: '+str(instance[attr]) +
                                 ' of type: '+str(type(instance[attr])) +
                                 ' in instance: '+str(instance) +
                                 ',\n'+type(self).__name__ +
                                 ' only works with hashable values.')
            if instance[attr] is None:
                raise ValueError("Attributes with value None should"
                                 " be manually removed.")

    def ifit(self, instance):
        self.instance = instance
        return super().ifit(instance)

    def ifit_text(self, text, track=None):
        ctxt_nodes = []

        for anchor_idx, anchor_wd in enumerate(text):
            if self.log_times:
                start = timeit.default_timer()
            while ((len(ctxt_nodes) < anchor_idx + self.window + 1) and
                   len(ctxt_nodes) < len(text)):

                if text[len(ctxt_nodes)] is None:
                    ctxt_nodes.append(None)
                else:
                    ctxt_nodes.append(self.categorize(
                        self.create_instance(len(ctxt_nodes),
                                            text[len(ctxt_nodes)], ctxt_nodes, ignore=())))

            if anchor_wd is None:
                continue

            for _ in range(2):
                for i in range(self.window + 1):
                    idx = anchor_idx + i # - self.window
                    if idx < 0 or idx >= len(text):
                        continue
                    instance = self.create_instance(
                        idx, anchor_wd, ctxt_nodes, ignore=())
                    ctxt_nodes[idx] = self.categorize(instance)

            instance = self.create_instance(anchor_idx, anchor_wd, ctxt_nodes)

            base = word_to_base(anchor_wd)
            if track and base in track:
                try:
                    concept = self.terminating_categorize(instance)
                    total = 0
                    syn_count = 0
                    for word, count in concept.av_counts[anchor_key].items():
                        total += count
                        if base == word_to_base(word):
                            syn_count += count
                    track[base].append(syn_count/count)
                except KeyError:
                    track[base].append(0.0)

            ctxt_nodes[anchor_idx] = self.ifit(instance)

            self.word_to_leaf.setdefault(anchor_wd, set()).add(ctxt_nodes[anchor_idx])

            if self.log_times:
                stop = timeit.default_timer()
                with open('out.csv', 'a') as fout:
                    fout.write("{},{:.8f}\n".format(anchor_idx, stop - start))

    def categorize_text(self, text):
        ctxt_nodes = []

        for anchor_idx, anchor_wd in enumerate(text):
            while ((len(ctxt_nodes) < anchor_idx + self.window + 1) and
                   len(ctxt_nodes) < len(text)):
                ctxt_nodes.append(self.categorize(
                    self.create_instance(len(ctxt_nodes),
                                         text[len(ctxt_nodes)], ctxt_nodes, ignore=())))

            for _ in range(2):
                for i in range(self.window + 1):
                    idx = anchor_idx + i # - self.window
                    if idx < 0 or idx >= len(text):
                        continue
                    instance = self.create_instance(
                        idx, anchor_wd, ctxt_nodes, ignore=())
                    ctxt_nodes[idx] = self.categorize(instance)

            instance = self.create_instance(anchor_idx, anchor_wd, ctxt_nodes)
            ctxt_nodes[anchor_idx] = super()._cobweb_categorize(instance)

        return ctxt_nodes

    def create_instance(self, anchor_idx, word, context_nodes, ignore=()):
        context = filter(None, self.surrounding(
            context_nodes, anchor_idx, self.window, ignore=ignore))

        instance = {ca_key: Counter(), 'anchor': word}
        for n in context:
            instance[ca_key].update(get_path(n))

        return instance

    def surrounding(self, sequence, center, dist, ignore=()):
        if ignore:
            return list(
                oslice(sequence, max(0, center-dist), *sorted((*(x for x in ignore if 0 < x < center+dist+1), center)), center+dist+1))
        return list(
            skip_slice(sequence, max(0, center-dist), center+dist+1, center))

    def cobweb(self, instance):
        """
        Updated version of cobweb algorithm that updates
        cross-concept references
        """
        current = self.root

        while current:

            # the current.count == 0 here is for the initially empty tree.
            if not current.children and (current.is_exact_match(instance) or
                                         current.count == 0):
                # print("leaf match")
                current.increment_counts(instance, track=True)
                break

            elif not current.children:
                # print("fringe split")
                new = current.__class__(current, track=True)
                current.parent = new
                new.children.append(current)

                if new.parent:
                    new.parent.children.remove(current)
                    new.parent.children.append(new)
                else:
                    self.root = new

                # REGISTER UPDATE
                current.register_new_parent(new)

                new.increment_counts(instance, track=True)
                new.prune()
                current = new.create_new_child(instance)
                self.n_concepts += 2
                break

            else:
                best1_cu, best1, best2 = current.two_best_children(instance)
                _, best_action = current.get_best_operation(instance, best1,
                                                            best2, best1_cu)

                # print(best_action)
                if best_action == 'best':
                    current.increment_counts(instance, track=True)
                    current.prune()
                    current = best1
                elif best_action == 'new':
                    current.increment_counts(instance, track=True)
                    current.prune()
                    current = current.create_new_child(instance, track=True)
                    self.n_concepts += 1
                    break
                elif best_action == 'merge':
                    current.increment_counts(instance, track=True)
                    current.prune()
                    new_child = current.merge(best1, best2, track=True)

                    # REGISTER UPDATE
                    best1.register_new_parent(new_child)
                    best2.register_new_parent(new_child)

                    current = new_child
                    self.n_concepts += 1

                elif best_action == 'split':
                    # REGISTER UPDATE
                    best1.register_delete()
                    current.split(best1)
                    self.n_concepts -= 1
                else:
                    raise Exception('Best action choice "' + best_action +
                                    '" not a recognized option. This should be'
                                    ' impossible...')

        return current

    def similarity_categorize(self, instance):
        current = self.root

        def similarity(node):
            """Average probability"""
            correct_guesses = 0
            if anchor_key in instance:
                correct_guesses += node.av_counts[anchor_key].get(
                    instance[anchor_key], 0) * self.anchor_weight
            if ca_key in instance:
                correct_guesses += sum(
                    node.av_counts[ca_key].get(ctxt, 0)
                    for ctxt in instance[ca_key]) * self.context_weight

            return correct_guesses  # / node.count

        while current:
            if not current.children:
                return current

            best = max(current.children, key=similarity)
            current = best
        return current

    def guess_missing(self, text, options, options_needed):
        """
        None used to represent missing words
        """
        text = [word for word in text if word not in stop_words]

        missing_idx = text.index(None)
        ctxt_nodes = []

        for anchor_idx, anchor_wd in enumerate(text):
            while ((len(ctxt_nodes) < anchor_idx + self.window + 1) and
                    len(ctxt_nodes) < len(text)):
                if len(ctxt_nodes) == missing_idx:
                    ctxt_nodes.append(None)
                    continue
                ctxt_nodes.append(self.categorize(
                    self.create_instance(len(ctxt_nodes),
                                         text[len(ctxt_nodes)], ctxt_nodes, ignore=(missing_idx,))))

            if anchor_idx == missing_idx:
                continue

            for _ in range(2):
                for i in range(self.window + 1):
                    idx = anchor_idx + i # - self.window
                    if idx < 0 or idx >= len(text) or idx == missing_idx:
                        continue

                    instance = self.create_instance(
                        idx, text[idx], ctxt_nodes, ignore=(missing_idx,))
                    ctxt_nodes[idx] = self.categorize(instance)

            instance = self.create_instance(
                anchor_idx, anchor_wd, ctxt_nodes, ignore=(missing_idx,))
            ctxt_nodes[anchor_idx] = self.categorize(instance)

        missing_instance = self.create_instance(
            missing_idx, anchor_wd, ctxt_nodes)
        del missing_instance['anchor']

        # Defined here so as to clear the cache after each run
        @lru_cache(maxsize=None)
        def __get_anchor_counts(node):
            if not node.children:
                return Counter(node.av_counts['anchor'])
            return sum([__get_anchor_counts(child)
                        for child in node.children], start=Counter())

        concept = self.categorize(missing_instance)
        while sum([(option in __get_anchor_counts(concept))
                   for option in options]) < options_needed:
            concept = concept.parent
            if concept is None:
                raise ValueError('None of the options have been seen')

        return max(options,
                   key=lambda opt: __get_anchor_counts(concept)[opt])

    def _cobweb_categorize(self, instance):
        """
        A cobweb specific version of categorize, not intended to be
        externally called.
        .. seealso:: :meth:`CobwebTree.categorize`
        """
        if anchor_key not in instance:
            return super()._cobweb_categorize(instance)
        try:
            paths = [list(reversed(list(get_path(word)))) for word in self.word_to_leaf[instance[anchor_key]]]
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


class ContextualCobwebNode(CobwebNode):
    def __init__(self, other_node=None, track=False):
        self.n_context_elements = 0
        self.registered = set()

        self.concept_id = self.gensym()
        self.count = 0.0
        self.av_counts = {}
        self.children = []
        self.parent = None
        self.tree = None

        if other_node:
            self.tree = other_node.tree
            self.parent = other_node.parent
            self.set_counts_from_node(other_node, track)

            self.children.extend(
                self.__class__(child) for child in other_node.children)

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

    def create_new_child(self, instance, track=False):
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
        new_child.increment_counts(instance, track)
        self.children.append(new_child)
        return new_child

    def merge(self, best1, best2, track=False):
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

        new_child.update_counts_from_node(best1, track)
        new_child.update_counts_from_node(best2, track)
        best1.parent = new_child
        # best1.tree = new_child.tree
        best2.parent = new_child
        # best2.tree = new_child.tree
        new_child.children.append(best1)
        new_child.children.append(best2)
        self.children.remove(best1)
        self.children.remove(best2)
        self.children.append(new_child)

        return new_child

    def get_concepts(self):
        s = {self}
        for c in self.children:
            s.update(c.get_concepts())
        return s

    def test_valid(self, valid_concepts):
        for attr in self.av_counts:
            if attr == ca_key:
                for concept in self.av_counts[ca_key]:
                    assert concept.concept_id in valid_concepts
        for c in self.children:
            c.test_valid(valid_concepts)

    def increment_counts(self, inst, track=False):
        """
        Adds binomial distribution for estimating concept counts
        """
        self.count += 1

        for attr in inst:

            if attr == ca_key:
                self.av_counts.setdefault(ca_key, Counter())
                for concept in inst[ca_key]:
                    self.av_counts[ca_key][concept] += inst[ca_key][concept]
                    # only count if it is a terminal, don't count nonterminals
                    if not concept.children:
                        self.n_context_elements += inst[ca_key][concept]

                    if track:
                        concept.register(self)

                continue

            self.av_counts.setdefault(attr, {})
            prior_count = self.av_counts[attr].get(inst[attr], 0)
            self.av_counts[attr][inst[attr]] = prior_count + 1

    def update_counts_from_node(self, node, track=False):
        """
        Adds binomial distribution for estimating concept counts
        """
        self.count += node.count
        self.n_context_elements += node.n_context_elements

        for attr in node.attrs('all'):
            if attr == ca_key:
                ctxt_counts = self.av_counts.setdefault(ca_key, Counter())
                # self.av_counts[ca_key] += node.av_counts[ca_key]
                for concept in node.av_counts[ca_key]:
                    ctxt_counts[concept] += node.av_counts[ca_key][concept]

                    if track:
                        concept.register(self)

                continue

            counts = self.av_counts.setdefault(attr, {})
            for val in node.av_counts[attr]:
                counts[val] = counts.get(val, 0)+node.av_counts[attr][val]

        # self.prune_low_probability()

    def set_counts_from_node(self, node, track=False):
        self.count = node.count
        self.n_context_elements = node.n_context_elements
        if track:
            ctxt = Counter(node.av_counts[ca_key])
            for concept in node.av_counts[ca_key]:
                concept.register(self)
        else:
            ctxt = Counter(node.av_counts[ca_key])
        self.av_counts = {attr: (dict(val) if attr != ca_key else ctxt)
                          for attr, val in node.av_counts.items()}

    def prune(self):
        del_nodes = []
        del_av = []

        for attr in self.attrs('all'):
            if attr == ca_key:
                for concept in self.av_counts[ca_key]:
                    if ((self.av_counts[ca_key][concept]
                         / self.n_context_elements)
                            < self.tree.prune_threshold):
                        del_nodes.append(concept)

                continue

            for val in self.av_counts[attr]:
                if ((self.av_counts[attr][val] / self.count)
                        < self.tree.prune_threshold):
                    del_av.append((attr, val))

        for n in del_nodes:
            n.unregister(self)
            del self.av_counts[ca_key][n]

        for a, v in del_av:
            del self.av_counts[a][v]
            # if len(self.av_counts[a]) == 0:
            #     del self.av_counts[a]

    def register_delete(self):
        for concept in self.av_counts.get(ca_key, ()):
            concept.unregister(self)

        for c in self.registered:
            del c.av_counts[ca_key][self]

        if (self.tree.instance is not None
                and self in self.tree.instance[ca_key]):
            del self.tree.instance[ca_key][self]

    def register_new_parent(self, parent):
        for c in self.registered:
            context_dict = c.av_counts[ca_key]
            parent.register(c)
            context_dict[parent] = (
                context_dict.get(parent, 0)
                + context_dict[self])

        if (self.tree.instance is not None
                and self in self.tree.instance[ca_key]):
            context_dict = self.tree.instance[ca_key]
            context_dict[parent] = (context_dict.get(parent, 0)
                                    + context_dict[self])

    def __str__(self):
        return "Concept-{}".format(self.concept_id)

    def register(self, other):
        self.registered.add(other)

    def unregister(self, other):
        self.registered.discard(other)

    def expected_correct_guesses(self):
        """
        Modified to handle attribute that are concepts to tally and compute
        correct guesses over all concepts in path.
        """
        correct_guesses = 0.0
        context_guesses = 0.0

        for attr in self.attrs():
            if attr == ca_key:
                for concept, concept_count in self.av_counts[ca_key].items():
                    prob = concept_count / self.n_context_elements
                    # context_guesses += (prob * prob)
                    # context_guesses += ((1-prob) * (1-prob))
                    context_guesses -= 2 * prob * (1-prob)
                    # Should add 1, but it's accounted for in final processing

                continue

            correct_guesses += sum(
                (prob * prob) for prob in self.av_counts[attr].values())

        # context_guesses += n_concepts
        # context_guesses *= self.tree.context_weight / n_concepts
        context_guesses /= self.tree.n_concepts
        context_guesses += 1

        # Convert counts to probabilities
        correct_guesses /= (self.count * self.count)

        return (self.tree.anchor_weight * correct_guesses
                + self.tree.context_weight * context_guesses)

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
        instance_attrs = set(filter(lambda x: x[0] != "_", instance))
        self_attrs = set(self.attrs())

        if self_attrs != instance_attrs:
            return False

        for attr in self_attrs:
            attr_counts = self.av_counts[attr]
            if attr == ca_key:
                if instance[ca_key] != attr_counts.keys():
                    return False
                for ctxt_count in attr_counts.values():
                    if ctxt_count != self.count:
                        return False
            elif attr_counts.get(instance[attr], 0) != self.count:
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
        output["name"] = "Concept" + str(self.concept_id)
        output["size"] = self.count
        output["children"] = []

        temp = {}

        temp['aa-context-aa'] = {}

        for attr in self.attrs('all'):

            if attr == ca_key:
                for concept in self.av_counts[ca_key]:
                    temp['aa-context-aa'][
                        str(concept)] = (self.av_counts[ca_key][concept] /
                                         self.n_context_elements * self.count)

                    continue

            temp[str(attr)] = {}
            for val in self.av_counts[attr]:
                temp[str(attr)][str(val)] = self.av_counts[attr][val]

        for child in self.children:
            output["children"].append(child.output_json())

        output["counts"] = temp

        # from pprint import pprint
        # pprint(output)

        return output


if __name__ == "__main__":

    if LOAD:
        tree = pickle.load(open(MODEL_LOAD_LOCATION, mode='rb'))
    else:
        tree = ContextualCobwebTree(window=4)

    for text_num in range(1):
        text = [word for word in load_text(text_num) if word not in
                stop_words][:5000]

        tree.fit_to_text(text)
        questions = create_questions(text, 10, 4, 200)
        correct = 0
        answers_needed = 1
        for question in questions:
            guess = tree.guess_missing(*question, answers_needed)
            answer = question[1][0]
            # print(question)
            if guess == answer:
                correct += 1
                ...  # print('correct')
            else:
                ...  # print('incorrect. guessed "{}" when "{}" was correct'.format(guess, answer))
        print(correct/200, 'answers needed: ', answers_needed)
    visualize(tree)

    # valid_concepts = tree.root.get_concepts()
    # tree.root.test_valid(valid_concepts)

    if SAVE:
        setrecursionlimit(TREE_RECURSION)
        pickle.dump(tree, open(MODEL_SAVE_LOCATION, mode='xb'))

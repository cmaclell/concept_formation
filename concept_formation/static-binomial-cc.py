from cProfile import run
from functools import lru_cache
from collections import Counter
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
from visualize import visualize
from preprocess_text import load_text, stop_words

TREE_RECURSION = 0x10000

# May segfault without this line. 0x100 is a guess at the size of stack frame.
try:
    resource.setrlimit(
        resource.RLIMIT_STACK, [0x100 * TREE_RECURSION, resource.RLIM_INFINITY])
except ValueError:
    print(Warning("Warning: Saving this model may result in a segfault"))
setrecursionlimit(TREE_RECURSION)

SAVE = False
TREE_RECURSION = 900000
LOAD = False
MODELS_PATH = join(dirname(__file__), 'saved_models')
MODEL_SAVE_LOCATION = join(MODELS_PATH, 'saved_model_%s' % time())
if LOAD:
    MODEL_LOAD_LOCATION = join(MODELS_PATH, listdir(MODELS_PATH)[0])
print(listdir(MODELS_PATH)[0])

random.seed(17)
ca_key = '#Ctxt#'


def get_path(node):
    cur = node
    while cur:
        yield cur
        cur = cur.parent


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
        self.context_weight = 2

        self.log_times = False

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

    def fit_to_text(self, text):
        ctxt_nodes = []

        for anchor_idx, anchor_wd in tqdm(list(enumerate(text))):
            if self.log_times:
                start = timeit.default_timer()
            while ((len(ctxt_nodes) < anchor_idx + self.window + 1) and
                   len(ctxt_nodes) < len(text)):
                ctxt_nodes.append(
                    self.categorize({'anchor': text[len(ctxt_nodes)]}))

            for _ in range(2):
                for i in range(self.window + 1):
                    idx = anchor_idx + i
                    if idx < 0 or idx >= len(text):
                        continue
                    instance = self.create_instance(idx, anchor_wd, ctxt_nodes)
                    ctxt_nodes[idx] = self.categorize(instance)

            instance = self.create_instance(anchor_idx, anchor_wd, ctxt_nodes)

            ctxt_nodes[anchor_idx] = self.ifit(instance)

            if self.log_times:
                stop = timeit.default_timer()
                with open('out.csv', 'a') as fout:
                    fout.write("{},{:.8f}\n".format(anchor_idx, stop - start))

    def create_instance(self, anchor_idx, word, context_nodes):
        context = context_nodes[max(0, anchor_idx-self.window): anchor_idx]
        context += context_nodes[anchor_idx+1:anchor_idx+self.window+1]

        instance = {ca_key: Counter(), 'anchor': word}
        for n in context:
            instance[ca_key].update(get_path(n))

        return instance

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

    def create_categorization_instance(self, anchor_idx, word, context_nodes):
        context = context_nodes[max(0, anchor_idx-self.window): anchor_idx]
        context += context_nodes[anchor_idx+1:anchor_idx+self.window+1]

        instance = {ca_key: Counter(), 'anchor': word}
        for n in filter(None, context):
            instance[ca_key].update(get_path(n))

        return instance

    def guess_missing(self, text, options, options_needed):
        """
        None used to represent missing words
        """
        assert len(options) >= options_needed
        missing_idx = text.index(None)
        ctxt_nodes = []

        for anchor_idx, anchor_wd in tqdm(enumerate(text)):
            while ((len(ctxt_nodes) < anchor_idx + self.window + 1) and
                    len(ctxt_nodes) < len(text)):
                if len(ctxt_nodes) == missing_idx:
                    ctxt_nodes.append(None)
                    continue
                ctxt_nodes.append(
                    self.categorize({'anchor': text[len(ctxt_nodes)]}))

            if anchor_idx == missing_idx:
                continue

            for _ in range(2):
                for i in range(self.window + 1):
                    idx = anchor_idx + i
                    if idx < 0 or idx >= len(text) or idx == missing_idx:
                        continue
                    instance = self.create_categorization_instance(
                        idx, anchor_wd, ctxt_nodes)
                    ctxt_nodes[idx] = self.categorize(instance)

            instance = self.create_categorization_instance(
                anchor_idx, anchor_wd, ctxt_nodes)
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

    def increment_counts(self, instance, track=False):
        """
        Adds binomial distribution for estimating concept counts
        """
        self.count += 1

        for attr in instance:

            if attr == ca_key:
                self.av_counts.setdefault(ca_key, Counter())
                for concept in instance[ca_key]:
                    self.av_counts[ca_key][concept] += instance[ca_key][concept]
                    # only count if it is a terminal, don't count nonterminals
                    if not concept.children:
                        self.n_context_elements += instance[ca_key][concept]

                    if track:
                        concept.register(self)

                continue

            self.av_counts.setdefault(attr, {})
            prior_count = self.av_counts[attr].get(instance[attr], 0)
            self.av_counts[attr][instance[attr]] = prior_count + 1

    def update_counts_from_node(self, node, track=False):
        """
        Adds binomial distribution for estimating concept counts
        """
        self.count += node.count
        self.n_context_elements += node.n_context_elements

        for attr in node.attrs('all'):
            if attr == ca_key:
                self.av_counts.setdefault(ca_key, Counter())
                # self.av_counts[ca_key] += node.av_counts[ca_key]
                for concept in node.av_counts[ca_key]:
                    self.av_counts[ca_key][concept] += node.av_counts[ca_key][concept]

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
        self.av_counts = {attr: (dict(val) if attr != ca_key else ctxt) for attr, val in node.av_counts.items()}

    def prune(self):
        del_nodes = []
        del_av = []

        for attr in self.attrs('all'):
            if attr == ca_key:
                for concept in self.av_counts[ca_key]:
                    if ((self.av_counts[ca_key][concept] / self.n_context_elements)
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

        if (self.tree.instance is not None and self in self.tree.instance[ca_key]):
            del self.tree.instance[ca_key][self]

    def register_new_parent(self, parent):
        for c in self.registered:
            context_dict = c.av_counts[ca_key]
            parent.register(c)
            context_dict[parent] = (
                context_dict.get(parent, 0)
                + context_dict[self])

        if (self.tree.instance is not None and self in self.tree.instance[ca_key]):
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
        n_concepts = self.tree.n_concepts

        for attr in self.attrs():
            if attr == ca_key:
                for concept_count in self.av_counts[ca_key].values():
                    prob = concept_count / self.n_context_elements
                    # context_guesses += (prob * prob)
                    # context_guesses += ((1-prob) * (1-prob))
                    context_guesses -= 2 * prob * (1-prob)
                    # Should add 1, but it's accounted for in the final processing

                continue

            correct_guesses += sum(
                [(prob * prob) for prob in self.av_counts[attr].values()])

        # Weight and convert counts to probabilities
        correct_guesses *= self.tree.anchor_weight / (self.count * self.count)

        context_guesses += n_concepts
        context_guesses *= self.tree.context_weight / n_concepts

        return correct_guesses + context_guesses

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


def create_questions(text, question_length, num_answers, n):
    questions = []
    for _ in range(n):
        pos = random.randint(0, len(text)-question_length-1)
        blank = random.randint(2, question_length-3)
        question = text[pos:pos+question_length]
        answer = question[blank]
        question[blank] = None
        questions.append((question, [answer, *(random.choice(text) for _ in range(num_answers - 1))]))
    return questions


if __name__ == "__main__":

    if LOAD:
        tree = pickle.load(open(MODEL_LOAD_LOCATION, mode='rb'))
    else:
        tree = ContextualCobwebTree(window=4)

    for text_num in range(1):
        text = [word for word in load_text(text_num) if word not in
                stop_words][:10000]

        tree.fit_to_text(text)
        correct = 0
        answers_needed = 5
        questions = create_questions(text, 10, 5, 200)
        for question in questions:
            guess = tree.guess_missing(*question, answers_needed)
            answer = question[1][0]
            # print(question)
            if guess == answer:
                correct += 1
                ... # print('correct')
            else:
                ... # print('incorrect. guessed "{}" when "{}" was correct'.format(guess, answer))
        print(correct/200, 'answers needed: ', answers_needed)
        correct = 0
        answers_needed = 1
        for question in questions:
            guess = tree.guess_missing(*question, answers_needed)
            answer = question[1][0]
            # print(question)
            if guess == answer:
                correct += 1
                ... # print('correct')
            else:
                ... # print('incorrect. guessed "{}" when "{}" was correct'.format(guess, answer))
        print(correct/200, 'answers needed: ', answers_needed)
    visualize(tree)

    # valid_concepts = tree.root.get_concepts()
    # tree.root.test_valid(valid_concepts)

    if SAVE:
        setrecursionlimit(TREE_RECURSION)
        pickle.dump(tree, open(MODEL_SAVE_LOCATION, mode='xb'))

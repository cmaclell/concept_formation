from cProfile import run
from collections import Counter
# from multiprocess import Pool
from tqdm import tqdm
import pickle
from os.path import dirname, join
from sys import setrecursionlimit
from os import listdir
import resource
from time import time
import random

from concept_formation.cobweb import CobwebNode
from concept_formation.cobweb import CobwebTree
from concept_formation.utils import skip_slice
from visualize import visualize
from preprocess_text import load_text, stop_words, load_microsoft_qa

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
    MODEL_LOAD_LOCATION = join(MODELS_PATH, listdir(MODELS_PATH)[0])
print(listdir(MODELS_PATH)[0])
run

random.seed(16)
minor_key = '#MinorCtxt#'
major_key = '#MajorCtxt#'
anchor_key = 'anchor'


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
                if not isinstance(instance[attr], str) and not isinstance(instance[attr], int):
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
        output_nodes = []
        text = [word for word in text if word not in stop_words]

        for anchor_idx, anchor_wd in enumerate(tqdm(text)):
            output_nodes.append(self.ifit(
                self.create_instance(anchor_idx, anchor_wd, text)))

        return output_nodes

    def create_instance(self, anchor_idx, anchor_word, text,
                        filter_stop_for_minor=False):
        major_context = self.surrounding(text, anchor_idx, self.major_window)
        if filter_stop_for_minor:
            raise NotImplementedError
        else:
            minor_context = self.surrounding(
                text, anchor_idx, self.minor_window)

        return {minor_key: minor_context,
                major_key: major_context,
                anchor_key: anchor_word,
                '_idx': anchor_idx}

    def surrounding(self, sequence, center, dist):
        return list(
            skip_slice(sequence, max(0, center-dist), center+dist+1, center))

    def guess_missing(self, text, options, options_needed=1,
                      filter_stop_for_minor=False):
        text = [word for word in text if word not in stop_words]
        index = text.index(None)
        major_context = self.surrounding(text, index, self.major_window)
        if filter_stop_for_minor:
            raise NotImplementedError
        else:
            minor_context = self.surrounding(text, index, self.minor_window)

        concept = self.categorize({minor_key: minor_context,
                                   major_key: major_context})
        while sum([(option in concept.av_counts[anchor_key])
                   for option in options]) < options_needed:
            concept = concept.parent
            if concept is None:
                print('Words not seen')
                return random.choice(options)
                raise ValueError('None of the options have been seen')

        return max(options,
                   key=lambda opt: concept.av_counts[anchor_key].get(opt, 0))


class ContextualCobwebNode(CobwebNode):
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

    def update_counts_from_node(self, node):
        """
        Adds binomial distribution for estimating concept counts
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
        raise NotImplementedError()

    def __str__(self):
        return "Concept-{}".format(self.concept_id)

    def __repr__(self):
        return str(self)

    def expected_correct_guesses(self):
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

        for attr in self.attrs():
            temp = 0
            counts = self.av_counts[attr]
            for val in counts:
                prob = counts[val] / self.count
                temp += (prob * prob)

            if attr == anchor_key:
                correct_guesses += temp * self.tree.anchor_weight
            elif attr == minor_key:
                correct_guesses += temp * self.tree.minor_weight
            elif attr == major_key:
                correct_guesses += temp * self.tree.major_weight
            else:
                assert False

        return correct_guesses

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
            if attr == minor_key or attr == major_key:
                if instance[attr] != attr_counts.keys():
                    return False
                for ctxt_count in attr_counts.values():
                    if ctxt_count != self.count:
                        return False
            elif attr_counts.get(instance[attr], 0) != self.count:
                return False
        return True


def create_questions(text, question_length, nimposters, n):
    questions = []
    for _ in range(n):
        pos = random.randint(0, len(text)-question_length-1)
        blank = random.randint(2, question_length-3)
        question = text[pos:pos+question_length]
        answer = question[blank]
        question[blank] = None
        questions.append((question,
                         [answer, *(random.choice(text)
                          for _ in range(nimposters))]))
    return questions


def test_microsoft(model):
    correct = 0
    for total, (question, answers, answer) in enumerate(load_microsoft_qa()):
        if model.guess_missing(question, answers, 1) == answers[answer]:
            correct += 1
    total += 1
    return correct / total


if __name__ == "__main__":

    if LOAD:
        tree = pickle.load(open(MODEL_LOAD_LOCATION, mode='rb'))
    else:
        tree = ContextualCobwebTree(1, 4)

    for text_num in range(1):
        text = list(load_text(text_num))[:5000]

        tree.fit_to_text_wo_stopwords(text)
        text = [word for word in text if word not in stop_words]

        # print(test_microsoft(tree))

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

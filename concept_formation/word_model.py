from cProfile import run
from collections import Counter
from email.errors import NonPrintableDefect
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
from preprocess_text import load_text, stop_words


def word_to_base(word):
    return word.split('-')[0]


class ContextualCobwebTree(CobwebTree):

    def __init__(self, window_size):
        """
        Note window only specifies how much context to add to each side,
        doesn't include the anchor word.

        E.g., to get a window with 2 before and 2 words after the anchor, then
        set the window=2
        """
        super().__init__()
        self.root = ContextualCobwebNode()
        self.root.tree = self
        self.anchor_weight = 1
        self.context_weight = 1

        self.window_size = window_size

        self.word_to_leaf = {}

    def _sanity_check_instance(self, instance):
        return True

    def ifit_text(self, text, track=None):
        output_nodes = []

        for anchor_idx, anchor_wd in enumerate(text):
            if anchor_wd is None:
                output_nodes.append(None)
                continue
            instance = self.create_instance(anchor_idx, anchor_wd, text)

            base = word_to_base(anchor_wd)
            if track and base in track:
                try:
                    concept = self.terminating_categorize(instance)
                    # concept = self.terminating_categorize({'context': instance['context']})
                    # print('Instance:', instance)
                    # print(concept.av_counts)
                    total = 0
                    syn_count = 0
                    for word, count in concept.av_counts['anchor'].items():
                        total += count
                        if base == word_to_base(word):
                            syn_count += count
                    track[base].append(syn_count/count)
                except KeyError:
                    track[base].append(0.0)

            output_nodes.append(self.ifit(instance))

            self.word_to_leaf.setdefault(anchor_wd, set()).add(output_nodes[-1])

        return output_nodes

    def categorize_text(self, text):
        output_nodes = []

        for anchor_idx, anchor_wd in enumerate(text):
            output_nodes.append(self.categorize(
                self.create_instance(anchor_idx, anchor_wd, text)))

        return output_nodes

    def term_cat(self, text, nodes, idx):
        return self.terminating_categorize(self.create_instance(idx, text[idx], text))


    def create_instance(self, anchor_idx, anchor_word, text,
                        filter_stop_for_minor=False):
        major_context = self.surrounding(text, anchor_idx, self.window_size)

        ctx = Counter(major_context)
        for attr in ctx:
            ctx[attr] = ctx[attr] / len(major_context)

        # if filter_stop_for_minor:
        #     raise NotImplementedError
        # else:
        #     minor_context = self.surrounding(
        #         text, anchor_idx, self.minor_window)

        return {"anchor": {anchor_word: 1},
                "context": ctx}

    def surrounding(self, sequence, center, dist):
        return list(
            skip_slice(sequence, max(0, center-dist), center+dist+1, center))

    def guess_missing(self, text, options, options_needed=1,
                      filter_stop_for_minor=False):
        text = [word for word in text if word not in stop_words]
        index = text.index(None)
        major_context = self.surrounding(text, index, self)

        ctx = Counter(major_context)
        for attr in ctx:
            ctx[attr] = ctx[attr] / len(major_context)

        # if filter_stop_for_minor:
        #     raise NotImplementedError
        # else:
        #     minor_context = self.surrounding(text, index, self.minor_window)

        concept = self.categorize({'context': ctx})
        while sum([(option in concept.av_counts['anchor'])
                   for option in options]) < options_needed:
            concept = concept.parent
            if concept is None:
                print('Words not seen')
                return random.choice(options)
                raise ValueError('None of the options have been seen')

        return max(options,
                   key=lambda opt: concept.av_counts['anchor'].get(opt, 0))


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
            for val in instance[attr]:
                self.av_counts.setdefault(attr, {})
                self.av_counts[attr].setdefault(val, 0)
                self.av_counts[attr][val] += instance[attr][val]

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
        attr_count = 0

        for attr in self.attrs():
            attr_count += 1
            if attr in self.av_counts:
                weight = 1
                if attr == "anchor":
                    weight = self.tree.anchor_weight
                if attr == "context":
                    weight = self.tree.context_weight
                for val in self.av_counts[attr]:
                    prob = (self.av_counts[attr][val]) / self.count
                    correct_guesses += weight * (prob * prob)

        return correct_guesses / attr_count

    def __str__(self):
        return "Concept-{}".format(self.concept_id)

    def __repr__(self):
        return str(self)

    def is_exact_match(self, instance):
        """
        Returns true if the concept exactly matches the instance.
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: whether the instance perfectly matches the concept
        :rtype: boolean
        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        # ak = 'anchor'
        # if ak not in self.av_counts:
        #     return False

        # return (len(instance[ak]) == len(self.av_counts[ak]) and list(instance[ak])[0] == list(self.av_counts[ak])[0])

        for attr in set(instance).union(set(self.attrs())):
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
                    if val in self.av_counts[attr] and val in instance[attr]:
                        if abs(self.av_counts[attr][val] / self.count - instance[attr][val]) > 0.001:
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


'''tree = ContextualCobwebTree(1, 4)
data = list(generate_ms_sentence_variant_synonyms(3, 10, 50))
random.shuffle(data)
for sent in tqdm(data):
    tree.fit_to_text_wo_stopwords([word for word in sent if word[:-2] not in stop_words])
visualize(tree)'''


if __name__ == "__main__":

    tree = ContextualCobwebTree(4)

    for text_num in range(1):
        text = list(load_text(text_num))[:5000]

        text = [word for word in text if word not in stop_words]

        tree.ifit_text(text)

        visualize(tree)
        raise Exception('beep')

        # print(test_microsoft(tree))

        questions = create_questions(text, 10, 4, 200)
        tree.fit_to_text_wo_stopwords(text)

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

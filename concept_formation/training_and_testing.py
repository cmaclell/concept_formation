"""
The file contains the code for training contextual cobweb from large datasets
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from os.path import dirname
from os.path import join
import re
import random
from csv import reader
from concept_formation.preprocess_text import stop_words, _preprocess
from itertools import chain, product, islice
from collections import Counter


module_path = dirname(__file__)


def load_microsoft_qa():
    let_to_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    with open(join(module_path, 'data_files',
                   'MSC_testing_data.csv'), newline='') as questions:
        with open(join(module_path, 'data_files',
                       'MSC_test_answer.csv'), newline='') as answers:
            data = zip(reader(questions), reader(answers))
            # Remove top row
            data.__next__()
            for quest, ans in data:
                yield (list(_preprocess(quest[1], True)), quest[2:],
                       let_to_num[ans[1]])


def test_microsoft(model):
    correct = 0
    for total, (question, answers, answer) in enumerate(load_microsoft_qa()):
        if model.guess_missing(question, answers, 1) == answers[answer]:
            correct += 1
    total += 1
    return correct / total


def full_ms_sentences():
    return map(question_to_sentence, load_microsoft_qa())


def question_to_sentence(question):
    sent, quest, ans = question
    sent[sent.index(None)] = quest[ans]
    return sent


def generate_ms_sentence_variant_synonyms(nsynonyms=2, ncopies=5, nms_sentences=1500):
    """
    args:
        nsynonyms (int): number of possible synonyms
        ncopies (int): number of times each sentence appears"""
    sentences = list(full_ms_sentences())[:nms_sentences]
    for _ in range(ncopies):
        for sentence in sentences:
            yield synonymize(sentence, nsynonyms)


def synonymize(sentence, nsynonyms=2):
    return [(word+'-%s' % random.randint(1, nsynonyms) if word else None) for word in sentence]


def synonymize_question(question, nsynonyms=2):
    return (synonymize(question[0], nsynonyms), synonymize(question[1], nsynonyms), question[2])


def generate_ms_sentence_variant_homographs(homographs=[], nsenses=2, ncopies=5, nms_sentences=1500):
    """
    homographs (Seq): List of words to be turned into homographs
    nsenses (int): Number of senses for each homographs"""
    sentences = list(full_ms_sentences())[:nms_sentences]
    for _ in range(ncopies):
        for sentence in sentences:
            for i in range(nsenses):
                yield [('#%s#' % word if word in homographs else word+'-%s' % i) for word in sentence]


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


def word_to_base(word):
    return word.split('-')[0]


def word_to_suffix(word):
    if word[-1] == '#':
        return None
    return word.split('-')[1]


def word_to_hidden_homograph_attr(word, homo_type):
    return word[1:-1] + '-' + homo_type


def get_path(node):
    while node:
        yield node
        node = node.parent


def nearby_leaf_pairs(leaves, cutoff):
    for leaf_1, leaf_2 in product(leaves, repeat=2):
        first_path = list(get_path(leaf_1))[:cutoff]
        if any(node in first_path for node in islice(get_path(leaf_2), 0, cutoff)):
            yield (leaf_1, leaf_2)


def synonym_similarity(cutoff, word_to_leaf):
    """cutoff: maximum number of levels above before leaves are considered to far apart"""
    # The base of a word to the variants of that base
    base_to_variants = {}
    # The base of a word to the number of occurances
    base_to_counts = Counter()
    for variant, leaves in word_to_leaf.items():
        base = word_to_base(variant)
        base_to_variants.setdefault(base, set()).add(variant)
        base_to_counts[base] += sum(leaf.count for leaf in leaves)
    # Pick a base-word uniformly at random, then pick a leaf pair (could be two of the same leaf)
    # Expected number of pairs whose common ancestor is within cutoff nodes of both.
    result = 0
    for base, variants in base_to_variants.items():
        for node_1, node_2 in nearby_leaf_pairs(chain.from_iterable(word_to_leaf[variant] for variant in variants), cutoff):
            result += node_1.count * node_2.count / (base_to_counts[base] * base_to_counts[base])
    return result / len(base_to_variants)


def leaf_to_homograph_counts(leaf, anch_key, maj_key):
    # result = Counter(word_to_suffix(next(iter(ctxt_leaf.av_counts[anch_key]))) for ctxt_leaf in leaf.av_counts[maj_key].elements())
    # del result[None]
    return Counter(leaf.av_counts['_homograph'])


def homograph_difference(homographs, cutoff, word_to_leaf, anch_key, maj_key):
    # Given two of the same word, expected proportion
    # that have their common ancestor above cutoff generations above them or are the same.
    total = 0
    for homograph in homographs:
        temp = 0
        leaves = list(word_to_leaf[homograph])
        homograph_counts = {leaf: leaf_to_homograph_counts(leaf, anch_key, maj_key) for leaf in leaves}
        for leaf_1, leaf_2 in nearby_leaf_pairs(leaves, cutoff):
            temp += sum(homograph_counts[leaf_1].values()) * sum(homograph_counts[leaf_2].values()) - sum(count * homograph_counts[leaf_2][sense] for sense, count in homograph_counts[leaf_1].items())
        temp /= sum(sum(counts.values()) for counts in homograph_counts.values()) ** 2
        total += temp
        # Σ

    return 1 - total / len(homographs)

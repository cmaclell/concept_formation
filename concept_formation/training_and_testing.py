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


def generate_ms_sentence_variant_homographs(homographs=[], nsenses=2):
    """
    homographs (Seq): List of words to be turned into homographs
    nsenses (int): Number of senses for each homographs"""
    for sentence in full_ms_sentences():
        for i in range(nsenses):
            yield [(word if word in homographs else word+'-%s' % i) for word in sentence]


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

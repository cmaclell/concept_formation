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
from csv import reader


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


def load_text(file_num):
    with open(join(module_path, 'data_files', 'texts',
                   'Holmes_Training_Data', '%s.txt' % file_num)) as f:
        for word in _preprocess(f.read()):
            yield word


def _preprocess(text, blanks=False):
    latin_abbreviations = ('i.e.', 'e.g.', 'cf.')
    for abbrv in latin_abbreviations:
        text = text.replace(abbrv, '')
    is_word = re.compile(r'.??([a-zA-Z]+).??')
    # is_word_iter = re.compile(r'\w+')  includes numbers :(
    whitespace = re.compile(r'\s')
    # Rplaces hyphens and other punctuation with spaces
    # Removes urls, numbers, and other non-word junk
    for word in whitespace.split(text):
        if blanks and word == '_____':
            yield None
        match = is_word.fullmatch(word)
        if match:
            yield match.group(1).lower()


stop_words = {*"i me my myself we our ours ourselves you your yours yourself "
              "yourselves he him his himself she her hers herself it its "
              "itself they them their theirs themselves what which who whom "
              "this that these those am is are was were be been being have "
              "has had having do does did doing a an the and but if or "
              "because as until while of at by for with about against "
              "between into through during before after above below to from "
              "up down in out on off over under again further then once here "
              "there when where why how all any both each few more most "
              "other some such no nor not only own same so than too very s t "
              "can will just don't should now th".split(' ')}

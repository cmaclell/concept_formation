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
from time import time
from visualize import visualize


module_path = dirname(__file__)

def word_to_obj(word):
    return {'Anchor': word}


def _load_text(file_num):
    with open(join(module_path, 'data_files',
                   'texts', '%s.txt' % file_num)) as f:
        for word in _preprocess(f.read()):
            yield word


def _preprocess(text):
    latin_abbreviations = ('i.e.', 'e.g.', 'cf.')
    for abbrv in latin_abbreviations:
        text = text.replace(abbrv, '')
    is_word = re.compile(r'.??([-a-zA-Z]+).??')
    # is_word_iter = re.compile(r'\w+')  includes numbers :(
    whitespace = re.compile(r'\s')
    # Rplaces hyphens and other punctuation with spaces
    # Removes urls, numbers, and other non-word junk
    for word in whitespace.split(text):
        match = is_word.fullmatch(word)
        if match:
            yield match.group(1).lower()

"""
The file contains the code for training contextual cobweb from large datasets
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from os.path import dirname
from os.path import join
from concept_formation.contextual_cobweb import ContextualCobwebTree, ca_key
from cProfile import run
import re
from time import time
# import xml.etree.ElementTree as ET


run  # silence linting
module_path = dirname(__file__)
find_context = re.compile(r"'%s': {.*?}" % ca_key)


def ellide_context(string):
    return re.sub(find_context, "'%s': {...}" % ca_key, string)


def print_tree(tree, ctxt=True):
    if ctxt:
        print(str(tree).replace('Node', 'N'))
    else:
        print(ellide_context(str(tree)))


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


window_size = 4
context_weight = 2.5
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
               "can will just don't should now".split(' ')}
# Thanks Porter... Thorter
tree = ContextualCobwebTree(ctxt_weight=context_weight)
for text_num in range(1):
    text = [word_to_obj(word)
            for word in _load_text(text_num) if word not in stop_words]
    print('iterations needed', len(text))
    start = time()
    run("tree.contextual_ifit(text, context_size=window_size)")
    print(time()-start)
    print(text_num)
print_tree(tree, ctxt=False)

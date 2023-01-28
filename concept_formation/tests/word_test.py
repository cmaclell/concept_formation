import json
import sys
import os
import re
from collections import Counter
from random import shuffle

from tqdm import tqdm
from spacy.lang.en import English

# from concept_formation.word_cobweb import CobwebWordTree
from concept_formation.cobweb import CobwebTree
from concept_formation.visualize import visualize

from os.path import dirname
from os.path import join

serialization_filename = 'serialization.json'

nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.tokenizer

def get_instance(text, anchor_idx, anchor_wd, window=5):
    # ctx = text[max(0, anchor_idx-window):anchor_idx] + text[anchor_idx+1:anchor_idx+1+window]
    # ctx = Counter(ctx)
    # ctx = {attr: ctx[attr]/len(ctx) for attr in ctx}

    # if anchor_wd is None:
    #     return {'context': ctx}

    # return {'anchor': {anchor_wd: 1}, 'context': ctx}

    ctx = text[max(0, anchor_idx-window):anchor_idx] + text[anchor_idx+1:anchor_idx+1+window]
    ctx = set(ctx)
    example = {word: "T" for word in ctx}

    if anchor_wd is None:
        return example

    example['##anchor##'] = anchor_wd
    return example

def get_text_instances(text, window=5):
    
    instances = []

    for anchor_idx, anchor_wd in enumerate(text):
        if anchor_wd is None:
            continue
        instances.append(get_instance(text, anchor_idx, anchor_wd, window=window))

    return instances

def preprocess_text(text):
    punc = re.compile(r"[^a-zA-Z0-9\s]")
    whitespace = re.compile(r"\s+")
    text = punc.sub("", text)
    text = whitespace.sub(" ", text)
    text = text.strip().lower()
    return text


def training_texts():

    training_dir = "/Users/cmaclellan3/Projects/Microsoft-Sentence-Completion-Challenge/data/raw_data/Holmes_Training_Data"

    file_counts = []
    ranked_names = [(57686, 'LESMS10.TXT'), (38594, '1DONQ10.TXT'), (35690, 'VFAIR10.TXT'), (33805, '2DFRE10.TXT'), (33052, 'DOMBY10.TXT'), (31256, 'CPRFD10.TXT'), (30900, '4DFRE10.TXT'), (30193, 'LDORT10.TXT'), (30181, '1DFRE10.TXT'), (30131, 'CHUZZ10.TXT')]

    for path, subdirs, files in os.walk(training_dir):

        for idx, (_, name) in enumerate(ranked_names[:1]):
            print("Processing file {} of {}".format(idx, len(files)))
            if not re.search(r'^[A-Z0-9]*.TXT$', name):
                continue
            print(name)
            with open(os.path.join(path, name), 'r', encoding='latin-1') as fin:
                skip = True
                text = ""
                for line in fin:
                    if not skip:
                        text += line
                    elif "*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS" in line:
                        skip = False
                        
                text = preprocess_text(text)
                doc = tokenizer(text)
                # doc = " ".join([token.text for token in doc if 
                #                 # not token.is_stop and 
                #                 not token.is_punct])
                text = [token.text for token in doc] #  if not token.is_stop]
                yield text

                # counter = Counter(text)
                # answer_count = sum([counter[a] for a in answers])
                # file_counts.append((answer_count, name))

                # for answer in answers:
                #     if answer in text:
                #         print("answer contained", answer)
                #         yield text
                #         break
                # for token in doc:
                #     print(token.text, token.pos_, token.dep_)

def get_microsoft_test_items():
    question_file = "/Users/cmaclellan3/Projects/Microsoft-Sentence-Completion-Challenge/data/raw_data/testing_data.csv"
    answer_file = "/Users/cmaclellan3/Projects/Microsoft-Sentence-Completion-Challenge/data/raw_data/test_answer.csv"

    items = {}
    with open(question_file, 'r') as fin:
        first = True
        for line in fin:
            if first:
                first = False
                continue

            line = line.replace('"60,000"', '60000').replace('"1,200"', "1200").replace("-", "").replace("'", "").strip().lower().split(",")

            item = line[0]
            question = ",".join(line[1:-5])
            answers = line[-5:]

            # print(line)
            # print(answers)
            # print()
            items[item] = {'question': question,
                           'answers': answers}

    with open(answer_file, 'r') as fin:
        first = True
        for line in fin:
            if first:
                first = False
                continue
            line = line.strip().split(",")

            key = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
            items[line[0]]['answer'] = items[line[0]]['answers'][key[line[1]]]

    return items

def evaluate_tree_on_test(tree, test_items):
    for item in test_items:
        item = test_items[item]
        sent = item['question'].replace('"', '').replace("_____", "_")

        punc = re.compile(r"[^_a-zA-Z0-9\s]")
        whitespace = re.compile(r"\s+")
        text = punc.sub("", sent)
        text = whitespace.sub(" ", text)
        text = text.strip().lower()

        doc = tokenizer(text)
        text = [token.text for token in doc] # if not token.is_stop]
        idx = text.index('_')
        instance = get_instance(text, idx, None)

        leaf = tree.categorize(instance)
        basic = leaf.get_basic_level()

        leaf_pred = leaf.predict("##anchor##")
        basic_pred = basic.predict("##anchor##")

        root_weights = sorted([(tree.root.av_counts['##anchor##'][w]/tree.root.count if w in tree.root.av_counts['##anchor##'] else 0.0, w) for w in item['answers']], reverse=True)
        leaf_weights = sorted([(leaf.av_counts['##anchor##'][w]/leaf.count if w in leaf.av_counts['##anchor##'] else 0.0, w) for w in item['answers']], reverse=True)
        basic_weights = sorted([(basic.av_counts['##anchor##'][w]/basic.count if w in basic.av_counts['##anchor##'] else 0.0, w) for w in item['answers']], reverse=True)

        if root_weights[0][0] > 0: 
            print(text)
            print(item['answers'])
            print(instance)
            print('root', root_weights)
            print('basic', basic_weights)
            print('leaf', leaf_weights)
            print('actual', item['answer'])
            print()

if __name__ == "__main__":

    test_items = get_microsoft_test_items()

    answers = set([a for idx in test_items for a in test_items[idx]['answers']])

    tree = CobwebTree()

    for text in training_texts():
        examples = [e for e in get_text_instances(text)]
        shuffle(examples)
        examples = examples[:30000]
        
        for i, example in enumerate(tqdm(examples)):
            tree.ifit(example)

        # visualize(tree)

        js = tree.root.save_json()
        with open(join(dirname(__file__), serialization_filename), 'w+') as f:
            f.write(js)

        # The code below just serves as a test/demo for JSON saving and loading
        # tree.clear()
        # tree.load_json(js)
        # new_js = tree.root.save_json()

        # js_norm = json.dumps(json.loads(js), sort_keys=True)
        # new_js_norm = json.dumps(json.loads(new_js), sort_keys=True)

        # print('same JS? %s' % (js_norm == new_js_norm))

    # from pprint import pprint
    # pprint(tree.root.av_counts['##anchor##'])
    # print('"walking" COUNT', tree.root.av_counts['##anchor##']['walking'])
    evaluate_tree_on_test(tree, test_items)

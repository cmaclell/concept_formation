import json
import sys
import os
import re
import itertools
from collections import Counter
from random import shuffle
from random import random
from time import sleep

from tqdm import tqdm
import spacy

from concept_formation.multinomial_cobweb import MultinomialCobwebTree
from concept_formation.visualize import visualize
from concept_formation.ngram import NGram
from concept_formation.ngram import NGramBayes

from os.path import dirname
from os.path import join

serialization_filename = 'serialization.json'

nlp = spacy.load("en_core_web_sm", disable = ['parser'])
# nlp = spacy.load('en_core_web_trf')
nlp.add_pipe("sentencizer")
nlp.max_length = float('inf')
# nlp.add_pipe("lemmatizer")
# nlp.initialize()

print(nlp.pipe_names)
 
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
# tokenizer = nlp.tokenizer

remove_words = set(['thou', 'thy', 'thee', 'shall', 'unto', 'o', 'ye', 'nt',
                    't', 'th', 'yeh', 's', 'oh', 'got', 'ses', 'em', 'm', 've',
                    'd', 'de', 'p', 'll', 'wo', 'c', 'l', 'ai', 'ii', 'ah',
                    'i', 'st', 'l', 'iv', 'ere', 'et', 'ay', 'v', 'eh', 'en',
                    'le', 'jo', 'ing', 'b', 'vi', 'ter', 'yer', 'vii', 'a',
                    'h', 'x', 'viii', 'sha', 'fer', 'vol', 'e', 'n', 'wi',
                    'ix', 'j', 'g', 's', 'w', 'wuz', 'er', 'du', 't', 'xi',
                    'xii', 'xiv', 'yo', 'dis', 'y', 'lo', 'd', 'f', 'xiii',
                    'xv', 'o', 'xvi', 'r', 'mme', 'ne', 'didst', 'dey',
                    'ex', 'anon', 'afar', 'sur', 'ole', 'sed', 'si', 'xviii',
                    'canst', 'tion', 'numa', 'xvii', 'ud', 'com', 'thar',
                    'mis', 'erec', 'hon', 'qui', 'yaqui', 'xx', 'din', 'yon',
                    'xxii', 'dian', 'xxi', 'mo', 'ee', 'ef', 'co', 'ut', 'vv',
                    'ado', 'wat', 'ment', 'hal', 'ozmas', 'mag', 'que', 'xxiv',
                    'vit', 'el', 'jes', 'hae', 'huh', 'ork', 'ki', 'ap', 'ja',
                    'viz', 'il', 'hm', 'quay', 'ed', 'ain', 'nae', 'ave', 'lor',
                    'sabe', 'xxvii', 'mien', 'vie', 'xxvi', 'se', 'je', 'gre', 
                    'u', 'iti', 'deh', 'neer', 'oft', 'xxviii', 'duc', 'ony',
                    'au', 'und', 'ould', 'vith', 'oo', 'q', 'dem', 'kep', 'youi',
                    'and', 'cos', 'sid', 'xxix', 'ac', 'na', 'fo', 'te', 'rd', 
                    'ie', 'quae', 'tal', 'hev', 'ugu', 'noi', 'sae', 'ull', 'ce'
                    'dei', 'ta', 'xxxiii', 'alls', 'ou', 'es', 'pm', 'zat', 'thern',
                    'izz', 'ful', 'roun', 'giry', 'vas', 'gos', 'tions',
                    'enide', 'afeard', 'gat', 'muda', 'ob', 'mi', 'ont', 'aa',
                    'xxxiv', 'quo', 'youyou', 'xxxii', 'xxxi', 'xxx', 'xxv', 'xxiii',
                    'xix', 'un', 'um', 'une'])

with open('answers.json', 'r') as fin:
    answers = set(json.load(fin))

with open('frequency.json', 'r') as fin:
    frequency = json.load(fin)
    for word in frequency:
        if frequency[word] < 80 and word not in answers:
            remove_words.add(word)

def get_instance(text, anchor_idx, anchor_wd, window=8):
    ctx = text[max(0, anchor_idx-window):anchor_idx] + text[anchor_idx+1:anchor_idx+1+window]
    ctx = Counter(ctx)
    example = {}
    example['context'] = {word: ctx[word] for word in ctx}

    if anchor_wd is None:
        return example

    example['anchor'] = {anchor_wd: 1}
    return example

def get_text_instances(text, window=8):
    
    instances = []

    for anchor_idx, anchor_wd in enumerate(text):
        if anchor_wd is None:
            continue
        instances.append(get_instance(text, anchor_idx, anchor_wd, window=window))

    return instances

def preprocess_text(text, test=False):
    if test:
        punc = re.compile(r"[^_a-zA-Z,.!?:;\s]")
    else:
        punc = re.compile(r"[^a-zA-Z,.!?:;\s]")
    whitespace = re.compile(r"\s+")
    text = punc.sub("", text)
    text = whitespace.sub(" ", text)
    text = text.strip().lower()
    return text


def process_text(text, test=False):

    text = preprocess_text(text, test=test)
    doc = nlp(text)

    if test:
        char_only = re.compile(r"[^a-z_]")
    else:
        char_only = re.compile(r"[^a-z]")

    output = []
    for sent in doc.sents:
        out = [char_only.sub("", token.lemma_) if token.text not in answers else token.text for token in sent if (test and token.text == "_") or
               (not token.is_punct and not token.is_stop)]
        out = [w for w in out if w not in remove_words]
        output.append(out)
    return output


def training_texts():

    training_dir = "/home/cmaclellan3/Microsoft-Sentence-Completion-Challenge/data/raw_data/Holmes_Training_Data"

    for path, subdirs, files in os.walk(training_dir):

        shuffle(files)

        for idx, name in enumerate(files):
            print("Processing file {} of {}".format(idx, len(files)))
            if not re.search(r'^[A-Z0-9]*.TXT$', name):
                continue
            print(name)
            with open(os.path.join(path, name), 'r', encoding='latin-1') as fin:
                skip = True
                text = ""
                for line in fin:
                    if not skip and not "project gutenberg" in line.lower():
                        text += line
                    elif "*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS" in line:
                        skip = False

                output = process_text(text)

                yield output

def get_microsoft_test_items():
    question_file = "/home/cmaclellan3/Microsoft-Sentence-Completion-Challenge/data/raw_data/testing_data.csv"
    answer_file = "/home/cmaclellan3/Microsoft-Sentence-Completion-Challenge/data/raw_data/test_answer.csv"

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
    accuracy = []

    for item in test_items:
        item = test_items[item]
        sent = item['question'].replace('"', '').replace("_____", "_")

        punc = re.compile(r"[^_a-zA-Z\s]")
        sent = punc.sub("", sent)
        output = process_text(sent, test=True)

        assert len(output) == 1
        text = output[0]

        idx = text.index('_')
        instance = get_instance(text, idx, None)

        basic = tree.categorize(instance)
        # basic = tree.categorize.get_best_level(instance)

        from pprint import pprint
        has_pred = False
        while not has_pred and basic.parent:
            for w in item['answers']:
                if (w in basic.av_counts['anchor']):
                    has_pred = True
                    break
            else:
                basic = basic.parent

        preds = sorted([(basic.av_counts['anchor'][w]/basic.attr_counts['anchor'] if (w in basic.av_counts['anchor']) else 0.0, random(), w) for w in item['answers']], reverse=True)

        accuracy.append(int(preds[0][2] == item['answer']))

    print("Accuracy", sum(accuracy) / len(accuracy))
    print()

def generate_frequency_json():
    x = Counter([w for sentences in training_texts() for s in sentences for w in s])
    from pprint import pprint
    pprint(x.most_common())
    
    freq = dict(x)
    with open('frequency.json', 'w') as fout:
        json.dump(freq, fout, indent=4)

def generate_answers_json():
    test_items = get_microsoft_test_items()
    answers = list(set([a for idx in test_items for a in test_items[idx]['answers']]))

    with open('answers.json', 'w') as fout:
        json.dump(answers, fout, indent=4)

if __name__ == "__main__":
    # generate_frequency_json()
    # generate_answers_json()

    test_items = get_microsoft_test_items()

    # tree = MultinomialCobwebTree(sizes={'anchor': 3700, 'context': 13000}, device="cuda:1")
    tree = MultinomialCobwebTree()

    with open('examples.json', 'r') as fin:
        examples = json.load(fin)
        examples = examples[:1500]

        for i, example in enumerate(tqdm(examples)):
            if sum(example['context'][w] for w in example['context']) > 0:
                # print(example)
                # subset = {}
                # subset['anchor'] = example['anchor']
                # subset['context'] = example['context']
                # tree.ifit(subset)
                tree.ifit(example)
                # if i % 3000 == 0:
                #     visualize(tree)
                #     sleep(0.25)
                # input("Press Enter to continue...")
                # sleep(1)

        print("Anchor vocab size: ", len(tree.root.av_counts['anchor']))
        print("Context vocab size: ", len(tree.root.av_counts['context']))
        # visualize(tree)
        # break

        evaluate_tree_on_test(tree, test_items)
        # break

        # with open('checkpoint.json', 'w') as fout:
        #     fout.write(tree.dump_json());

        # js = tree.root.save_json()
        # with open(join(dirname(__file__), serialization_filename), 'w+') as f:
        #     f.write(js)

        # The code below just serves as a test/demo for JSON saving and loading
        # tree.clear()
        # tree.load_json(js)
        # new_js = tree.root.save_json()

        # js_norm = json.dumps(json.loads(js), sort_keys=True)
        # new_js_norm = json.dumps(json.loads(new_js), sort_keys=True)

        # print('same JS? %s' % (js_norm == new_js_norm))

    # from pprint import pprint
    # pprint(tree.root.av_counts['anchor'])
    # print('"walking" COUNT', tree.root.av_counts['anchor']['walking'])
    # evaluate_tree_on_test(tree, test_items)

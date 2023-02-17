import xml.etree.ElementTree as ET
import sys
import os
import re
from collections import Counter
from spacy.lang.en import English
from tqdm import tqdm 
import json
from concept_formation.cobweb import CobwebTree
from concept_formation.multinomial_cobweb import MultinomialCobwebTree
from concept_formation.visualize import visualize
from random import shuffle
from sklearn.metrics.cluster import adjusted_rand_score as ari

nlp = English()
tokenizer = nlp.tokenizer


def get_instances(text, word_categories, valid_words, window=5):
    vocab = set(word_categories).union(valid_words)

    sent = [word for word in text if word in word_categories or word in valid_words]

    for anchor_idx, word in enumerate(sent):
        if word in word_categories:
            ctx = sent[max(0, anchor_idx-window):anchor_idx] + sent[anchor_idx+1:anchor_idx+1+window]
            ctx = set(ctx)
            ctx = Counter(ctx)
            # example = {word: "T" if word in ctx else "F" for word in vocab}
            # example = {word: "T" for word in ctx}
            example = {}
            example['anchor'] = {word: 1}
            example['context'] = {word: ctx[word] for word in ctx}
            # example['_word_category'] = word_categories[word]
            yield example

def get_words_and_categories(word_category_data):

    with open(word_category_data, 'r') as fin:
        for line in fin:
            line = line.split()
            if line[0] == 'ID#' or line[2] != "object":
                continue
            yield line[1].lower(), line[3].lower()

def read_bnc_files(training_dir):
    files = [os.path.join(path, name) for path, subdirs, files in os.walk(training_dir) for idx, name in enumerate(files) if re.search(r'^[A-Z0-9]*.xml$', name)]

    for filename in tqdm(files): 
        for sent in read_bnc_file(filename):
            yield sent
            
        # for idx, (_, name) in enumerate(ranked_names[:1]):
        # for idx, name in enumerate(files):


def read_bnc_file(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    text = []
    for child in root.iter('s'):
        text += [word.attrib['hw'].lower() for word in child.findall('w') 
                 if re.search(r'[a-z]+', word.attrib['hw'].lower())]
    yield text

def generate_bnc_examples():
    bnc_training_data = "/Users/cmaclellan3/Projects/concept_formation/concept_formation/tests/bnc_data/baby_bnc/Texts/"

    word_category_data = "/Users/cmaclellan3/Projects/concept_formation/concept_formation/tests/bnc_data/Vinson-BRM-2008/word_categories.txt"

    word_categories = {word: category for word, category in get_words_and_categories(word_category_data)}

    sentences = []
    text = []
    for sent in read_bnc_files(bnc_training_data):
        sentences.append(sent)
        text += sent
        # print()
        # print(sent)
        # for example in get_instances(sent, word_categories):
        #     print(example)
    counter = Counter(text)
    print(counter.most_common(60))
    print("Common words not on stop list")
    for word, count in counter.most_common(60):
        if word not in nlp.Defaults.stop_words:
            print(word, count)

    valid_words = set(w for w in counter if counter[w] > 800 and w not in nlp.Defaults.stop_words)
    print(word_categories)
    print(valid_words)
    print("# of valid anchor words", len(word_categories))
    print("# of valid ctx words", len(valid_words))

    examples = []
    for sent in sentences:
        for example in get_instances(sent, word_categories, valid_words):
            examples.append(example)

    print("# of example", len(examples))

    with open("bnc_examples.json", 'w') as fout:
        json.dump(examples, fout, indent=4)


if __name__ == "__main__":
    word_category_data = "/Users/cmaclellan3/Projects/concept_formation/concept_formation/tests/bnc_data/Vinson-BRM-2008/word_categories.txt"
    word_categories = {word: category for word, category in get_words_and_categories(word_category_data)}

    # generate_bnc_examples()

    with open("bnc_examples.json", "r") as fin:
        examples = json.load(fin)

    anchor_count = Counter()
    for example in examples:
        for word in example['anchor']:
            anchor_count[word] += 1
    from pprint import pprint
    # print(anchor_count)

    top_n = list(word for word, _ in anchor_count.most_common(300))
    # top_n = ["book", "head", "hand", "eye"]
    print(top_n)
    examples = [e for e in examples if list(e['anchor'])[0] in top_n]
    # examples = [e for e in examples if anchor_count[e['##anchor##']] > 100]

    shuffle(examples)
    test = examples[7000:8000]
    examples = examples[:3000]

    tree = MultinomialCobwebTree()
    print('Fitting tree')
    accuracy = []
    for example in tqdm(examples):
        leaf = tree.categorize({a: {w: example[a][w] for w in example[a]} for a
                                in example if a != "anchor"})
        pred = leaf.predict("anchor")
        accuracy.append(int(pred == list(example["anchor"])[0]))
        # print(example["##anchor##"], pred, leaf.concept_id)
        tree.ifit(example)
    # print(accuracy)
    print("Accuracy = {}".format(sum(accuracy)/len(accuracy)))

    visualize(tree)

    accuracy = []
    cobweb_preds = []

    from math import log

    for example in tqdm(test):
        leaf = tree.categorize({a: {w: example[a][w] for w in example[a]} for a
                                in example if a != "anchor"})
        assert len(leaf.children) == 0
        pred = leaf.predict("anchor")
        accuracy.append(int(pred == list(example["anchor"])[0]))
        cobweb_preds.append(top_n.index(pred)) 
    print("Test Accuracy = {}".format(sum(accuracy)/len(accuracy)))

    print()
    print("Testing Naive Bayes")

    y = [top_n.index(list(e['anchor'])[0]) for e in examples]
    X = [{w: example['context'][w] for w in example['context']} for example in
          examples]

    from sklearn.feature_extraction import DictVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.naive_bayes import BernoulliNB
    from concept_formation.multinomialNB import MultiNB
    import numpy as np
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(X)
    # model = BernoulliNB()
    sk_model = MultinomialNB()
    model = MultiNB()
    model.fit(X, y)
    sk_model.fit(X, y)

    testy = [top_n.index(list(e['anchor'])[0]) for e in test]
    testX = [{w: example['context'][w] for w in example['context']} for example
              in test]
    testX = dv.transform(testX)
    pred = model.predict(testX)
    sk_pred = sk_model.predict(testX)
    
    diff = pred == testy
    print("NB Accuracy", diff.astype(int).mean())

    diff = sk_pred == testy
    print("SK NB Accuracy", diff.astype(int).mean())

    num_anchors = 0
    for i, v in enumerate(top_n):
        if (np.array(y) == i).mean() > 0:
            num_anchors += 1
        print("Training Prob of {} = {}".format(v, (np.array(y) == i).mean()))
        print("Testing Prob of {} = {}".format(v, (np.array(testy) == i).mean()))

    print("Num Anchors", num_anchors)

    raise Exception("")

    cobweb_preds = np.array(cobweb_preds)
    for i, b in enumerate(cobweb_preds != pred):
        if b:
            print(cobweb_preds[i])
            print(pred[i])

            print(test[i])
            singleX = [{w: test[i]['context'][w] for w in test[i]['context']}]
            print("MultiNB", top_n[model.predict(dv.transform(singleX))[0]])
            print("Multi labels", top_n)
            print("Multi priors", model.prior)
            print("Multi Probs", model.predict_proba(dv.transform(singleX)))
            print("Sklearn", top_n[sk_model.predict(dv.transform(singleX))[0]])
            leaf = tree.categorize({a: {w: test[i][a][w] for w in
                                        test[i][a]} for a in test[i] if
                                    a != "anchor"})
            print("COBWEB", leaf.predict("anchor"))

            for i, child in enumerate(tree.root.children):
                print('child{}'.format(i), tree.root.children[i].av_counts['anchor'])
                print('child{}'.format(i), tree.root.children[i].log_prob_class_given_instance({a: {w: test[i][a][w] for w in test[i][a]} for a in test[i] if a != "anchor"}))

            print()

    # pred = []
    # actual = []
    # label = {v: i for i, v in enumerate(word_categories.values())}
    # for word in word_categories:
    #     leaf = tree.categorize({"##anchor##": word})
    #     basic = leaf.get_basic_level()
    #     pred.append(basic.concept_id)
    #     # pred.append(leaf.concept_id)
    #     actual.append(label[word_categories[word]])

    # print(pred)
    # print(actual)
    # print("ARI SCORE:", ari(pred, actual))
        
  


import os
import re
import json
from collections import Counter
from random import shuffle
from random import random
from tqdm import tqdm
from concept_formation.multinomial_cobweb import MultinomialCobwebTree
from concept_formation.visualize import visualize
import spacy
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
from time import time
from datetime import datetime
from pprint import pprint
from multiprocessing import Pool


nlp = spacy.load("en_core_web_sm", disable = ['parser'])
# nlp = spacy.load('en_core_web_trf')
nlp.add_pipe("sentencizer")
nlp.max_length = float('inf')
# nlp.add_pipe("lemmatizer")
# nlp.initialize()

# import logging  # Setting up the loggings to monitor gensim
# logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

def get_instance(text, anchor_idx, anchor_wd, window):
    text = [w.lower() for w in text]
    anchor_wd = anchor_wd.lower()
    before_text = text[max(0, anchor_idx-window):anchor_idx]
    after_text = text[anchor_idx+1:anchor_idx+1+window]
    ctx_text = before_text + after_text
    ctx = {}
    before_ctx = {}
    after_ctx = {}
    for i, w in enumerate(before_text):
        before_ctx[w] = 1/abs(len(before_text) - i)
        ctx[w] = 1/abs(len(before_text) - i)
    for i, w in enumerate(after_text):
        after_ctx[w] = 1/(i+1)
        ctx[w] = 1/(i+1)
    example = {}
    example['context'] = ctx
    # example['ctx_before'] = before_ctx
    # example['ctx_after'] = after_ctx

    if anchor_wd is None:
        return example

    example['anchor'] = {anchor_wd: 1}
    return example

def get_instances(story, window):
    for anchor_idx, anchor_wd in enumerate(story):
        yield anchor_idx, get_instance(story, anchor_idx, anchor_wd, window=window)

def process_story(line):
    line = line.replace("\n", "").split("\t")

    story = []
    for sent in line[2:]:
        # word_char = re.compile(r"[^_a-zA-Z,.!?:';\s]")
        word_char = re.compile(r"[^_a-zA-Z\s]")
        whitespace = re.compile(r"\s+")
        sent = word_char.sub("", sent)
        sent = whitespace.sub(" ", sent)

        # sent = " ".join(sent)
        # sent = nlp(sent)
        # sent = [token.lemma_.lower() for token
        #         in sent if (not token.is_punct and not token.is_stop)]

        # story.append(sent)
        story += sent.split()

    story = " ".join(story)
    story = nlp(story)
    story = [token.lemma_.lower() for token in story if (not token.is_punct and
                                                         not token.is_stop)]
    # story = [token.text.lower() for token in story if (not token.is_punct)]
    return story

def get_roc_stories(limit=None):
    with open("../data/ROCStories_winter2017 - ROCStories_winter2017.txt", 'r') as fin:

        lines = list(fin)
        if limit is None:
            limit = len(lines)-1

        with Pool(8) as p:
            stories = p.map(process_story, tqdm(lines[1:limit+1]))
            return stories
            # sents = [s for story in stories for s in story]
            # pprint(sents[:10])
            # return sents

        # for line in tqdm(lines[1:limit+1]):

        #     line = line.lower().replace("\n", "").split("\t")

        #     story = []
        #     for sent in line[2:]:

        #         word_char = re.compile(r"[^_a-zA-Z,.!?:';\s]")
        #         sent = word_char.sub("", sent)
        #         words = sent.split()
        #         story += words

        #     story = " ".join(story)
        #     story = nlp(story)
        #     story = [token.lemma_ for token in story if (not token.is_punct and
        #                                                  not token.is_stop)]
        #     yield story

if __name__ == "__main__":

    tree = MultinomialCobwebTree(alpha=0.0000001, weight_attr=False, objective=2, children_norm=True) 

    occurances = Counter()
    n_training_words = 0
    window = 10

    if not os.path.isfile("roc_stories.json"):
        print("Reading and preprocessing stories.")
        stories = list(get_roc_stories())
        with open("roc_stories.json", "w") as fout:
            json.dump(stories, fout, indent=4)
        print("done.")
    else:
        print("Loading preprocessed stories.")
        with open("roc_stories.json", "r") as fin:
            stories = json.load(fin)
        print("done.")

    # pprint(stories[:10])

    overall_freq = Counter([w for s in stories for w in s])

    # pprint(overall_freq.most_common(1000))

    # TODO PICK OUTFILE NAME
    outfile = 'cobweb_freq_5_rocstories_out'

    with open(outfile + ".csv", 'w') as fout:
        fout.write("n_training_words,n_training_stories,model,word,word_freq,word_obs_count,vocab_size,pred_word,prob_word,correct,story\n")

    instances = []

    for story_idx, story in enumerate(tqdm(stories)):
        # drop low frequency words
        story = [w for w in story if overall_freq[w] >= 200]
        if len(story) <= 3:
            continue

        for anchor_idx, instance in get_instances(story, window=window):
            instances.append((instance, story, anchor_idx))

    shuffle(instances)
    training_queue = []
    last_checkpoint_time = time()

    for idx, (instance, story, anchor_idx) in enumerate(tqdm(instances)):
        if idx > 1 and idx % 50000 == 0:
            visualize(tree)

        actual_anchor = list(instance['anchor'].keys())[0]
        text = " ".join([w for w in story[max(0, anchor_idx-window):anchor_idx]])
        text += " _ "
        text += " ".join([w for w in story[max(0, anchor_idx+1):anchor_idx+window+1]])

        ## cobweb
        # no_anchor = {'context': {cv: instance['context'][cv] for cv in
        #                          instance['context']}}
        no_anchor = {'context': instance['context']}
        # no_anchor = {'ctx_before': instance['ctx_before'], 'ctx_after': instance['ctx_after']}
        probs = tree.categorize(no_anchor).predict_probs()
        # probs = tree.categorize(no_anchor).get_best_level(no_anchor).predict_probs()
        p = 0
        best_word = "NONE"

        if 'anchor' in probs and actual_anchor in probs['anchor']:
            p = probs['anchor'][actual_anchor]

        if 'anchor' in probs and len(probs['anchor']) > 0:
            best_word = sorted([(probs['anchor'][w], random(), w) for w in probs['anchor']], reverse=True)[0][2]

        # print(no_anchor['context'])
        # print(best_word, 'vs', actual_anchor)
        # print()

        # Append to training queue so we only predict on things that are
        # completely outside the context window. I'm trying to prevent any kind of
        # cheating that cobweb might do that word2vec can't.
        training_queue.append(instance)

        with open(outfile + ".csv", 'a') as fout:
            fout.write("{},{},cobweb,{},{},{},{},{},{},{},{}\n".format(n_training_words,
                                                             story_idx,
                                                             actual_anchor,
                                                             overall_freq[actual_anchor],
                                                             occurances[actual_anchor],
                                                             len(occurances),
                                                             best_word,
                                                             p,
                                                             1 if best_word == actual_anchor else 0,
                                                             text))

        if len(training_queue) > 0:
            old_inst = training_queue.pop(0)
            tree.ifit(old_inst)
            old_anchor = list(old_inst['anchor'].keys())[0]
            occurances[old_anchor] += 1
            n_training_words += 1

        # if len(training_queue) > window:
        #     old_inst = training_queue.pop(0)
        #     tree.ifit(old_inst)
        #     old_anchor = list(old_inst['anchor'].keys())[0]
        #     occurances[old_anchor] += 1
        #     n_training_words += 1

        if (time() - last_checkpoint_time) > 3600:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open('{}-{}-{}.json'.format(outfile, story_idx, timestamp), 'w') as fout:
                fout.write(tree.dump_json())
                last_checkpoint_time = time()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open('{}-{}-{}-final.json'.format(outfile, story_idx, timestamp), 'w') as fout:
        fout.write(tree.dump_json())
        last_checkpoint_time = time()

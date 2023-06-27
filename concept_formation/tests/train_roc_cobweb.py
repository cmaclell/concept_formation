import os
import re
# import json
import string
from collections import Counter
from random import shuffle
# from random import random
from tqdm import tqdm
from concept_formation.multinomial_cobweb import MultinomialCobwebTree
# from concept_formation.visualize import visualize
import spacy
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from matplotlib import pyplot as plt
# from gensim.models import Word2Vec
# from gensim.models.word2vec import LineSentence
from multiprocessing import Pool
# from time import time
from time import perf_counter
# from datetime import datetime
from functools import partial
# import itertools
# from pprint import pprint

nlp = spacy.load("en_core_web_sm", disable=['parser'])
# nlp = spacy.load('en_core_web_trf')
nlp.add_pipe("sentencizer")
nlp.max_length = float('inf')
# nlp.add_pipe("lemmatizer")
# nlp.initialize()

# import logging  # Setting up the loggings to monitor gensim
# logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


def get_instance(text, anchor_idx, anchor_wd, window):
    ctx = text[max(0, anchor_idx - window):anchor_idx] + text[anchor_idx + 1:anchor_idx + 1 + window]
    ctx = Counter(ctx)
    example = {}
    example['context'] = {word: ctx[word] for word in ctx}

    if anchor_wd is None:
        return example

    example['anchor'] = {anchor_wd: 1}
    return example


def get_instance_list(story, window):
    return list(get_instances(story, window))


def get_instances(story, window):
    for anchor_idx, anchor_wd in enumerate(story):
        yield anchor_idx, get_instance(story, anchor_idx, anchor_wd, window=window)


def get_raw_roc_stories(limit=None):
    with open("ROCStories_winter2017 - ROCStories_winter2017.txt", 'r') as fin:

        lines = list(fin)
        if limit is None:
            limit = len(lines) - 1

        for line in tqdm(lines[1:limit + 1]):

            line = line.lower().replace("\n", "").split("\t")

            story = []
            for sent in line[2:]:

                word_char = re.compile(r"[^_a-zA-Z,.!?:';\s]")
                sent = word_char.sub("", sent)
                words = sent.split()
                story += words

            story = " ".join(story)
            story = re.sub('['+re.escape(string.punctuation)+']', '', story).split()
            yield story

def get_roc_stories(limit=None):
    with open("ROCStories_winter2017 - ROCStories_winter2017.txt", 'r') as fin:

        lines = list(fin)
        if limit is None:
            limit = len(lines)-1

        for line in tqdm(lines[1:limit+1]):

            line = line.lower().replace("\n", "").split("\t")

            story = []
            for sent in line[2:]:

                word_char = re.compile(r"[^_a-zA-Z,.!?:';\s]")
                sent = word_char.sub("", sent)
                words = sent.split()
                story += words

            story = " ".join(story)
            story = nlp(story)
            story = [token.lemma_ for token in story if (not token.is_punct and
                                                         not token.is_stop)]
            yield story

if __name__ == "__main__":

    tree = MultinomialCobwebTree(True, # Use mutual information (rather than expected correct guesses)
                                 1, # alpha weight
                                 True, # dynamically compute alpha
                                 True) # weight attr by avg occurance of attr

    occurances = Counter()
    n_training_words = 0
    window = 6

    print("Loading ROC stories")
    with Pool(processes=os.cpu_count()-2) as p:
        stories = list(get_raw_roc_stories())[:5000]
        f = partial(get_instance_list, window=window)
        # print(stories[:5])
        story_instances = p.map(f, tqdm(stories))
        # pprint(story_instances[:4])
        instances = [i for si in story_instances for _, i in si]

    instances = instances[:5000]
    shuffle(instances)

    # instances = [instance for story in get_raw_roc_stories()
    #         for _, instance in get_instances(story, window=window)]

    print("training synchronously")
    start = perf_counter()
    fut1t = [tree.ifit(i) for i in tqdm(instances[:len(instances)])]
    result1t = [f.wait() for f in tqdm(fut1t)]
    end = perf_counter()
    print("Done in {}".format(end - start))
    # print("AV key write wait time: {}".format(tree.av_key_wait_time))
    # print("Tree write wait time: {}".format(tree.write_wait_time))
    # print("Root write wait time: {}".format(tree.root.write_wait_time))

    # print("catgorize synchronously")
    # fut1c = [tree.categorize(i) for i in tqdm(instances[len(instances)//2:])]
    # result1c = [f.wait() for f in tqdm(fut1c)]

    # visualize(tree)

    tree2 = MultinomialCobwebTree(True, # Use mutual information (rather than expected correct guesses)
                                 1, # alpha weight
                                 True, # dynamically compute alpha
                                 True) # weight attr by avg occurance of attr

    print("training asynchronously")
    start = perf_counter()
    fut2t = [tree2.async_ifit(i) for i in tqdm(instances[:len(instances)])]
    result2t = [f.wait() for f in tqdm(fut2t)]
    end = perf_counter()
    print("Done in {}".format(end - start))
    # print("AV key write wait time: {}".format(tree2.av_key_wait_time))
    # print("Tree write wait time: {}".format(tree2.write_wait_time))
    # print("Root write wait time: {}".format(tree2.root.write_wait_time))

    # print("categorize asynchronously")
    # fut2c = [tree2.async_categorize(i) for i in tqdm(instances[len(instances)//2:])]
    # result2c = [f.wait() for f in tqdm(fut2c)]

    # visualize(tree)

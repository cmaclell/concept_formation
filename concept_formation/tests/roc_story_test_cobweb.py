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

nlp = spacy.load("en_core_web_sm", disable = ['parser'])
# nlp = spacy.load('en_core_web_trf')
nlp.add_pipe("sentencizer")
nlp.max_length = float('inf')
# nlp.add_pipe("lemmatizer")
# nlp.initialize()

# import logging  # Setting up the loggings to monitor gensim
# logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

def get_instance(text, anchor_idx, anchor_wd, window):
    ctx = text[max(0, anchor_idx-window):anchor_idx] + text[anchor_idx+1:anchor_idx+1+window]
    ctx = Counter(ctx)
    example = {}
    example['context'] = {word: ctx[word] for word in ctx}

    if anchor_wd is None:
        return example

    example['anchor'] = {anchor_wd: 1}
    return example

def get_instances(story, window):
    for anchor_idx, anchor_wd in enumerate(story):
        yield anchor_idx, get_instance(story, anchor_idx, anchor_wd, window=window)

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
            story = [# token.lemma_
                     token.text
                     for token in story if (not token.is_punct
                                                         # and not token.is_stop
                                                         )]
            yield story

if __name__ == "__main__":

    tree = MultinomialCobwebTree(True, # Use mutual information (rather than expected correct guesses)
                                 1, # alpha weight
                                 True, # dynamically compute alpha
                                 True) # weight attr by avg occurance of attr

    occurances = Counter()
    n_training_words = 0
    window = 3

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

    overall_freq = Counter([w for s in stories for w in s])

    # TODO PICK OUTFILE NAME
    outfile = 'cobweb_freq_5_rocstories_out'

    with open(outfile + ".csv", 'w') as fout:
        fout.write("n_training_words,n_training_stories,model,word,word_freq,word_obs_count,vocab_size,pred_word,prob_word,correct,story\n")

    training_queue = []
    last_checkpoint_time = time()

    for story_idx, story in enumerate(tqdm(stories)):

        # drop low frequency words
        story = [w for w in story if overall_freq[w] >= 5]

        for anchor_idx, instance in get_instances(story, window=window):
            actual_anchor = list(instance['anchor'].keys())[0]
            text = " ".join([w for w in story[max(0, anchor_idx-window):anchor_idx]])
            text += " _ "
            text += " ".join([w for w in story[max(0, anchor_idx+1):anchor_idx+window+1]])

            ## cobweb
            no_anchor = {'context': {cv: instance['context'][cv] for cv in
                                     instance['context']}}
            f = tree.categorize(no_anchor)
            probs = f.predict()
            p = 0
            best_word = "NONE"

            if 'anchor' in probs and actual_anchor in probs['anchor']:
                p = probs['anchor'][actual_anchor]

            if 'anchor' in probs and len(probs['anchor']) > 0:
                best_word = sorted([(probs['anchor'][w], random(), w) for w in probs['anchor']], reverse=True)[0][2]

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

            if len(training_queue) >= window:
                old_inst = training_queue.pop(0)
                tree.ifit(old_inst)
                old_anchor = list(old_inst['anchor'].keys())[0]
                occurances[old_anchor] += 1
                n_training_words += 1

        if (time() - last_checkpoint_time) > 3600:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open('{}-{}-{}.json'.format(outfile, story_idx, timestamp), 'w') as fout:
                fout.write(tree.dump_json())
                last_checkpoint_time = time()

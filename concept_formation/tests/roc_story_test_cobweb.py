import os
import re
import json
import string
from collections import Counter
# from random import shuffle
from random import random
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
# import multiprocessing
from time import time
from datetime import datetime

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
            story = re.sub('[' + re.escape(string.punctuation) + ']', '', story).split()
            yield story


def get_preprocessed_roc_stories(limit=None):
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
            story = nlp(story)
            story = [
                token.lemma_
                for token in story
                if (not token.is_punct and not token.is_stop)
            ]
            yield story


def write_predictions(queue, fout, overall_freq, occurances):
    while queue:
        leaf, actual_anchor, story_idx, text = queue.pop(0)
        probs = leaf.predict()
        p = 0.0
        best = "NONE"

        if 'anchor' in probs and actual_anchor in probs['anchor']:
            p = probs['anchor'][actual_anchor]

        if 'anchor' in probs and len(probs['anchor']) > 0:
            best = sorted(
                [(probs['anchor'][w], random(), w) for w in probs['anchor']],
                reverse=True
            )[0][2]

        fout.write(
            "{},{},cobweb,{},{},{},{},{},{},{},{}\n".format(
                n_training_words,
                story_idx,
                actual_anchor,
                overall_freq[actual_anchor],
                occurances[actual_anchor],
                len(occurances),
                best,
                p,
                1 if best == actual_anchor else 0,
                text
            )
        )


if __name__ == "__main__":

    tree = MultinomialCobwebTree(
        True,
        1,
        True,
        True,
    )

    occurances = Counter()
    n_training_words = 0
    window = 6

    # Buffer to avoid "cheating"
    buffer = 100

    batch_size = 200
    batch_idx = 0

    save_interval = 3600

    if not os.path.isfile("roc_stories.json"):
        print("Reading and preprocessing stories.")
        stories = list(get_raw_roc_stories())
        with open("roc_stories.json", "w") as fout:
            json.dump(stories, fout, indent=4)
        print("done.")
    else:
        print("Loading preprocessed stories.")
        with open("roc_stories.json", "r") as fin:
            stories = json.load(fin)
        print("done.")

    outfile = 'cobweb_roc_story_out'

    with open(outfile + ".csv", 'w') as fout:
        fout.write(
            "n_training_words,"
            "n_training_stories,"
            "model,"
            "word,"
            "word_freq,"
            "word_obs_count,"
            "vocab_size,"
            "pred_word,"
            "prob_word,"
            "correct,"
            "story\n"
        )
        fout.close()

    training_queue = []
    categorization_queue = []
    last_checkpoint = time()

    overall_freq = Counter([w for s in stories for w in s])

    for story_idx, story in enumerate(tqdm(stories)):

        for anchor_idx, instance in get_instances(story, window=window):
            batch_idx += 1
            text = story[max(0, anchor_idx - window):min(len(story), anchor_idx + window)]
            text[anchor_idx - max(0, anchor_idx - window)] = '_'
            text = ' '.join(text)

            actual_anchor = list(instance['anchor'].keys())[0]

            instance_no_anchor = {'context': instance['context']}

            categorization_queue.append((
                tree.async_categorize(instance),
                actual_anchor,
                story_idx,
                text,
            ))
            training_queue.append(instance)

            if batch_idx >= batch_size:
                batch_idx = 0

                with open(outfile + ".csv", 'a') as fout:
                    write_predictions(categorization_queue, fout, overall_freq, occurances)

                training_futures = []

                while len(training_queue) > buffer:
                    old_inst = training_queue.pop(0)
                    training_futures.append(tree.async_ifit(old_inst))
                    old_anchor = list(old_inst['anchor'].keys())[0]
                    occurances[old_anchor] += 1
                    n_training_words += 1

                [i.wait() for i in training_futures]

        if (time() - last_checkpoint) > save_interval:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open('{}-{}-{}.json'.format(outfile, story_idx, timestamp), 'w') as fout:
                fout.write(tree.dump_json())
            last_checkpoint = time()

    with open(outfile + ".csv", 'a') as fout:
        write_predictions(categorization_queue, fout, overall_freq, occurances)

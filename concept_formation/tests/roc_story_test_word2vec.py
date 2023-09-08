import os
import re
import json
import string
from collections import Counter
# from random import shuffle
from tqdm import tqdm
# from concept_formation.multinomial_cobweb import MultinomialCobwebTree
# from concept_formation.visualize import visualize
import spacy
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from matplotlib import pyplot as plt
from gensim.models import Word2Vec
# from gensim.models.word2vec import LineSentence
import multiprocessing


# nlp = spacy.load("en_core_web_sm", disable=['parser'])
# # nlp = spacy.load('en_core_web_trf')
# nlp.add_pipe("sentencizer")
# nlp.max_length = float('inf')
# # nlp.add_pipe("lemmatizer")
# # nlp.initialize()

en = spacy.load('en_core_web_sm')
stopwords = list(en.Defaults.stop_words)

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


def get_raw_holmes_stories(limit=None):
    non_letters = re.compile(r"[^a-z]")
    punctuation = re.compile(r'[\'\",\?\!.;:`]*')
    for i in range(12):
        with open("Holmes_Training_Data/" + str(i) + ".txt") as f:
            text = punctuation.sub('', f.read().lower())
            story = non_letters.sub(' ', text).split()

            yield story


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


def get_go_roc_stories(limit=None):
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
            story = [i for i in story if i not in stopwords]
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
            story = [token.lemma_ for token in story if (not token.is_punct and not token.is_stop)]
            yield story


if __name__ == "__main__":
    row_num = 0

    occurances = Counter()
    n_training_words = 0
    window = 3

    cores = multiprocessing.cpu_count()
    training_data = []
    training_freq = 100

    # if not os.path.isfile("roc_stories.json"):
    #     print("Reading and preprocessing stories.")
    #     stories = list(get_go_roc_stories())
    #     with open("roc_stories.json", "w") as fout:
    #         json.dump(stories, fout, indent=4)
    #     print("done.")
    # else:
    #     print("Loading preprocessed stories.")
    #     with open("roc_stories.json", "r") as fin:
    #         stories = json.load(fin)
    #     print("done.")

    stories = list(get_go_roc_stories(5000))

    outfile = 'word2vec_out_small'

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

    overall_freq = Counter([w for s in stories for w in s])

    for i, training_story in enumerate(tqdm(stories)):

        training_data.append(training_story)
        n_training_words += len(training_story)
        occurances.update(training_story)

        if not (len(training_data) % training_freq):
            model = Word2Vec(
                min_count=1,
                window=window,
                workers=cores - 1,
                vector_size=100,
                alpha=0.1,
                min_alpha=0.0007,
            )
            model.build_vocab(training_data)
            model.train(training_data, total_examples=len(training_data), epochs=30)
            vocab_size = len(model.wv)

            for story_idx, story in enumerate(stories[i + 1:i + 1 + training_freq]):
                for anchor_idx, instance in get_instances(story, window=window):
                    actual_anchor = list(instance['anchor'].keys())[0]

                    if actual_anchor in stopwords:
                        continue

                    text = story[max(0, anchor_idx - window):min(len(story), anchor_idx + window)]
                    text[anchor_idx - max(0, anchor_idx - window)] = '_'
                    text = ' '.join(text)

                    # shhhh
                    context_word_list = [w for w in instance['context']]
                    # context_word_list = [word for word, repeats in instance['context'].items() for _ in range(1)]

                    preds = model.predict_output_word(context_word_list, topn=vocab_size)
                    best = "NONE"

                    p = 0.0

                    if preds:
                        preds.sort(reverse=True, key=lambda x: x[1])
                        best = preds[0][0]
                        preds = {word: prob for word, prob in preds}
                        if actual_anchor in preds:
                            p = preds[actual_anchor]

                    with open(outfile + ".csv", 'a') as fout:
                        fout.write(
                            "{},{},word2vec,{},{},{},{},{},{},{},{}\n".format(
                                n_training_words,
                                story_idx + i + 1,
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
                        row_num += 1
                        if row_num > 50000:
                            1 / 0

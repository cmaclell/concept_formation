import re
import time
import spacy
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from functools import reduce
from collections import Counter
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from random import random, shuffle, randint
from concept_formation.visualize import visualize
from concept_formation.multinomial_cobweb import MultinomialCobwebTree


en = spacy.load('en_core_web_sm')
stopwords = list(en.Defaults.stop_words)


def get_instance(text, anchor_idx, anchor_wd, window, story_idx=None):
    ctx = wind_slice(text, anchor_idx, window)
    ctx = Counter(ctx)
    example = {}
    example['context'] = {word: ctx[word] for word in ctx}

    if anchor_wd is None:
        return example

    example['anchor'] = {anchor_wd: 1}
    example['_kind'] = {'stopword' if anchor_wd in stopwords else 'word': 1}
    example['_story'] = {str(story_idx): 1}
    return example


def instance_iter(it, window):
    for story_idx, story in enumerate(it):
        for anchor_idx, instance in get_instances(story, window, story_idx):
            yield (story_idx, story, anchor_idx, instance)


def window_iter(it, window):
    for story_idx, story in enumerate(it):
        for w in get_windows(story, window):
            yield w


def get_instances(story, window, story_idx=None):
    for anchor_idx, anchor_wd in enumerate(story):
        yield anchor_idx, get_instance(story, anchor_idx, anchor_wd, window, story_idx)


def get_windows(story, window):
    for anchor_idx, anchor_wd in enumerate(story):
        yield (
            story[max(0, anchor_idx - window):anchor_idx],
            story[anchor_idx],
            story[anchor_idx + 1:anchor_idx + 1 + window]
        )


def get_go_roc_stories(limit=None):
    with open("ROCStories_winter2017.txt", 'r') as fin:

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


def dict_sum(d1, d2):
    out = {}
    for i in d1:
        out[i] = d1[i] + d2.get(i, 0)
    for j in d2:
        if j not in d1.keys():
            out[j] = d2[j]
    return out


def wind_slice(array, index, window):
    return array[max(0, index - window):index] + array[index + 1:index + 1 + window]


def train_path(word_tree, layer_2_tree, instances, window, noise=False):
    # print('training word tree')
    def make_noise():
        if noise:
            return Counter([str(randint(1, 1000)) for i in range(30)])
        else:
            return {}
    training_futures = [word_tree.ifit({
        "anchor": {i[1]: 1},
        "context": Counter(i[0] + i[2]),
        "noise": make_noise(),
        "_base": {i[1][0]: 1},
        "_variant": {i[1][1]: 1},
    }) for i in instances]
    # visualize(word_tree)

    # print('training layer 2 tree')
    for i, _ in (enumerate(list(training_futures))):
        anchor = instances[i][1]
        context = Counter(
            reduce(
                lambda a, b: (a.extend(b), a)[1],
                map(
                    lambda x: x.get_ancestry(),
                    wind_slice(training_futures, i, window)
                )
            )
        )
        instance = {
            'anchor': {anchor: 1},
            'context': context,
            "_base": {anchor[0]: 1},
            "_variant": {anchor[1]: 1},
        }

        layer_2_tree.ifit(instance)


def test_path(word_tree, layer_2_tree, instances, window):
    out = []
    # print('testing tree...')
    for i in (list(instances)):
        no_anchor = i[0] + [None] + i[2]
        cat_futures = []
        for j in range(len(no_anchor)):
            x = {
                'context':
                Counter(filter(None, no_anchor[max(0, j - window):j + window + 1]))
            }
            if no_anchor[j]:
                x.update({'anchor': {no_anchor[j]: 1}})
                n = word_tree.categorize(x)
                cat_futures.extend(n.get_weighted_ancestry(x))

        instance = {'context': Counter(
            cat_futures
        )}
        probs = layer_2_tree\
            .categorize(instance)\
            .predict_basic()

        p = 0.0
        best = 'NONE'

        if 'anchor' in probs and i[1] in probs['anchor']:
            p = probs['anchor'][i[1]]

        if 'anchor' in probs and len(probs['anchor']) > 0:
            best = sorted(
                [(probs['anchor'][w], random(), w) for w in probs['anchor']],
                reverse=True
            )[0][2]

        word_probs = word_tree.categorize(
            {'context': Counter(filter(None, no_anchor))}
        ).predict()

        word_p = 0.0
        word_best = 'NONE'

        if 'anchor' in word_probs and i[1] in word_probs['anchor']:
            word_p = word_probs['anchor'][i[1]]

        if 'anchor' in word_probs and len(word_probs['anchor']) > 0:
            word_best = sorted(
                [(word_probs['anchor'][w], random(), w) for w in word_probs['anchor']],
                reverse=True
            )[0][2]

        out.append((
            i[1], best, p,
            int(best[:-1] == i[1][:-1]),
            word_best, word_p,
            int(word_best[:-1] == i[1][:-1]),
            ' '.join(map(lambda x: x if x else '_', no_anchor))
        ))

    return out


def train(word_tree, layer_2_tree, instances, window):
    # print('training word tree')
    training_futures = [word_tree.ifit(i[3]) for i in instances]
    # visualize(word_tree)
    # 1 / 0

    # print('training layer 2 tree')
    for i, anchor in (enumerate(training_futures)):
        anchor = instances[i][1][instances[i][2]]
        context = Counter(map(lambda x: x.get_basic_id(), wind_slice(training_futures, i, window)))
        instance = {'anchor': {anchor: 1}, 'context': context}

        layer_2_tree.ifit(instance)


def test(word_tree, layer_2_tree, instances, window):
    out = []
    # print('testing tree...')
    for i in (list(instances)):
        no_anchor = i[0] + [None] + i[2]
        cat_futures = []
        for j in range(len(no_anchor)):
            n = word_tree.categorize({
                'context':
                Counter(filter(None, no_anchor[max(0, j - window):j + window + 1]))
            })
            cat_futures.append(n.get_basic_id())
        probs = layer_2_tree.categorize(
            {'context': Counter(cat_futures[:len(i[0])] + cat_futures[-len(i[2]):])}
        ).predict()

        p = 0.0
        best = 'NONE'

        if 'anchor' in probs and i[1] in probs['anchor']:
            p = probs['anchor'][i[1]]

        if 'anchor' in probs and len(probs['anchor']) > 0:
            best = sorted(
                [(probs['anchor'][w], random(), w) for w in probs['anchor']],
                reverse=True
            )[0][2]

        out.append((
            i[1], best, p,
            int(best == i[1]),
            ' '.join(map(lambda x: x if x else '_', no_anchor))
        ))

    return out


def train_word2vec(model, stories, train_num):
    training_text = []
    i = 0
    while sum((len(i) for i in training_text)) < train_num:
        training_text.append(stories[i])
        i += 1
    # print("training word2vec...")
    model.build_vocab(training_text)
    model.train(training_text, total_examples=len(training_text), epochs=30)


def test_word2vec(model, instances, window):
    out = []
    vocab_size = len(model.wv)
    # print('testing word2vec...')
    for i in (list(instances)):
        no_anchor = i[0] + [None] + i[2]

        context_word_list = i[0] + i[2]

        probs = model.predict_output_word(context_word_list, topn=vocab_size)

        p = 0.0
        best = 'NONE'

        if probs:
            probs.sort(reverse=True, key=lambda x: x[1])
            best = probs[0][0]
            probs = {word: prob for word, prob in probs}
            if i[1] in probs:
                p = probs[i[1]]

        out.append((
            i[1], best, p,
            int(best == i[1]),
            ' '.join(map(lambda x: x if x else '_', no_anchor))
        ))

    return out


if __name__ == "__main__":

    train_per_iter = 100
    iters = 100
    buffer = 0
    test_num = 300
    window = 3
    graph_window = 100

    import synthetic_data

    test_start = train_per_iter * iters + buffer
    # stories = list(get_go_roc_stories(10_000))

    stories = synthetic_data.get_data(train_per_iter * iters + buffer + test_num)

    outfile = 'cobweb_out_layers_path'

    with open(outfile + ".csv", 'w') as fout:
        fout.write(
            "correct_word,"
            "cobweb_pred_word,"
            "cobweb_prob_correct,"
            "cobweb_correct,"
            "cobweb_word_pred_word,"
            "cobweb_word_prob_correct,"
            "cobweb_word_correct,"
            "word2vec_pred_word,"
            "word2vec_prob_correct,"
            "word2vec_correct,"
            "story\n"
        )
        fout.close()

    cores = multiprocessing.cpu_count()
    instances = list(window_iter(stories, window))
    testing_instances = instances[test_start:test_start + test_num]

    for i in tqdm(range(iters)):
        train_num = train_per_iter * (i + 1)

        word_tree = MultinomialCobwebTree(
            True,
            1,
            True,
            True,
        )

        layer_2_tree = MultinomialCobwebTree(
            True,
            1,
            True,
            True,
        )

        w2v = Word2Vec(
            min_count=1,
            window=window,
            workers=cores - 1,
            vector_size=100,
            alpha=0.1,
            min_alpha=0.0007,
        )

        training_instances = instances[:train_num]

        train_path(word_tree, layer_2_tree, training_instances, window, noise=True)
        cobweb_data = test_path(word_tree, layer_2_tree, testing_instances, window)

        # train_groups(word_tree, layer_2_tree, training_instances, window)
        # cobweb_data = test_groups(word_tree, layer_2_tree, testing_instances, window)

        if i == 1:
            visualize(word_tree)
            time.sleep(5)
            visualize(layer_2_tree)

        train_word2vec(w2v, stories, train_num)
        word2vec_data = test_word2vec(w2v, testing_instances, window)

        with open(outfile + ".csv", 'a') as fout:
            for idx in range(test_num):
                fout.write(
                    "{},{},{},{},{},{},{},{},{},{},{}\n"
                    .format(
                        *cobweb_data[idx][:-1],
                        *word2vec_data[idx][1:],
                    )
                )

    data = pd.read_csv(outfile + ".csv")
    # data['cobweb_percent_correct'] = np.array(data.cobweb_correct).reshape(train_per_iter, -1).mean(axis=0)
    # data['cobweb_word_percent_correct'] = np.array(data.cobweb_word_correct).reshape(train_per_iter, -1).mean(axis=0)
    # data['word2vec_percent_correct'] = data.word2vec_correct.rolling(window=graph_window).mean()

    plt.rcParams["figure.figsize"] = [7.00, 7.50]

    plt.subplot(1, 1, 1)
    plt.plot(range(iters), np.array(data.cobweb_correct).reshape(test_num, -1).mean(axis=0), color='blue', label='cobweb layered')
    plt.plot(range(iters), np.array(data.cobweb_word_correct).reshape(test_num, -1).mean(axis=0), color='red', label='cobweb word')
    # plt.plot(data.index, data.word2vec_percent_correct, color='black', label='word2vec')
    plt.legend()
    plt.show()

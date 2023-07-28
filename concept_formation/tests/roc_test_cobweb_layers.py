import re
import spacy
import string
from tqdm import tqdm
from random import random
# from functools import reduce
from collections import Counter
from concept_formation.visualize import visualize
from concept_formation.multinomial_cobweb import MultinomialCobwebTree


en = spacy.load('en_core_web_sm')
stopwords = list(en.Defaults.stop_words)


def get_instance(text, anchor_idx, anchor_wd, window):
    ctx = wind_slice(text, anchor_idx, window)
    ctx = Counter(ctx)
    example = {}
    example['context'] = {word: ctx[word] for word in ctx}

    if anchor_wd is None:
        return example

    example['anchor'] = {anchor_wd: 1}
    example['_kind'] = {'stopword' if anchor_wd in stopwords else 'word': 1}
    return example


def instance_iter(it, window):
    for story_idx, story in enumerate(it):
        for anchor_idx, instance in get_instances(story, window):
            yield (story_idx, story, anchor_idx, instance)


def window_iter(it, window):
    for story_idx, story in enumerate(it):
        for w in get_windows(story, window):
            yield w


def get_instances(story, window):
    for anchor_idx, anchor_wd in enumerate(story):
        yield anchor_idx, get_instance(story, anchor_idx, anchor_wd, window=window)


def get_windows(story, window):
    for anchor_idx, anchor_wd in enumerate(story):
        yield (story[max(0, anchor_idx - window):anchor_idx], story[anchor_idx], story[anchor_idx + 1:anchor_idx + 1 + window])


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


def train(word_tree, layer_2_tree, instances, window):
    print('training word tree')
    training_futures = [word_tree.ifit(i[3]) for i in tqdm(instances)]
    # visualize(word_tree)

    print('training layer 2 tree')
    for i, anchor in tqdm(enumerate(training_futures)):
        anchor = instances[i][1][instances[i][2]]
        context = Counter(map(lambda x: x.get_basic_id(), wind_slice(training_futures, i, window)))
        instance = {'anchor': {anchor: 1}, 'context': context}

        layer_2_tree.ifit(instance)


def test(word_tree, layer_2_tree, instances, window, outfile):
    print('testing tree...')
    for i in tqdm(list(instances)):
        no_anchor = i[0] + [None] + i[2]
        cat_futures = []
        for j in range(len(no_anchor)):
            n = word_tree.categorize({
                'context':
                Counter(filter(None, no_anchor[max(0, j - window):j + window + 1]))
            })
            cat_futures.append(n.get_basic_id())
        probs = layer_2_tree.categorize({'context': Counter(cat_futures[:len(i[0])] + cat_futures[-len(i[2]):])}).predict()

        p = 0.0
        best = 'NONE'

        if 'anchor' in probs and i[1] in probs['anchor']:
            p = probs['anchor'][i[1]]

        if 'anchor' in probs and len(probs['anchor']) > 0:
            best = sorted(
                [(probs['anchor'][w], random(), w) for w in probs['anchor']],
                reverse=True
            )[0][2]

        with open(outfile + ".csv", 'a') as fout:
            fout.write("{},{},{},{},{}\n".format(
                i[1], best, p, int(best == i[1]), ' '.join(map(lambda x: x if x else '_', no_anchor))
            ))
            fout.close()


if __name__ == "__main__":

    train_num = 5000
    buffer = 500
    test_num = 5000

    outfile = 'cobweb_out_layers'

    with open(outfile + ".csv", 'w') as fout:
        fout.write(
            "correct_word,"
            "pred_word,"
            "prob_correct,"
            "correct,"
            "story\n"
        )
        fout.close()

    word_tree = MultinomialCobwebTree(
        True,
        0,
        True,
        True,
    )

    layer_2_tree = MultinomialCobwebTree(
        True,
        1,
        True,
        True,
    )

    window = 6

    stories = list(get_go_roc_stories(5000))

    training_instances = list(instance_iter(stories, window))[:train_num]
    testing_instances = list(window_iter(stories, window))[train_num + buffer:train_num + test_num + buffer]

    train(word_tree, layer_2_tree, training_instances, window)
    # visualize(layer_2_tree)
    test(word_tree, layer_2_tree, testing_instances, window, outfile)

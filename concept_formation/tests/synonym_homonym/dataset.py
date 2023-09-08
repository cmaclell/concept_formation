import re
import spacy
import string
import random
import time
from tqdm import tqdm
from functools import reduce
from collections import Counter
from concept_formation.visualize import visualize
from concept_formation.multinomial_cobweb import MultinomialCobwebTree

en = spacy.load('en_core_web_sm')
stopwords = list(en.Defaults.stop_words)


def get_go_roc_stories(limit=None, start=0):
    with open("ROCStories_winter2017.txt", 'r') as fin:

        lines = list(fin)
        if limit is None:
            limit = len(lines) - 1

        for line in tqdm(lines[start + 1:limit + start + 1]):

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


def get_synonym_dataset(stories, top_n, copies):
    stories_cloned = list(stories)
    frequencies = reduce(
        lambda a, b: (a.update(b), a)[1],
        map(
            lambda x: Counter(x),
            stories_cloned,
        ),
    )

    most_frequent = list(map(
        lambda a: a[0],
        sorted(
            list(frequencies.items()),
            key=lambda a: -a[1]
        ),
    ))[:top_n]

    for story in stories_cloned:
        for i in range(copies):
            copy = map(lambda x: (x + str(i) if x in most_frequent else x), story)
            yield list(copy)


def make_instances(stories, window):
    for story in stories:
        out = []
        for anchor_idx, anchor_wd in enumerate(story):
            no_anchor = (
                story[max(0, anchor_idx - window):anchor_idx]
                + [None]
                + story[anchor_idx + 1:anchor_idx + window + 1]
            )
            out.append((
                anchor_wd,
                {
                    "anchor": {anchor_wd: 1},
                    "context": Counter(filter(None, no_anchor)),
                    "_variant": ({
                        anchor_wd[-1]: 1
                    } if anchor_wd[-1] in "1234567890" else {}),
                    "_base": ({
                        anchor_wd[:-1]: 1
                    } if anchor_wd[-1] in "1234567890" else {}),
                },
                no_anchor
            ))
        yield out


def train(word_tree, layer_2_tree, story, window):
    training_futures = [((anchor, word_tree.ifit(i), i), visualize(word_tree), time.sleep(0.3))[0] for anchor, i, _ in story]
    training_futures = list(map(lambda a: (a[0], set(a[1].get_ancestry()), a[2]), training_futures))
    # print(training_futures)
    for (idx, (anchor, _, inst)) in enumerate(training_futures):
        instance = {
            "anchor": {anchor: 1},
            "context": Counter(reduce(
                lambda x, y: (x.update(y), x)[1],
                [
                    ancestry
                    for (_, ancestry, _)
                    in (training_futures[
                        max(0, idx - window):idx
                    ] + training_futures[
                        idx + 1:idx + window + 1
                    ])
                ],
            )),
            "_base": inst["_base"],
        }
        # print(instance["context"])
        layer_2_tree.ifit(instance)


def test(word_tree, layer_2_tree, story, window):
    out = []
    for (anchor_wd, i, no_anchor) in list(story):
        print(no_anchor)
        cat_futures = []
        for j in range(len(no_anchor)):
            x = {
                'context':
                Counter(filter(
                    None,
                    no_anchor[max(0, j - window):j] + no_anchor[j + 1:j + window + 1]
                ))
            }
            if no_anchor[j]:
                x.update({'anchor': {no_anchor[j]: 1}})
                n = word_tree.categorize(x)
                print("thing", x, "categorized to", n.get_ancestry())
                cat_futures.extend(n.get_ancestry())

        instance = {'context': Counter(
            cat_futures[:no_anchor.index(None)] + cat_futures[no_anchor.index(None):]
        )}
        probs = layer_2_tree\
            .categorize(instance)\
            .predict_best(instance)

        p = 0.0
        best = 'NONE'

        if 'anchor' in probs and anchor_wd in probs['anchor']:
            p = probs['anchor'][anchor_wd]

        if 'anchor' in probs and len(probs['anchor']) > 0:
            best = sorted(
                [(probs['anchor'][w], random.random(), w) for w in probs['anchor']],
                reverse=True
            )[0][2]

        word_probs = word_tree.categorize(
            {'context': Counter(filter(None, no_anchor))}
        ).predict_best({'context': Counter(filter(None, no_anchor))})

        word_p = 0.0
        word_best = 'NONE'

        if 'anchor' in word_probs and anchor_wd in word_probs['anchor']:
            word_p = word_probs['anchor'][anchor_wd]

        if 'anchor' in word_probs and len(word_probs['anchor']) > 0:
            word_best = sorted(
                [(word_probs['anchor'][w], random.random(), w) for w in word_probs['anchor']],
                reverse=True
            )[0][2]

        out.append((
            anchor_wd, best, p,
            int(best[:-1] == anchor_wd[:-1]),
            word_best, word_p,
            int(word_best[:-1] == anchor_wd[:-1]),
            ' '.join(map(lambda x: x if x else '_', no_anchor))
        ))

    return out


if __name__ == "__main__":
    word_tree = MultinomialCobwebTree(
        True,
        1,
        True,
        True,
    )

    import synthetic_data
    for _ in range(5):
        for sentence in synthetic_data.get_data(5):
            for i in range(3):
                word_tree.ifit({
                    "anchor": {sentence[i]: 1},
                    "context": {
                        sentence[i - 1]: 1,
                        sentence[i - 2]: 1
                    },
                    "_base": {sentence[i][:-1]: 1},
                    "_type": {
                        str("acebdf".index(sentence[i][:-1]) % 3): 1
                    }
                })

        visualize(word_tree)

    # random.seed(0)
    # window = 3

    # outfile = "CFG_test"

    # # stories = get_go_roc_stories(20, start=20)
    # # synonyms = list(get_synonym_dataset(stories, 5, window))
    # # random.shuffle(synonyms)
    # # instances = make_instances(synonyms, window)

    # import synthetic_data

    # stories = synthetic_data.get_data(8)
    # instances = make_instances(stories, window)

    # word_tree = MultinomialCobwebTree(
    #     True,
    #     1,
    #     True,
    #     True,
    # )

    # layer_2_tree = MultinomialCobwebTree(
    #     True,
    #     1,
    #     True,
    #     True,
    # )

    # for story in tqdm(instances):
    #     train(word_tree, layer_2_tree, story, 3)
    #     # visualize(word_tree)
    #     # time.sleep(0.5)
    #     # visualize(layer_2_tree)
    #     # time.sleep(0.5)

    # tests = 1

    # testing = synthetic_data.get_data(tests)
    # testing_instances = make_instances(testing, window)
    # output = []
    # for test_story in tqdm(testing_instances):
    #     output.extend(test(word_tree, layer_2_tree, test_story, 3))

    # layer_accuracy = sum(map(lambda x: x[3], output))
    # word_accuracy = sum(map(lambda x: x[6], output))
    # print("layers accuracy:", layer_accuracy / len(output))
    # print("word accuracy:", word_accuracy / len(output))

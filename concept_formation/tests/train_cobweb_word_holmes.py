import re
from collections import Counter
from matplotlib import pyplot as plt
import math
from random import shuffle
from random import random
from time import time
from datetime import datetime

from tqdm import tqdm
from multiprocessing import Pool

from preprocess_holmes import load_holmes_data
from concept_formation.multinomial_cobweb import MultinomialCobwebTree
from concept_formation.visualize import visualize

def get_instance(text, anchor_idx, anchor_wd, window):
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

def process_story(story, window):
    return list(get_instances(story, window))

if __name__ == "__main__":

    tree = MultinomialCobwebTree(alpha=0.000001, weight_attr=False,
            objective=0, children_norm=True,
            norm_attributes=False)
    window = 10
    n_training_words = 0
    occurances = Counter()

    stories = load_holmes_data()
    print(len(stories))
    overall_freq = Counter([w for s in stories for w in s])
    print(overall_freq.most_common(100))

    stories = [[w for w in s if overall_freq[w] >= 200] for s in stories]

    # TODO PICK OUTFILE NAME
    outfile = 'cobweb_10_holmes_out'

    with open(outfile + ".csv", 'w') as fout:
        fout.write("n_training_words,n_training_stories,model,word,word_freq,word_obs_count,vocab_size,pred_word,prob_word,correct,story\n")

    instances = []

    with Pool() as pool:
        processed_stories = pool.starmap(process_story, [(story, window) for story in stories])
        for story_idx, story_instances in enumerate(processed_stories):
            for anchor_idx, instance in story_instances:
                instances.append((instance, story_idx, anchor_idx))

    # for story_idx, story in enumerate(tqdm(stories)):
    #     for anchor_idx, instance in get_instances(story, window=window):
    #         instances.append((instance, story_idx, anchor_idx))

    shuffle(instances)
    training_queue = []
    last_checkpoint_time = time()

    for idx, (instance, story_idx, anchor_idx) in enumerate(tqdm(instances)):
        # if idx > 1 and idx % 10000 == 0:
        #     visualize(tree)

        actual_anchor = list(instance['anchor'].keys())[0]
        story = stories[story_idx]
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

            # if len(training_queue) > window:
        if len(training_queue) > 0:
            old_inst = training_queue.pop(0)
            tree.ifit(old_inst)
            old_anchor = list(old_inst['anchor'].keys())[0]
            occurances[old_anchor] += 1
            n_training_words += 1

        if (time() - last_checkpoint_time) > 3600:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open('{}-{}-{}.json'.format(outfile, idx, timestamp), 'w') as fout:
                fout.write(tree.dump_json())
                last_checkpoint_time = time()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open('{}-{}-{}-final.json'.format(outfile, len(instances), timestamp), 'w') as fout:
        fout.write(tree.dump_json())
        last_checkpoint_time = time()



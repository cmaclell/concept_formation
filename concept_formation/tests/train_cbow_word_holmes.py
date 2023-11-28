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
from cbow_pytorch import CbowModel

def get_instance(text, anchor_idx, anchor_wd, window):
    before_text = text[max(0, anchor_idx-window):anchor_idx]
    after_text = text[anchor_idx+1:anchor_idx+1+window]
    ctx_text = before_text + after_text
    # ctx = {}
    # before_ctx = {}
    # after_ctx = {}
    # for i, w in enumerate(before_text):
    #     before_ctx[w] = 1/abs(len(before_text) - i)
    #     ctx[w] = 1/abs(len(before_text) - i)
    # for i, w in enumerate(after_text):
    #     after_ctx[w] = 1/(i+1)
    #     ctx[w] = 1/(i+1)
    example = {}
    example['context'] = ctx_text
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

    window = 10
    n_training_words = 0
    occurances = Counter()

    stories = load_holmes_data()
    print(len(stories))
    overall_freq = Counter([w for s in stories for w in s])
    vocab_size = len([w for w in overall_freq if overall_freq[w] >= 200])
    print("VOCAB SIZE", vocab_size)
    cbow_model = CbowModel(vocab_size, 100, window=window)
    print(overall_freq.most_common(100))

    stories = [[w for w in s if overall_freq[w] >= 200] for s in stories]

    # TODO PICK OUTFILE NAME
    outfile = 'cbow_10_holmes_out'

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

    for idx, (instance, story_idx, anchor_idx) in enumerate(tqdm(instances)):

        actual_anchor = list(instance['anchor'].keys())[0]
        story = stories[story_idx]
        text = " ".join([w for w in story[max(0, anchor_idx-window):anchor_idx]])
        text += " _ "
        text += " ".join([w for w in story[max(0, anchor_idx+1):anchor_idx+window+1]])

        context = instance['context']
        probs = cbow_model.predict(context)
    
        p = 0
        best_word = "NONE"

        if actual_anchor in probs:
            p = probs[actual_anchor]

        if len(probs) > 0:
            best_word = sorted([(probs[w], random(), w) for w in probs], reverse=True)[0][2]

        training_queue.append((context, actual_anchor))

        with open(outfile + ".csv", 'a') as fout:
            fout.write("{},{},cbow,{},{},{},{},{},{},{},{}\n".format(n_training_words,
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
            old_ctx, old_anchor = training_queue.pop(0)
            cbow_model.train(old_ctx, old_anchor)
            occurances[old_anchor] += 1
            n_training_words += 1

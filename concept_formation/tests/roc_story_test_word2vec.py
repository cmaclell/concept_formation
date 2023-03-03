import os
import re
import json
from collections import Counter
from random import shuffle
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
            story = [token.lemma_ for token in story if (not token.is_punct and
                                                         not token.is_stop)]
            yield story

if __name__ == "__main__":

    cores = multiprocessing.cpu_count()
    occurances = Counter()
    training_data = []
    word2vec_obs = []
    word2vec_accuracy = []
    n_training_words = 0

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

    train_freq = 100

    with open('word2vec_rocstories_out.csv', 'w') as fout:
        fout.write("n_training_words,n_training_stories,model,word,word_freq,word_obs_count,vocab_size,pred_word,prob_word,correct,story\n")

    for i, story in enumerate(tqdm(stories)):

        training_data.append(story)
        n_training_words += len(story)
        for w in story:
            occurances[w] += 1

        if len(training_data) % train_freq < train_freq-1:
            continue

        print("building model")

        window = 3
        model = Word2Vec(min_count=5,
                         window=window,
                         workers=cores-1,
                         vector_size=100,
                         # sample=6e-5, 
                         alpha=0.1, 
                         min_alpha=0.0007, 
                         # negative=20,
                         # workers=cores-1)
                         )
        model.build_vocab(training_data)
        model.train(training_data, total_examples=len(training_data), epochs=30)
        vocab_size = len(model.wv)
        print("vocab_size = ", vocab_size)
        print("done")

        print("testing model")
        for test_story in tqdm(stories[i+1:i+1+train_freq]):
            for anchor_idx, instance in get_instances(test_story, window=window):
                actual_anchor = list(instance['anchor'].keys())[0]
                text = " ".join([w for w in test_story[max(0, anchor_idx-window):anchor_idx]])
                text += " _ "
                text += " ".join([w for w in test_story[max(0, anchor_idx+1):anchor_idx+window]])

                ## word2vec
                context_word_list = [w for w in instance['context']]
                preds = model.predict_output_word(context_word_list, topn=vocab_size)
                best_word = "NONE"

                p = 0.0

                if preds:
                    preds.sort(reverse=True, key=lambda x: x[1])
                    best_word = preds[0][0]
                    preds = {word: prob for word, prob in preds}
                    if actual_anchor in preds:
                        p = preds[actual_anchor]
                
                with open('word2vec_rocstories_out.csv', 'a') as fout:
                    fout.write("{},{},word2vec,{},{},{},{},{},{},{},{}\n".format(n_training_words,
                                                                     len(training_data),
                                                                     actual_anchor,
                                                                     overall_freq[actual_anchor],
                                                                     occurances[actual_anchor],
                                                                     vocab_size,
                                                                     best_word,
                                                                     p,
                                                                     1 if best_word == actual_anchor else 0,
                                                                     text))
        print("done") 

        # print()
        # print("Progress after {} sentences".format(len(train_stories))
        # print("Average word2vec accuracy: ", sum(word2vec_accuracy) / len(word2vec_accuracy))
        # print("Average cobweb accuracy: ", sum(cobweb_accuracy) / len(cobweb_accuracy))

        # word2vec_x = [i for i in range(len(word2vec_accuracy))]

        # data = pd.DataFrame({'block': prediction_block + prediction_block,
        #                      'model': ["Cobweb" for i in range(len(prediction_block))] + ["Word2Vec" for i in range(len(prediction_block))],
        #                      'accuracy': cobweb_accuracy + word2vec_accuracy,
        #                      'obs': cobweb_obs + word2vec_obs, 
        #                      'n': cobweb_x + word2vec_x})

        # plt.clf()
        # sns.lineplot(data=data, x="block", y="accuracy", hue="model")

        # plt.savefig("accuracy_by_n.png")



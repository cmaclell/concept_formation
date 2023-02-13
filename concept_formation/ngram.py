from collections import defaultdict
from collections import Counter
from math import log

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
import numpy as np

class NGramBayes:

    def __init__(self):
        self.X = []
        self.y = []
        self.dv = None
        self.model = None
        self.vocab = None

    def ifit(self, instance):
        self.model = None
        self.vocab = None
        self.dv = None
        self.X.append({w: 1 for w in instance if w != "##anchor##"})
        self.y.append(instance['##anchor##'])

    def fit_model(self):
        print('fitting with {} examples'.format(len(self.X)))
        print('transforming')
        self.dv = DictVectorizer()
        X = self.dv.fit_transform(self.X)
        self.vocab = {w: i for i, w in enumerate(list(set(self.y)))}
        y = np.array([self.vocab[w] for w in self.y])
        print('done')

        print('fitting model with {} classes'.format(len(self.vocab)))
        self.model = MultinomialNB()
        self.model.fit(X, y)
        print('done')

    def predict(self, instance, choices):
        if self.model is None:
            self.fit_model()
        x = {w: 1 for w in instance if w != "##anchor##"}
        x = self.dv.transform([x])
        yh = self.model.predict_log_proba(x)[0]
        return {w: yh[self.vocab[w]] if w in self.vocab else float('-inf') for w in choices}


class NGram:

    def __init__(self):
        self.words = defaultdict(Counter)
        self.counts = Counter()
        self.vocab = set()
        self.total_count = 0
        self.alpha = 0.0001

    def ifit(self, instance):
        
        anchor = instance["##anchor##"]
        self.vocab.add(anchor)

        self.counts[anchor] += 1
        self.total_count += 1

        for ctx_word in instance:
            if ctx_word == "##anchor##":
                continue

            self.vocab.add(anchor)
            self.words[anchor][ctx_word] += 1

    def predict(self, instance, choices):

        predictions = defaultdict(lambda:float("-inf"))

        for anchor in choices:
            log_p = log(self.counts[anchor] + self.alpha / (self.total_count + self.alpha * len(self.counts))) # prob of anchor

            for context in self.words[anchor]:
                log_p += log((self.words[anchor][context] + self.alpha)/
                             (self.counts[anchor] + self.alpha * len(self.vocab)))
            log_p += (len(self.vocab) - len(self.words[anchor])) * self.alpha / (self.alpha * len(self.vocab))

            predictions[anchor] = log_p

        return predictions






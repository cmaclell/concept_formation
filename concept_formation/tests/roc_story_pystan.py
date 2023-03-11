import os
import json

from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm
import stan

from roc_story_test_cobweb import get_instances


if __name__ == "__main__":

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

    window = 3

    word_key = {}
    anchor_n = []
    anchor_word = []
    anchor_inst = []

    # contexts = []
    # context_counts = []

    n = 1
    instance_id = 1
    for story_idx, story in enumerate(tqdm(stories[:1])):
        for anchor_idx, instance in get_instances(story, window=window):
            anchor = list(instance['anchor'])[0]
            if anchor not in word_key:
                word_key[anchor] = len(word_key)

            anchor_n.append(n)
            anchor_inst.append(instance_id)
            anchor_word.append(word_key[anchor])

            n += 1
            instance_id += 1

    stan_code = """
    data {
      // training data
      int<lower=1> K;               // num topics
      int<lower=1> V;               // num words
      int<lower=0> M;               // num docs
      int<lower=0> N;               // total word instances
      int<lower=1,upper=V> w[N];    // word n
      int<lower=1,upper=M> instance[N];  // instance ID for word n
      // hyperparameters
      vector<lower=0>[K] alpha;     // topic prior
      vector<lower=0>[V] beta;      // word prior
    }
    parameters {
      simplex[K] theta;             // topic prevalence
      simplex[V] phi[K];            // word dist for topic k
    }
    model {
      real gamma[M, K];
      theta ~ dirichlet(alpha);
      for (k in 1:K)
        phi[k] ~ dirichlet(beta);
      for (m in 1:M)
        for (k in 1:K)
          gamma[m, k] = categorical_lpmf(k \mid theta);
      for (n in 1:N)
        for (k in 1:K)
          gamma[instance[n], k] = gamma[instance[n], k]
                             + categorical_lpmf(w[n] \mid phi[k]);
      for (m in 1:M)
        target += log_sum_exp(gamma[m]);
    }
    """

    k = 3
    data = {"K": k,
            "V": len(word_key),
            "M": instance_id-1,
            "N": n-1,
            "w": anchor_word,
            "instance": anchor_inst,
            "alpha": np.ones((k,)),
            "beta": np.ones((len(word_key),))
            }

    posterior = stan.build(stan_code, data=data)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    # eta = fit["theta"]  # array with shape (8, 4000)
    df = fit.to_frame()  # pandas `DataFrame, requires pandas
    print(df)

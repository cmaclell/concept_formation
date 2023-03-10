import os
import json

from matplotlib import pyplot as plt

import arviz as az
import numpy as np
from tqdm import tqdm
import pymc as pm
from sklearn.feature_extraction import DictVectorizer

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

    anchor_key = {}
    anchors = []
    contexts = []
    context_counts = []

    for story_idx, story in enumerate(tqdm(stories[:100])):
        for anchor_idx, instance in get_instances(story, window=window):
            anchor = list(instance['anchor'])[0]
            if anchor not in anchor_key:
                anchor_key[anchor] = len(anchor_key)
            anchors.append(anchor_key[anchor])
            contexts.append(instance['context'])
            context_counts.append(sum([instance['context'][w] for w in instance['context']]))

    dv_context = DictVectorizer(sparse=False)
    context_words = dv_context.fit_transform(contexts)
    context_counts = np.array(context_counts)
    anchors = np.array(anchors)

    print(anchors)
    print(context_words)
    print(context_counts)

    bayes_cat = pm.Model()

    K = 3
    D = anchors.shape[0]
    V = len(anchor_key)

    alpha = np.ones((1, K))
    beta = np.ones((1, V))
    with pm.Model() as model:
        thetas = pm.Dirichlet("thetas", a=alpha, shape=(D, K))
        phis = pm.Dirichlet("phis", a=beta, shape=(K, V))
        z = pm.Categorical("zx", p=thetas, shape=(D,))
        w = pm.Categorical("wx", 
                           p=phis[z], 
                           observed=anchors)
        trace = pm.sample(200)

    # with pm.Model() as m:
    #     w = pm.Dirichlet('w', a=np.ones(n, k))
    #     anchor_p = pm.Dirchlet('anchor_p', a=np.ones(k, V))
    #     d = pm.Categorical('d', p=anchor_p)


    #     d1 = pm.Categorical.dist(p=a)
    #     d2 = pm.Categorical.dist(p=b)
    #     mix  = pm.Mixture('mix', w=w, comp_dists=[d1, d2], observed=[0, 1, 9])
    #     trace = pm.sample(200)

    # with pm.Model() as bayes_cat:

    #     # priors
    #     p = pm.Dirichlet('p', a=np.ones(V))
    #     
    #     # likelihood of observation
    #     anchor_obs = pm.Categorical("anchor_obs", p, observed=anchors)
    #     # context_obs = pm.Multinomial(n=n_context_words, p, observed=context_counts)

    #     trace = pm.sample(200, target_accept=0.9)

    az.plot_trace(trace)
    plt.show()


    # az.plot_forest(trace, var_names=["w"])
    # plt.show()
    

    # y = np.random.normal(1, 3, 1000)
    # with pm.Model() as pooled:
    #     # Latent pooled effect size
    #     mu = pm.Normal("mu", 0, sigma=100)
    #     sigma = pm.HalfNormal("sigma", sigma=100)

    #     obs = pm.Normal("obs", mu, sigma, observed=y)

    #     trace = pm.sample(3000)

    # az.plot_trace(trace)
    # plt.show()


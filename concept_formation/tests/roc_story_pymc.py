import numpy as np
import pymc as pm
import arviz as az
from matplotlib import pyplot as plt
import os
import json
from tqdm import tqdm
# import theano.tensor as tt

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

    window = 1

    word_key = {}
    anchor_word = []
    anchor_inst = []

    context_word = []
    context_inst = []

    instance_id = 0
    for story_idx, story in enumerate(tqdm(stories[:1])):
        for anchor_idx, instance in get_instances(story, window=window):
            anchor = list(instance['anchor'])[0]
            if anchor not in word_key:
                word_key[anchor] = len(word_key)

            anchor_inst.append(instance_id)
            anchor_word.append(word_key[anchor])

            for context_w in instance['context']:
                if context_w not in word_key:
                    word_key[context_w] = len(word_key)

                context_inst.append(instance_id)
                context_word.append(word_key[context_w])

            instance_id += 1


    k = 3
    data = {"K": k,
            "V": len(word_key),
            "M": instance_id,
            "AN": len(anchor_word),
            "CN": len(context_word),
            "w": np.array(anchor_word),
            "c": np.array(context_word),
            "anchor_instance": np.array(anchor_inst),
            "context_instance": np.array(context_inst),
            "alpha": np.ones((k,)) * 0.01,
            "beta": np.ones((len(word_key),)) * 0.01
            }

    print(data)

    # with pm.Model() as model:
    #     # Global topic distribution
    #     theta = pm.Dirichlet("theta", a=data['alpha'])
    #     
    #     # Word distributions for K topics
    #     anchor_phi = pm.Dirichlet("anchor_phi", a=data['beta'], shape=(data['K'], data['V']))
    #     context_phi = pm.Dirichlet("context_phi", a=data['beta'], shape=(data['K'], data['V']))
    #     
    #     # Topic of documents
    #     z = pm.Categorical("z", p=theta, shape=data['M'])
    #     
    #     # Words in documents
    #     anchor_p = anchor_phi[z][data['anchor_instance']]
    #     context_p = context_phi[z][data['context_instance']]
    #     w = pm.Categorical("w", p=anchor_p, observed=data['w'])
    #     c = pm.Categorical("c", p=context_p, observed=data['c'])
    #     trace = pm.sample(1000, tune=1000)

    # # az.plot_trace(trace, var_names=['phi', 'theta', 'z'])
    # # plt.show()

    # az.plot_forest(trace, var_names=["z"], combined=True, hdi_prob=0.95, r_hat=True);
    # plt.show()

    # print(pm.summary(trace))

    with pm.Model() as model_marg:
        # Word distributions for K topics
        anchor_phi = pm.Dirichlet("anchor_phi", a=data['beta'], shape=(data['K'], data['V']))
        context_phi = pm.Dirichlet("context_phi", a=data['beta'], shape=(data['K'], data['V']))
        
        # Topic of documents
        z = pm.Dirichlet("z", a=data['alpha'], shape=(data['M'], data['K']))
        
        # Global topic distribution
        theta = pm.Deterministic("theta", z.mean(axis=0))
        
        # Words in documents
        anchor_comp_dists = pm.Categorical.dist(anchor_phi)
        context_comp_dists = pm.Categorical.dist(context_phi)
        w = pm.Mixture("w",
                       w=z[data['anchor_instance'], :],
                       comp_dists=anchor_comp_dists,
                       observed=data['w'])
        c = pm.Mixture("c",
                       w=z[data['context_instance'], :],
                       comp_dists=context_comp_dists,
                       observed=data['c'])

        trace_marg = pm.sample(1000, tune=1000)

    # az.plot_trace(trace_marg, var_names=['anchor_phi', 'theta', 'z']);
    # plt.show()
    # az.plot_forest(trace_marg, var_names=["theta"], combined=True, hdi_prob=0.95, r_hat=True);
    # plt.show()
    print(pm.summary(trace_marg))


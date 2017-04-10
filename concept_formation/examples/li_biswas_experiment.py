import numpy as np
import matplotlib.pyplot as plt
from random import choice
from random import shuffle
from random import random
from random import seed

from concept_formation.cobweb3 import Cobweb3Tree
from concept_formation.cluster import cluster

seed(0)

def run_clust_exp(nominal_noise=0, numeric_noise=0, scaling=False):
    data = []

    for i in range(60):
        x = {}
        x['_label'] = "G1"

        if random() >= nominal_noise:
            x['f1'] = "G1f1"
        else:
            x['f1'] = choice(['G2f1', 'G3f1'])

        if random() >= nominal_noise:
            x['f2'] = choice(["G1f2a", "G1f2b"])
        else:
            x['f2'] = choice(["G2f2a", "G2f2b","G3f2a", "G3f2b"])

        if random() >= numeric_noise:
            x['f3'] = np.random.normal(4,1,1)[0]
        else:
            x['f3'] = choice([np.random.normal(10,1,1)[0],np.random.normal(16,1,1)[0]])

        if random() >= numeric_noise:
            x['f4'] = np.random.normal(20,2,1)[0]
        else:
            x['f4'] = choice([np.random.normal(32,2,1)[0],np.random.normal(44,2,1)[0]])

        data.append(x)

    for i in range(60):
        x = {}
        x['_label'] = "G2"

        if random() >= nominal_noise:
            x['f1'] = "G2f1"
        else:
            x['f1'] = choice(["G2f1", "G3f1"])

        if random() >= nominal_noise:
            x['f2'] = choice(["G2f2a", "G2f2b"])
        else:
            x['f2'] = choice(["G1f2a", "G1f2b", "G3f2a", "G3f2b"])

        if random() >= numeric_noise:
            x['f3'] = np.random.normal(10,1,1)[0]
        else:
            x['f3'] = choice([np.random.normal(4,1,1)[0],np.random.normal(16,1,1)[0]])

        if random() >= numeric_noise:
            x['f4'] = np.random.normal(32,2,1)[0]
        else:
            x['f4'] = choice([np.random.normal(20,2,1)[0],np.random.normal(44,2,1)[0]])

        data.append(x)

    for i in range(60):
        x = {}
        x['_label'] = "G3"

        if random() >= nominal_noise:
            x['f1'] = "G3f1"
        else:
            x['f1'] = choice(["G1f1", "G2f1"])

        if random() >= nominal_noise:
            x['f2'] = choice(["G3f2a", "G3f2b"])
        else:
            x['f2'] = choice(["G1f2a", "G1f2b", "G2f2a", "G2f2b"])

        if random() >= numeric_noise:
            x['f3'] = np.random.normal(16,1,1)[0]
        else:
            x['f3'] = choice([np.random.normal(4,1,1)[0],np.random.normal(10,1,1)[0]])

        if random() >= numeric_noise:
            x['f4'] = np.random.normal(44,2,1)[0]
        else:
            x['f4'] = choice([np.random.normal(20,2,1)[0],np.random.normal(32,2,1)[0]])

        data.append(x)

    shuffle(data)
    t = Cobweb3Tree(scaling=scaling)
    clustering = cluster(t, data)
    return data, clustering[0]

def run_noise_exp(scaling=False):
    noise = np.arange(0.0, 0.8, 0.2)
    print(noise)

    miss_nominal = []
    miss_numeric = []
    #aris = []

    for n in noise:
        data, clustering = run_clust_exp(n,0,scaling)
        confusion = {}
        for i,c in enumerate(clustering):
            if c not in confusion:
                confusion[c] = {}
            if data[i]['_label'] not in confusion[c]:
                confusion[c][data[i]['_label']] = 0
            confusion[c][data[i]['_label']] += 1
        print(confusion)

        totals = sorted([(sum([confusion[c][g] for g in
                               confusion[c]]), c) for c in confusion], reverse=True)
        top = [c for t,c in totals[:3]]
        miss = 0
        
        for c in confusion:
            v = sorted([confusion[c][g] for g in confusion[c]], reverse=True)
            if c not in top:
                miss += v[0]

            for minorities in v[1:]:
                miss += 2 * minorities

        #labels = [d['_label'] for d in data]
        #aris.append(ari(labels, clustering))
        miss_nominal.append(miss)

    for n in noise:
        data, clustering = run_clust_exp(0,n, scaling)
        confusion = {}
        for i,c in enumerate(clustering):
            if c not in confusion:
                confusion[c] = {}
            if data[i]['_label'] not in confusion[c]:
                confusion[c][data[i]['_label']] = 0
            confusion[c][data[i]['_label']] += 1
        print(confusion)

        totals = sorted([(sum([confusion[c][g] for g in
                               confusion[c]]), c) for c in confusion], reverse=True)
        top = [c for t,c in totals[:3]]
        miss = 0
        
        for c in confusion:
            v = sorted([confusion[c][g] for g in confusion[c]], reverse=True)
            if c not in top:
                miss += v[0]

            for minorities in v[1:]:
                miss += 2 * minorities

        # labels = [d['_label'] for d in data]
        # aris.append(ari(labels, clustering))
        miss_numeric.append(miss)

    return noise, miss_nominal, miss_numeric

nominal = []
numeric = []

for i in range(2):
    noise, miss_nominal, miss_numeric = run_noise_exp(scaling=0.5)
    nominal.append(miss_nominal)
    numeric.append(miss_numeric)
    noise = noise

nominal = np.array(nominal)
numeric = np.array(numeric)

nominal = np.mean(nominal, axis=0)
numeric = np.mean(numeric, axis=0)

nominal_line, = plt.plot(noise, miss_nominal, linestyle="--", marker="o", color="b")
numeric_line, = plt.plot(noise, miss_numeric, linestyle="-", marker="o", color="g")
plt.legend([nominal_line, numeric_line], ["Noisy Nominal", "Noisy Numeric"],
           loc=2)

plt.xlabel("Percentage Noise")
plt.ylabel("Misclassification Count")


plt.ylim(0,200)
plt.show()

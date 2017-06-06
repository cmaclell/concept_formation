from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from random import shuffle
from random import seed

from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np

from concept_formation.trestle import TrestleTree
from concept_formation.cluster import cluster_split_search
from concept_formation.cluster import AIC, BIC, AICc, CU
from concept_formation.datasets import load_rb_wb_03
from concept_formation.datasets import load_rb_com_11
from concept_formation.datasets import load_rb_s_13
from concept_formation.preprocessor import ObjectVariablizer

seed(5)

hueristics = [AIC, BIC, CU, AICc]


def calculate_aris(dataset):
    shuffle(dataset)
    dataset = dataset[:60]

    variablizer = ObjectVariablizer()
    dataset = [variablizer.transform(t) for t in dataset]

    tree = TrestleTree()
    tree.fit(dataset)

    clusters = [cluster_split_search(tree, dataset, h, minsplit=1, maxsplit=40,
                                     mod=False) for h in hueristics]
    human_labels = [dataset['_human_cluster_label'] for dataset in dataset]

    return [max(adjusted_rand_score(human_labels, huer), 0.01) for huer in
            clusters]


x = np.arange(len(hueristics))
width = 0.3

hueristic_names = ['AIC', 'BIC', 'CU', 'AICc']
# for i in range(len(clusters)):
#     hueristic_names[i] +=  '\nClusters='+str(len(set(clusters[i])))

b1 = plt.bar(x-width, calculate_aris(load_rb_wb_03()), width, color='r',
             alpha=.8, align='center')
b2 = plt.bar(x, calculate_aris(load_rb_com_11()), width, color='b', alpha=.8,
             align='center')
b3 = plt.bar(x+width, calculate_aris(load_rb_s_13()), width, color='g',
             alpha=.8, align='center')
plt.legend((b1[0], b2[0], b3[0]), ('wb_03', 'com_11', 's_13'))
plt.title("TRESTLE Clustering Accuracy of Best Clustering by Different"
          "Hueristics")
plt.ylabel("Adjusted Rand Index (Agreement Correcting for Chance)")
plt.ylim(0, 1)
plt.xlabel("Hueristic")
plt.xticks(x, hueristic_names)
plt.show()

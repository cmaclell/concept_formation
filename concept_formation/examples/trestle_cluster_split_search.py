from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from random import shuffle
<<<<<<< HEAD
from random import seed
=======
>>>>>>> f1bf25831a608cae96f608080b6af48a310e80d7

from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np

from concept_formation.trestle import TrestleTree
from concept_formation.cluster import cluster_split_search
from concept_formation.cluster import AIC, BIC, AICc, CU
from concept_formation.datasets import load_rb_wb_03
from concept_formation.preprocessor import ObjectVariablizer

<<<<<<< HEAD
seed(5)

=======
>>>>>>> f1bf25831a608cae96f608080b6af48a310e80d7
towers = load_rb_wb_03()
shuffle(towers)
towers = towers[:60]

variablizer = ObjectVariablizer()
towers = [variablizer.transform(t) for t in towers]

tree = TrestleTree()
tree.fit(towers)

hueristics = [AIC,BIC,CU,AICc]

<<<<<<< HEAD
clusters = [cluster_split_search(tree,towers,h,minsplit=1,maxsplit=40,mod=False) for h in hueristics]
human_labels = [tower['_human_cluster_label'] for tower in towers]

x = np.arange(len(hueristics))
y = [max(adjusted_rand_score(human_labels, huer),0.01) for huer in clusters]
width = 0.45

hueristic_names = ['AIC','BIC','CU','AICc']
for i in range(len(clusters)):
    hueristic_names[i] +=  '\nClusters='+str(len(set(clusters[i])))

plt.bar(x,y,width,color='r',alpha=.8,align='center')
plt.title("TRESTLE Clustering Accuracy of Best Clustering by Different Hueristics")
plt.ylabel("Adjusted Rand Index (Agreement Correcting for Chance)")
plt.ylim(0,1)
plt.xlabel("Hueristic")
plt.xticks(x,hueristic_names)
=======
shuffle(towers)
clusters = [cluster_split_search(tree,towers,h,minsplit=1,maxsplit=20,mod=True) for h in hueristics]
human_labels = [tower['_human_cluster_label'] for tower in towers]

x = np.arange(len(hueristics))
y = [adjusted_rand_score(human_labels, huer) for huer in clusters]
width = 0.35

hueristic_names = ['AIC','BIC','CU','AICc']
for i in range(len(clusters)):
    hueristic_names[i] +=  ' Clusters='+str(len(set(clusters[i])))

plt.bar(x,y,width,color='r')
plt.title("TRESTLE Clustering Accuracy of Best Clustering by Different Hueristics")
plt.ylabel("Adjusted Rand Index (Agreement Correcting for Chance)")
plt.xlabel("Hueristic")
plt.xticks(x+width,hueristic_names)
plt.legend(loc=4)
>>>>>>> f1bf25831a608cae96f608080b6af48a310e80d7
plt.show()
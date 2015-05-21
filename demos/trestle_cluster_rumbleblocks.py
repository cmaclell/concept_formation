import json
from random import shuffle

from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

from trestle import TrestleTree
from cluster import cluster

############### LOAD THE DATA ################

## Choose a datafile to load
filename = 'data_files/rb_wb_03_continuous.json'
#filename = 'data_files/rb_com_11_continuous.json'
#filename = 'data_files/rb_s_13_continuous.json'

with open(filename) as dat:
    towers = json.load(dat)
shuffle(towers)
towers = towers[:30]

############## CLUSTER THE DATA ##############

tree = TrestleTree()
clusters = cluster(tree, towers, maxsplit=10)

############# PLOT THE RESULTS ###############

human_labels = [tower['_human_cluster_label'] for tower in towers]

x = [num_splits for num_splits in range(1,len(clusters)+1)]
y = [adjusted_rand_score(human_labels, split) for split in clusters]
plt.plot(x, y, label="TRESTLE")

plt.title("TRESTLE Clustering Accuracy (Given Human Ground Truth)")
plt.ylabel("Adjusted Rand Index (Agreement Correcting for Chance)")
plt.xlabel("# of Splits of Trestle Tree")
plt.legend(loc=4)
plt.show()

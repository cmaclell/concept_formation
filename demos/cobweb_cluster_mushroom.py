import json
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import adjusted_rand_score

from cobweb import CobwebTree
from cluster import cluster

############## LOAD DATA ###############################

with open('data_files/mushrooms.json') as fin:
    mushrooms = json.load(fin)
shuffle(mushrooms)
mushrooms = mushrooms[:150]

############### CLUSTER DATA ###########################

tree = CobwebTree()
mushrooms_no_class = [{a: mushroom[a] for a in mushroom 
                       if a != 'classification'} for mushroom in mushrooms]
clusters = cluster(tree, mushrooms_no_class)[0] 

############### COMPUTE AGREEMENT WITH TRUE CLASSES#####

mushroom_class = [mushroom[a] for mushroom in mushrooms for a in mushroom
                  if a == 'classification']
ari = adjusted_rand_score(clusters, mushroom_class)

############### PLOT RESULTS ###########################

dv = DictVectorizer(sparse=False)
mushroom_X = dv.fit_transform(mushrooms_no_class)

pca = PCA(n_components=2)
mushroom_2d_x = pca.fit_transform(mushroom_X)

colors = ['b', 'g', 'r', 'y', 'k', 'c', 'm']
shapes = ['o', '*']
clust_set = {v:i for i,v in enumerate(list(set(clusters)))}
class_set = {v:i for i,v in enumerate(list(set(mushroom_class)))}

for class_idx, class_label in enumerate(class_set):
    x = [x[0] for i,x in enumerate(mushroom_2d_x) if mushroom_class[i] == class_label]
    y = [x[1] for i,x in enumerate(mushroom_2d_x) if mushroom_class[i] == class_label]
    c = [colors[clust_set[clusters[i]]] for i,v in enumerate(mushroom_2d_x) if
         mushroom_class[i] == class_label]
    plt.scatter(x, y, color=c, marker=r"$ {} $".format(class_label[0]), label=class_label)

plt.title("COBWEB Mushroom Clustering (ARI w/ Hidden Edibility Labels = %0.2f)" % (ari))
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.legend(loc=4)
plt.show()


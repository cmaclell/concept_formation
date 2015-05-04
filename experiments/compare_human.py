import json
from trestle import TrestleTree
from random import shuffle
import pickle
from sklearn.metrics.cluster import adjusted_rand_score

with open('human.pickle', 'rb') as fin:
    human_dict = pickle.load(fin)

with open('data_files/rb_com_11_noCheck.json', "r") as json_data:
    instances = json.load(json_data)


for i in instances:
    #i['_guid'] = i['guid']
    #del i['guid']
    del i['success']

    del_list = []
    for attr in i:
        if isinstance(i[attr], dict):
            i[attr]['l'] = float(i[attr]['l'])
            i[attr]['r'] = float(i[attr]['r'])
            i[attr]['b'] = float(i[attr]['b'])
            i[attr]['t'] = float(i[attr]['t'])
            pass
        elif isinstance(i[attr], list):
            del_list.append(attr)

    for attr in del_list:
        del i[attr]

#shuffle(instances)
#instances = instances[0:15]

best_ordering = []
best_clustering = []
best_cu = -1
print("r1,s1\tr1,s2\tr1,s3\tr2,s1\tr2,s2\tr2,s3\n", end="")
      #r3,s1\tr3,s2\t,r3,s3\n")
for i in range(10):
    tree =  TrestleTree()

    shuffle(instances)
    #tree.cluster(instances)
    human_label = [human_dict[x['_guid']] for x in instances]
    a1 = [adjusted_rand_score(machine_label, human_label) for machine_label in
         tree.cluster(instances, maxsplit=3)]
    print("%0.4f\t%0.4f\t%0.4f\t" % (a1[0], a1[1], a1[2]), end="")
    raise Exception("DONE")

    shuffle(instances)
    #tree.cluster(instances)
    human_label = [human_dict[x['_guid']] for x in instances]

    a1 = [adjusted_rand_score(machine_label, human_label) for machine_label in
         tree.cluster(instances, maxsplit=3)]
    print("%0.4f\t%0.4f\t%0.4f" % (a1[0], a1[1], a1[2])),


    #shuffle(instances)
    #human_label = [human_dict[x['_guid']] for x in instances]

    #a1 = [adjusted_rand_score(machine_label, human_label) for machine_label in
    #     tree.cluster(instances, maxsplit=3)]
    #print("%0.4f\t%0.4f\t%0.4f\n" % (a1[0], a1[1], a1[2]))

    #if cu > best_cu:
    #    print("BEST:", cu)
    #    best_cu = cu
    #    best_ordering = [i for i in instances]

#tree = TrestleTree()
#
#human_label = [human_dict[x['_guid']] for x in best_ordering]
#
#for machine_label in tree.cluster(best_ordering, maxsplit=5):
#    print(adjusted_rand_score(machine_label, human_label))

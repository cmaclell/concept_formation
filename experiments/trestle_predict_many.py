import json
from trestle import TrestleTree
from random import shuffle

with open('data_files/rb_s_07.json') as fin:
    instances = json.load(fin)

for instance in instances:
    instance['_guid'] = instance['guid']
    del instance['guid']

    del_list = []
    for attr in instance:
        if isinstance(instance[attr], dict):
            instance[attr]['l'] = float(instance[attr]['l'])
            instance[attr]['r'] = float(instance[attr]['r'])
            instance[attr]['b'] = float(instance[attr]['b'])
            instance[attr]['t'] = float(instance[attr]['t'])
        elif isinstance(instance[attr], list):
            del_list.append(attr)

    for attr in del_list:
        del instance[attr]

with open('accuracy.csv', 'w') as fout:
    fout.write('accuracy,opp\n')

    # the number of runs
    for i in range(1000):
        print(i)
        shuffle(instances)
        insts = instances[0:31]

        tree = TrestleTree()
        opp = 0
        for j in insts:
            opp += 1
            fout.write("%0.5f,%i\n" % (tree.specific_prediction(j, 'success',
                                                guessing=False), opp))
            tree.ifit(j)

#tree.predictions('data_files/rb_s_07.json', 31, 100, attr='success')


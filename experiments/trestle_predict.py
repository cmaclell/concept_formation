import json
from trestle import TrestleTree

with open('data_files/rb_s_07.json') as fin:
    instances = json.load(fin)
    instances = {i['guid']: i for i in instances}

for guid in instances:
    instances[guid]['_guid'] = instances[guid]['guid']
    del instances[guid]['guid']

    del_list = []
    for attr in instances[guid]:
        if isinstance(instances[guid][attr], dict):
            instances[guid][attr]['l'] = float(instances[guid][attr]['l'])
            instances[guid][attr]['r'] = float(instances[guid][attr]['r'])
            instances[guid][attr]['b'] = float(instances[guid][attr]['b'])
            instances[guid][attr]['t'] = float(instances[guid][attr]['t'])
        elif isinstance(instances[guid][attr], list):
            del_list.append(attr)

    for attr in del_list:
        del instances[guid][attr]

data = {}
with open('data_files/human_s_07_success_prediction.csv') as fin:
    count = 0
    for row in fin:
        count += 1
        if count == 1:
            continue
        
        row = row.split(',')
        if row[0] not in data:
            data[row[0]] = []
        data[row[0]].append((row[-3], row[1]))

with open('accuracy.csv', 'w') as fout:
    fout.write('accuracy,opp')
    for user in data:
        data[user].sort()
        tree = TrestleTree()
        opp = 0
        for time, inst in data[user]:
            opp += 1
            print(time, inst)
            inst = inst[7:-4]
            inst = instances[inst]
            #fout.write("%0.5f,%i\n" % (tree.specific_prediction(inst, 'success',
            #                                    guessing=False), opp))
            tree.ifit(inst)

#tree.predictions('data_files/rb_s_07.json', 31, 100, attr='success')


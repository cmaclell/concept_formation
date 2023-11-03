from pprint import pprint
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from concept_formation.multinomial_cobweb import MultinomialCobwebTree
from concept_formation.visualize import visualize
  
# fetch dataset 
data = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = data.data.features 
y = data.data.targets 

# metadata 
# meta = json.loads(heart_disease.metadata)
print(type(data.metadata))
meta = dict(data.metadata)
print(list(meta))
print(X)
  
# variable information 
print(data.variables) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

inst_train = X_train.to_dict(orient='records')
labels_train = y_train.to_dict(orient='records')

inst_test = X_test.to_dict(orient='records')
labels_test = y_test.to_dict(orient='records')

print("Example Data")
pprint(inst_train[:2])
pprint(labels_train[:2])

# Build Cobweb model
tree = MultinomialCobwebTree(0.150, # alpha weight
                             True, # weight attr by avg occurance of attr
                             2, # 0 = MI, 1 = Theil's U, 2 = NMI
                             True, # normalize by # children
                             False) # norm by attribute

for i, instance in enumerate(tqdm(inst_train)):
    instance = {a: {instance[a]: 1.0} for a in instance}
    label = labels_train[i]
    for attr in label:
        instance[attr] = {label[attr]: 5.0}
    tree.ifit(instance)


leaf_acc = []
basic_acc = []
best_acc = []
for i, instance in enumerate(tqdm(inst_test)):
    instance = {a: {instance[a]: 1.0} for a in instance}
    label = labels_test[i]
    leaf = tree.categorize(instance)

    for attr in label:
        leaf_p = leaf.predict_probs()[attr]
        leaf_v = sorted([(leaf_p[val], val) for val in leaf_p])[-1][1]
        basic_p = leaf.get_basic_level().predict_probs()[attr]
        basic_v = sorted([(basic_p[val], val) for val in basic_p])[-1][1]
        best_p = leaf.get_best_level(instance).predict_probs()[attr]
        best_v = sorted([(best_p[val], val) for val in best_p])[-1][1]

        leaf_acc.append(int(leaf_v == label[attr]))
        basic_acc.append(int(basic_v == label[attr]))
        best_acc.append(int(best_v == label[attr]))

print("leaf acc: ", sum(leaf_acc)/len(leaf_acc))
print("basic acc: ", sum(basic_acc)/len(basic_acc))
print("best acc: ", sum(best_acc)/len(best_acc))

visualize(tree)






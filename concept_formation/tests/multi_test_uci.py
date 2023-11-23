from pprint import pprint

from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OrdinalEncoder

from concept_formation.multinomial_cobweb import MultinomialCobwebTree
from concept_formation.visualize import visualize

import viz_path

# Monks
# data = pd.read_csv('../data/monks/monks-1.train',
#                    header=None, 
#                    sep=" ",
#                    index_col=False) 
# print(data)
# data.columns = ["{}".format(i) for i in range(len(data.columns))]
# X_train = data.drop("0", axis=1).drop("1", axis=1).drop("8", axis=1)
# y_train = data[["1"]]
# y_train['1'] = y_train['1'].astype(str)
# 
# data = pd.read_csv('../data/monks/monks-1.test',
#                    header=None, 
#                    sep=" ",
#                    index_col=False) 
# print(data)
# data.columns = ["{}".format(i) for i in range(len(data.columns))]
# X_test = data.drop("0", axis=1).drop("1", axis=1).drop("8", axis=1)
# y_test = data[["1"]]
# y_test['1'] = y_test['1'].astype(str)

# NURSERY 
# data = pd.read_csv('../data/nursery/nursery.data',
#                    header=None, 
#                    index_col=False) 
# print(data)
# data.columns = ["{}".format(i) for i in range(len(data.columns))]
# X = data.drop("8", axis=1)
# y = data[["8"]]
  
# SOYBEAN
# data = pd.read_csv('../data/soybean_large/soybean-large.data',
#                    header=None, 
#                    index_col=False) 
# print(data)
# data.columns = ["{}".format(i) for i in range(len(data.columns))]
# X = data.drop("0", axis=1).replace("?", float('nan'))
# y = data[["0"]]

# TICTACTOE
# data = pd.read_csv('../data/tic_tac_toe/tic-tac-toe.data',
#                    header=None, 
#                    index_col=False) 
# print(data)
# data.columns = ["{}".format(i) for i in range(len(data.columns))]
# X = data.drop("9", axis=1)
# y = data[["9"]]

# fetch dataset 
# car = 19
# mushroom = 73
# breast cancer = 14
# congressional voting = 105

data = fetch_ucirepo(id=19) 
X = data.data.features 
y = data.data.targets 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #, random_state=40)

inst_train = X_train.to_dict(orient='records')
labels_train = y_train.to_dict(orient='records')

inst_test = X_test.to_dict(orient='records')
labels_test = y_test.to_dict(orient='records')

print("Example Data")
print(inst_train[:1])
print(labels_train[:1])

# Build Cobweb model
alpha = 0.000001
tree = MultinomialCobwebTree(alpha, # alpha weight
                             False, # weight attr by avg occurance of attr
                             0, # 0 = MI, 1 = Theil's U, 2 = NMI
                             True, # normalize by # children
                             False) # norm by attribute

for i, instance in enumerate(tqdm(inst_train)):
    instance = {a: {str(instance[a]): 1.0} for a in instance if not(isinstance(instance[a], float))}
    label = labels_train[i]
    for attr in label:
        instance[attr] = {label[attr]: 1.0}
    # pprint(instance)
    tree.ifit(instance)


leaf_acc = []
obj1_acc = []
obj1_greedy_acc = []
obj2_acc = []
obj2_greedy_acc = []
obj3_acc = []
obj3_greedy_acc = []

for i, instance in enumerate(tqdm(inst_test)):
    instance = {a: {str(instance[a]): 1.0} for a in instance if not(isinstance(instance[a], float))}
    label = labels_test[i]
    leaf = tree.categorize(instance)

    for attr in label:
        leaf_p = leaf.predict_probs()[attr]
        leaf_v = sorted([(leaf_p[val], val) for val in leaf_p])[-1][1]
        # print('leaf')
        # pprint(leaf_p)

        samples = 100
        obj1_p = tree.predict_probs_mixture(instance, samples, False, False, 1)[attr]
        obj1_v = sorted([(obj1_p[val], val) for val in obj1_p])[-1][1]

        obj1_greedy_p = tree.predict_probs_mixture(instance, samples, True, False, 1)[attr]
        obj1_greedy_v = sorted([(obj1_greedy_p[val], val) for val in obj1_greedy_p])[-1][1]

        obj2_p = tree.predict_probs_mixture(instance, samples, False, False, 2)[attr]
        obj2_v = sorted([(obj2_p[val], val) for val in obj2_p])[-1][1]

        obj2_greedy_p = tree.predict_probs_mixture(instance, samples, True, False, 2)[attr]
        obj2_greedy_v = sorted([(obj2_greedy_p[val], val) for val in obj2_greedy_p])[-1][1]

        obj3_p = tree.predict_probs_mixture(instance, samples, False, False, 3)[attr]
        obj3_v = sorted([(obj3_p[val], val) for val in obj3_p])[-1][1]

        obj3_greedy_p = tree.predict_probs_mixture(instance, samples, True, False, 3)[attr]
        obj3_greedy_v = sorted([(obj3_greedy_p[val], val) for val in obj3_greedy_p])[-1][1]

        leaf_acc.append(int(leaf_v == label[attr]))
        obj1_acc.append(int(obj1_v == label[attr]))
        obj1_greedy_acc.append(int(obj1_greedy_v == label[attr]))
        obj2_acc.append(int(obj2_v == label[attr]))
        obj2_greedy_acc.append(int(obj2_greedy_v == label[attr]))
        obj3_acc.append(int(obj3_v == label[attr]))
        obj3_greedy_acc.append(int(obj3_greedy_v == label[attr]))

print("leaf acc: ", sum(leaf_acc)/len(leaf_acc))
print("obj1 acc: ", sum(obj1_acc)/len(obj1_acc))
print("obj2 acc: ", sum(obj2_acc)/len(obj2_acc))
print("obj3 acc: ", sum(obj3_acc)/len(obj3_acc))
print("obj1_greedy acc: ", sum(obj1_greedy_acc)/len(obj1_greedy_acc))
print("obj2_greedy acc: ", sum(obj2_greedy_acc)/len(obj2_greedy_acc))
print("obj3_greedy acc: ", sum(obj3_greedy_acc)/len(obj3_greedy_acc))

visualize(tree)

try:
    clf = CategoricalNB(alpha=alpha)
    # dv = DictVectorizer()
    dv = OrdinalEncoder()

    train_encoded = pd.DataFrame(
        dv.fit_transform(X_train),
        columns=X_train.columns
    )
    test_encoded = pd.DataFrame(
        dv.transform(X_test),
        columns=X_test.columns
    )

    clf.fit(train_encoded, y_train)
    # print(y_train)
    y_h = clf.predict(test_encoded)
    # print(y_h)
    # print(y_test.to_numpy().flatten())
    print('NAIVE BAYES:', (y_h == y_test.to_numpy().flatten()).mean())
except:
    print("NAIVE BAYES: Failed")
    pass

try:
    clf = DecisionTreeClassifier()
    # dv = DictVectorizer()
    dv = OrdinalEncoder()

    train_encoded = pd.DataFrame(
        dv.fit_transform(X_train),
        columns=X_train.columns
    )
    test_encoded = pd.DataFrame(
        dv.transform(X_test),
        columns=X_test.columns
    )

    clf.fit(train_encoded, y_train)
    # print(y_train)
    y_h = clf.predict(test_encoded)
    # print(y_h)
    # print(y_test.to_numpy().flatten())
    print('DECISION TREE:', (y_h == y_test.to_numpy().flatten()).mean())
except:
    print("DECISION TREE: Failed")
    pass









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
alpha = 0.01
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
mix_acc = []
mix_leaves_acc = []
full_acc = []
basic_acc = []
best_acc = []
for i, instance in enumerate(tqdm(inst_test)):
    instance = {a: {str(instance[a]): 1.0} for a in instance if not(isinstance(instance[a], float))}
    label = labels_test[i]
    leaf = tree.categorize(instance)

    for attr in label:
        leaf_p = leaf.predict_probs()[attr]
        leaf_v = sorted([(leaf_p[val], val) for val in leaf_p])[-1][1]
        # print('leaf')
        # pprint(leaf_p)

        wleaf_p = leaf.predict_weighted_probs(instance)[attr]
        # print('mixture path')
        # pprint(wleaf_p)
        wleaf_v = sorted([(wleaf_p[val], val) for val in wleaf_p])[-1][1]

        wcleaf_p = leaf.predict_weighted_leaves_probs(instance)[attr]
        # print('mixture leaves')
        # pprint(wcleaf_p)
        wcleaf_v = sorted([(wcleaf_p[val], val) for val in wcleaf_p])[-1][1]

        basic_p = leaf.get_basic_level().predict_probs()[attr]
        basic_v = sorted([(basic_p[val], val) for val in basic_p])[-1][1]
        best_p = leaf.get_best_level(instance).predict_probs()[attr]
        best_v = sorted([(best_p[val], val) for val in best_p])[-1][1]

        full_p = tree.root.predict_probs_mixture(instance, 100)[attr]
        full_v = sorted([(full_p[val], val) for val in full_p])[-1][1]

        leaf_acc.append(int(leaf_v == label[attr]))
        mix_acc.append(int(wleaf_v == label[attr]))
        mix_leaves_acc.append(int(wcleaf_v == label[attr]))
        full_acc.append(int(full_v == label[attr]))
        basic_acc.append(int(basic_v == label[attr]))
        best_acc.append(int(best_v == label[attr]))

        # # if leaf_acc[-1] == 0:
        # print("Predicted: ", leaf_v)
        # print("Actual: ", label[attr])
        # viz_path.plot_frontier_paths(instance, tree, attr, label[attr], -22)
        # raise Exception("BEEP")

print("leaf acc: ", sum(leaf_acc)/len(leaf_acc))
print("path mix acc: ", sum(mix_acc)/len(mix_acc))
print("leaves mix acc: ", sum(mix_leaves_acc)/len(mix_leaves_acc))
print("full acc: ", sum(full_acc)/len(full_acc))
print("basic acc: ", sum(basic_acc)/len(basic_acc))
print("best acc: ", sum(best_acc)/len(best_acc))

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









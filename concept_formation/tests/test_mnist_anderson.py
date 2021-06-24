from pprint import pprint

from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from concept_formation.anderson import RadicalIncremental

digits = load_digits(n_class=2)

imgs = digits.images  # [:200, :, :]
labels = digits.target  # [:200]

# data = [(imgs[i, :, :], label) for i, label in enumerate(digits.target)]

errors = []

# runs = 20
# run_length = 50

runs = 1
run_length = 100

for r in range(runs):
    print()
    print("#############")
    print("RUN {} of {}".format(r, runs))
    print("#############")

    sss = StratifiedShuffleSplit(n_splits=1, train_size=run_length)
    for train_index, _ in sss.split(imgs, labels):
        X = imgs[train_index]
        # X = np.reshape(X, (X.shape[0], -1))
        # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        # X = X[:, ~np.isnan(X).any(axis=0)]
        # X = np.nan_to_num(X)

        print(X[0])
        y = labels[train_index]

    print(y)

    model = RadicalIncremental()

    pred = []

    for i, img in enumerate(X):
        print("loading {} of {}.".format(i, len(X)))
        # inst = {"{}".format(idx): v for idx, v in enumerate(img)}
        inst = {"{},{}".format(rowi, coli): v for rowi, row in enumerate(img)
                for coli, v in enumerate(row)}
        # inst['image_data'] = img
        # inst['test'] = random()
        # inst['_label'] = "{}".format(labels[i])

        p = model.predict(inst, 'label')

        if p is None:
            p = 0

        print("pred={}, actual={}".format(p, y[i]))
        pred.append(int(p))

        inst['label'] = "{}".format(y[i])

        model.ifit(inst)
        print('# clusters', len(model.clusters))

    print("Predicted")
    print([label for label in pred])
    print('Actual')
    print([label for label in y])
    print("ERROR")
    error = [int(pred[i] != v) for i, v in enumerate(y)]
    errors.append(error)
    print(error)
    print(sum(error) / len(y))

print("overall errors")
print(errors)

errors = np.array(errors)

plt.plot(np.mean(errors, 0))
plt.show()

for cluster in model.clusters:
    pprint(cluster.av)
    print()

from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit

from concept_formation.convo_cobweb import ConvoCobwebTree

# from concept_formation.visualize import visualize

digits = load_digits(n_class=10)

imgs = digits.images  # [:200, :, :]
labels = digits.target  # [:200]

# data = [(imgs[i, :, :], label) for i, label in enumerate(digits.target)]

errors = []
runs = 20

for r in range(runs):
    print()
    print("#############")
    print("RUN {} of {}".format(r, runs))
    print("#############")

    sss = StratifiedShuffleSplit(n_splits=1, train_size=50)
    for train_index, _ in sss.split(imgs, labels):
        X = imgs[train_index]
        y = labels[train_index]

    print(y)

    tree = ConvoCobwebTree(filter_size=4)

    pred = []

    for i, img in enumerate(X):
        print("loading {} of {}.".format(i, len(X)))
        inst = {}
        inst['image_data'] = img
        # inst['test'] = random()
        # inst['_label'] = "{}".format(labels[i])
        curr = tree.categorize(inst)

        while curr and '_label' not in curr.av_counts:
            curr = curr.parent

        if curr:
            p = curr.predict('_label')
        else:
            p = 0
        if p is None:
            p = 0
        print("pred={}, actual={}".format(p, y[i]))
        pred.append(int(p))

        inst['_label'] = "{}".format(y[i])

        tree.ifit(inst)

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


# visualize(tree)

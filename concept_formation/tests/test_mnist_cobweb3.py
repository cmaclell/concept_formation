from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit

from concept_formation.cobweb3 import Cobweb3Tree

from concept_formation.visualize import visualize

# digits = load_digits(n_class=10)

# imgs = digits.images  # [:200, :, :]
# labels = digits.target  # [:200]
# labels = np.array([str(label) for label in labels])

imgs, labels = fetch_openml('mnist_784', version=1, return_X_y=True,
                            as_frame=False)
mask = np.isin(labels, ["0", "1", "2"])
imgs = imgs[mask]
labels = labels[mask]
imgs = imgs.reshape(-1, 28, 28)

# data = [(imgs[i, :, :], label) for i, label in enumerate(digits.target)]

errors = []

# runs = 20
# run_length = 50

runs = 1
run_length = 20

for r in range(runs):
    print()
    print("#############")
    print("RUN {} of {}".format(r, runs))
    print("#############")

    sss = StratifiedShuffleSplit(n_splits=1, train_size=run_length)
    for train_index, _ in sss.split(imgs, labels):
        X = imgs[train_index]
        y = labels[train_index]

    print(y)

    tree = Cobweb3Tree()

    pred = []

    for i, img in enumerate(X):
        print("loading {} of {}.".format(i, len(X)))
        inst = {"{},{}".format(rowi, coli): v for rowi, row in enumerate(img)
                for coli, v in enumerate(row)}
        # inst['image_data'] = img
        # inst['test'] = random()
        # inst['_label'] = "{}".format(labels[i])
        curr = tree.categorize(inst)

        while curr and 'label' not in curr.av_counts:
            curr = curr.parent

        if curr:
            p = curr.predict('label')
        else:
            p = 0 if isinstance(labels[0], int) else "0"
        if p is None:
            p = 0 if isinstance(labels[0], int) else "0"
        print("pred={}, actual={}".format(p, y[i]))
        pred.append(p)

        inst['label'] = "{}".format(y[i])

        tree.ifit(inst)

        print('# clusters', len(tree.root.children))

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

visualize(tree)
